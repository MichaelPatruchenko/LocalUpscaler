"""ProcessingController: manages the worker process from the UI thread."""

import json
import logging
import uuid
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QThread, Signal, Slot


class ProcessingController(QObject):
    """Manages the engine worker process via QProcess.

    Signals:
        progress_updated(job_id, stage, percent, message)
        variants_ready(job_id, variants) — emitted before processing_complete;
            variants = [{"label": str, "path": str}, ...]
        processing_complete(job_id, output_path, metrics)
        processing_error(job_id, stage, error, recoverable)
        processing_cancelled(job_id)
    """

    progress_updated = Signal(str, str, int, str)
    variants_ready = Signal(str, list)
    processing_complete = Signal(str, str, dict)
    processing_error = Signal(str, str, str, bool)
    processing_cancelled = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._running = False
        self._current_job_id: str | None = None
        self._pending_msg: dict | None = None

    def start_engine(self):
        """Launch the worker process."""
        if self._process is not None:
            return
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.started.connect(self._on_started)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        if getattr(sys, "frozen", False):
            # PyInstaller one-file build: re-launch ourselves in worker mode
            self._process.start(sys.executable, ["--worker"])
        else:
            self._process.start(sys.executable, ["-m", "upscaler.engine.worker"])

    def stop_engine(self):
        """Gracefully shut down the worker process."""
        if self._process:
            self._send({"type": "shutdown"})
            self._process.waitForFinished(5000)
            if self._process.state() != QProcess.ProcessState.NotRunning:
                self._process.kill()
            self._process = None
            self._running = False

    def submit_job(self, image_path: str, config: dict) -> str:
        """Submit a processing job. Returns job_id."""
        if not self._process:
            self.start_engine()
        job_id = uuid.uuid4().hex[:12]
        self._current_job_id = job_id
        self._running = True
        self._pending_msg = {
            "type": "start_pipeline",
            "image_path": image_path,
            "config": config,
            "job_id": job_id,
        }
        # If process is already running, send immediately;
        # otherwise _on_started will flush the pending message.
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            self._send(self._pending_msg)
            self._pending_msg = None
        return job_id

    def cancel_job(self, job_id: str):
        """Cancel an in-progress job."""
        self._send({"type": "cancel", "job_id": job_id})

    def is_running(self) -> bool:
        return self._running

    def _send(self, msg: dict):
        """Send a JSON message to the worker process stdin."""
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            line = json.dumps(msg, separators=(",", ":")) + "\n"
            self._process.write(line.encode("utf-8"))

    @Slot()
    def _on_started(self):
        """Worker process has started — flush any pending message."""
        if self._pending_msg:
            self._send(self._pending_msg)
            self._pending_msg = None

    @Slot()
    def _on_stdout(self):
        """Read and parse messages from worker stdout."""
        while self._process and self._process.canReadLine():
            line = self._process.readLine().data().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            job_id = msg.get("job_id", "")

            if msg_type == "progress":
                self.progress_updated.emit(
                    job_id, msg.get("stage", ""), msg.get("percent", 0), msg.get("message", "")
                )
            elif msg_type == "result":
                self._running = False
                self.variants_ready.emit(job_id, msg.get("variants", []))
                self.processing_complete.emit(
                    job_id, msg.get("output_path", ""), msg.get("metrics", {})
                )
            elif msg_type == "error":
                self._running = False
                self.processing_error.emit(
                    job_id, msg.get("stage", ""), msg.get("error", ""), msg.get("recoverable", False)
                )
            elif msg_type == "cancelled":
                self._running = False
                self.processing_cancelled.emit(job_id)

    @Slot()
    def _on_stderr(self):
        """Drain stderr to prevent pipe buffer deadlock."""
        if self._process:
            data = self._process.readAllStandardError().data()
            try:
                text = data.decode("utf-8", errors="replace").strip()
            except Exception:
                text = repr(data)
            if text:
                logging.getLogger("upscaler.worker.stderr").debug(text)

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code, exit_status):
        """Handle worker process exit."""
        self._running = False
        if exit_status == QProcess.ExitStatus.CrashExit:
            if self._current_job_id:
                self.processing_error.emit(
                    self._current_job_id, "ENGINE", "Worker process crashed", True
                )
        self._process = None
