"""Engine worker process: reads JSON commands from stdin, writes results to stdout.

This module runs as a separate process spawned by ProcessingController.
"""

import json
import logging
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np


def serialize_message(msg: dict) -> str:
    """Serialize a message to JSON line."""
    return json.dumps(msg, separators=(",", ":"))


def deserialize_message(line: str) -> dict:
    """Deserialize a JSON line to message dict."""
    return json.loads(line.strip())


def send_message(msg: dict):
    """Write a JSON message to stdout."""
    sys.stdout.write(serialize_message(msg) + "\n")
    sys.stdout.flush()


def main():
    """Main loop: read commands from stdin, process, write results to stdout."""
    # Import heavy deps only when running as worker
    from upscaler.config import ensure_dirs, LOG_FILE, load_settings

    # Worker logging: file only, no stderr (stderr pipe can deadlock on Windows)
    ensure_dirs()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
    )
    # Redirect stderr to devnull so third-party libs (torch, etc.) don't fill the pipe
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

    log = logging.getLogger("upscaler.worker")

    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor, PipelineCancelled
    from upscaler.utils.image_io import read_image, write_image

    registry = PluginRegistry()
    registry.discover_builtin()
    settings = load_settings()
    executor = PipelineExecutor(registry)

    cancel_events: dict[str, threading.Event] = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = deserialize_message(line)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "shutdown":
            break

        elif msg_type == "cancel":
            job_id = msg.get("job_id", "")
            if job_id in cancel_events:
                cancel_events[job_id].set()
            send_message({"type": "cancelled", "job_id": job_id})

        elif msg_type == "start_pipeline":
            job_id = msg.get("job_id", "unknown")
            image_path = msg.get("image_path", "")
            config = msg.get("config", {})
            cancel_event = threading.Event()
            cancel_events[job_id] = cancel_event

            try:
                image, meta = read_image(Path(image_path))

                def progress_cb(stage, pct, message):
                    send_message({
                        "type": "progress",
                        "job_id": job_id,
                        "stage": stage,
                        "percent": pct,
                        "message": message,
                    })

                # Use device from config (sent per-job by UI), fallback to settings
                device = config.get("device") or settings.get("gpu_device", "auto")
                config.setdefault(
                    "prefer_gpu_denoise",
                    settings.get("prefer_gpu_denoise", True),
                )
                result = executor.execute(
                    image, config, meta,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    device=device,
                )

                # Write result image to temp file
                output_path = Path(image_path).parent / f"{job_id}_result.png"
                write_image(result["image"], output_path)

                # Блок G: write each per-step variant snapshot to its own PNG
                # so the UI can display intermediate results.
                var_msgs = []
                for i, variant in enumerate(result.get("variants", [])):
                    var_path = Path(image_path).parent / f"{job_id}_var{i}.png"
                    write_image(variant["image"], var_path)
                    var_msgs.append({"label": variant["label"], "path": str(var_path)})

                send_message({
                    "type": "result",
                    "job_id": job_id,
                    "output_path": str(output_path),
                    "metrics": {
                        "brisque": result["metrics"].get("brisque", 0),
                        "niqe": result["metrics"].get("niqe", 0),
                        "histogram_similarity": result["metrics"].get("histogram_similarity", 0),
                    },
                    "variants": var_msgs,
                })

            except PipelineCancelled:
                log.info(f"Job {job_id} cancelled")
                send_message({"type": "cancelled", "job_id": job_id})
            except Exception as e:
                log.error(f"Job {job_id} failed: {e}", exc_info=True)
                send_message({
                    "type": "error",
                    "job_id": job_id,
                    "stage": "UNKNOWN",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "recoverable": True,
                })
            finally:
                cancel_events.pop(job_id, None)


if __name__ == "__main__":
    main()
