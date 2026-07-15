"""Batch processing: sequential file-by-file execution with error handling."""

import logging
from pathlib import Path
from typing import Callable

import torch

from upscaler.engine.pipeline import PipelineExecutor
from upscaler.plugins.registry import PluginRegistry
from upscaler.utils.image_io import read_image, write_image

log = logging.getLogger(__name__)


class BatchRunner:
    """Runs pipeline on multiple files sequentially."""

    def __init__(self, registry: PluginRegistry):
        self.executor = PipelineExecutor(registry)

    def run(
        self,
        files: list[Path],
        output_dir: str,
        config: dict,
        device: str = "cpu",
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Process all files. Returns summary dict."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        succeeded = 0
        failed = 0
        errors = []

        for i, file_path in enumerate(files):
            filename = file_path.name
            if progress_cb:
                progress_cb(i + 1, len(files), filename)

            try:
                image, meta = read_image(file_path)
                result = self.executor.execute(image, config, meta, device=device)

                stem = file_path.stem
                ext = file_path.suffix or ".png"
                out_path = out / f"{stem}_upscaled{ext}"
                write_image(result["image"], out_path)
                succeeded += 1

            except Exception as e:
                log.error(f"Batch: failed on {filename}: {e}")
                failed += 1
                errors.append({"file": str(file_path), "error": str(e)})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"succeeded": succeeded, "failed": failed, "errors": errors}
