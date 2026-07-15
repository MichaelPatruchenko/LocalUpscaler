import numpy as np
import pytest
from pathlib import Path


class TestBatchRunner:
    def _make_test_images(self, tmp_path, count=3):
        import cv2
        tmp_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(count):
            img = np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths

    def test_batch_processes_all_files(self, tmp_path):
        from upscaler.engine.batch_runner import BatchRunner
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        files = self._make_test_images(tmp_path / "input")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = {"scale": 2, "enhance_only": False, "upscale": {"plugin": "Lanczos"}}
        runner = BatchRunner(reg)
        summary = runner.run(files, str(output_dir), config)
        assert summary["succeeded"] == 3
        assert summary["failed"] == 0
        assert len(list(output_dir.glob("*.png"))) == 3

    def test_batch_output_naming(self, tmp_path):
        from upscaler.engine.batch_runner import BatchRunner
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        files = self._make_test_images(tmp_path / "input", count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = {"scale": 2, "enhance_only": False, "upscale": {"plugin": "Lanczos"}}
        runner = BatchRunner(reg)
        runner.run(files, str(output_dir), config)
        expected = output_dir / "img_0_upscaled.png"
        assert expected.exists()

    def test_batch_continues_on_error(self, tmp_path):
        from upscaler.engine.batch_runner import BatchRunner
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        files = self._make_test_images(tmp_path / "input", count=2)
        files.insert(1, tmp_path / "input" / "corrupt.png")
        (tmp_path / "input" / "corrupt.png").write_bytes(b"not an image")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        config = {"scale": 2, "enhance_only": False, "upscale": {"plugin": "Lanczos"}}
        runner = BatchRunner(reg)
        summary = runner.run(files, str(output_dir), config)
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1
        assert len(summary["errors"]) == 1

    def test_progress_callback(self, tmp_path):
        from upscaler.engine.batch_runner import BatchRunner
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        files = self._make_test_images(tmp_path / "input", count=2)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        progress_calls = []
        def on_progress(current, total, filename):
            progress_calls.append((current, total, filename))
        config = {"scale": 2, "enhance_only": False, "upscale": {"plugin": "Lanczos"}}
        runner = BatchRunner(reg)
        runner.run(files, str(output_dir), config, progress_cb=on_progress)
        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 2
