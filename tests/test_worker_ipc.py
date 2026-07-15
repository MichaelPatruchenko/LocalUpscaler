import json
import pytest


class TestMessageProtocol:
    def test_serialize_start_pipeline(self):
        from upscaler.engine.worker import serialize_message, deserialize_message
        msg = {"type": "start_pipeline", "image_path": "/tmp/test.png", "config": {}, "job_id": "abc"}
        line = serialize_message(msg)
        parsed = deserialize_message(line)
        assert parsed["type"] == "start_pipeline"
        assert parsed["job_id"] == "abc"

    def test_serialize_progress(self):
        from upscaler.engine.worker import serialize_message, deserialize_message
        msg = {"type": "progress", "job_id": "abc", "stage": "SCALE", "percent": 50, "message": "Working..."}
        line = serialize_message(msg)
        parsed = deserialize_message(line)
        assert parsed["percent"] == 50

    def test_serialize_error(self):
        from upscaler.engine.worker import serialize_message, deserialize_message
        msg = {"type": "error", "job_id": "abc", "stage": "SCALE", "error": "OOM", "recoverable": True}
        line = serialize_message(msg)
        parsed = deserialize_message(line)
        assert parsed["recoverable"] is True

    def test_serialize_result_with_variants(self):
        """Блок G: result-сообщение несёт список вариантов {label, path}."""
        from upscaler.engine.worker import serialize_message, deserialize_message
        msg = {
            "type": "result", "job_id": "abc",
            "output_path": "/tmp/abc_result.png",
            "metrics": {"brisque": 0, "niqe": 0, "histogram_similarity": 0},
            "variants": [
                {"label": "Резкость", "path": "/tmp/abc_var0.png"},
                {"label": "Апскейл", "path": "/tmp/abc_var1.png"},
            ],
        }
        line = serialize_message(msg)
        parsed = deserialize_message(line)
        assert parsed["type"] == "result"
        assert isinstance(parsed["variants"], list)
        assert parsed["variants"][0]["label"] == "Резкость"
        assert parsed["variants"][0]["path"] == "/tmp/abc_var0.png"
        assert parsed["variants"][1]["path"] == "/tmp/abc_var1.png"
