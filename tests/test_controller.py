import pytest
from unittest.mock import MagicMock, patch


class TestProcessingController:
    def test_can_instantiate(self):
        from upscaler.engine.controller import ProcessingController
        ctrl = ProcessingController()
        assert ctrl is not None

    def test_is_running_initially_false(self):
        from upscaler.engine.controller import ProcessingController
        ctrl = ProcessingController()
        assert ctrl.is_running() is False
