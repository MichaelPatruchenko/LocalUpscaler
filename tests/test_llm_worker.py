import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.llm_worker import LLMRefineThread

_app = QApplication.instance() or QApplication([])

_BASE = {"scale": 4, "post": {"sharpen": 0.2}}


class _FakeAdvisor:
    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises

    def refine(self, image, analysis, base_config, allow_deblur=True,
               allow_icedit=True, allow_face=True, blend_enabled=False):
        self.allow_deblur = allow_deblur
        self.allow_icedit = allow_icedit
        self.blend_enabled = blend_enabled
        if self._raises:
            raise RuntimeError("boom")
        return self._result


def test_thread_emits_refined_config():
    captured = {}
    advisor = _FakeAdvisor(result={"scale": 2})
    t = LLMRefineThread(advisor, None, {}, _BASE)
    t.done.connect(lambda cfg: captured.update(cfg))
    t.run()  # run synchronously (direct connection fires the slot inline)
    assert captured == {"scale": 2}


def test_thread_forwards_allow_deblur():
    advisor = _FakeAdvisor(result={"scale": 2})
    t = LLMRefineThread(advisor, None, {}, _BASE, allow_deblur=False)
    t.run()
    assert advisor.allow_deblur is False


def test_thread_falls_back_to_base_on_error():
    captured = {}
    advisor = _FakeAdvisor(raises=True)
    t = LLMRefineThread(advisor, None, {}, _BASE)
    t.done.connect(lambda cfg: captured.update(cfg))
    t.run()
    assert captured == _BASE


def test_refine_thread_forwards_allow_icedit():
    from upscaler.ui.llm_worker import LLMRefineThread

    captured = {}

    class _Advisor:
        def refine(self, image, analysis, base_config, allow_deblur=True,
                   allow_icedit=True, allow_face=True, blend_enabled=False):
            captured["allow_icedit"] = allow_icedit
            captured["allow_deblur"] = allow_deblur
            return base_config

    t = LLMRefineThread(_Advisor(), None, {}, {"scale": 4},
                        allow_deblur=False, allow_icedit=False)
    t.run()
    assert captured["allow_icedit"] is False
    assert captured["allow_deblur"] is False


def test_evaluate_thread_emits_verdict():
    from upscaler.ui.llm_worker import LLMEvaluateThread

    captured = {}

    class _Advisor:
        def evaluate(self, image, analysis, allow_deblur=True,
                     allow_icedit=True, allow_face=True):
            captured["flags"] = (allow_deblur, allow_icedit, allow_face)
            return {"satisfied": False, "config": {"enhance_only": True}}

    results = []
    t = LLMEvaluateThread(_Advisor(), None, {}, allow_deblur=False,
                          allow_icedit=True, allow_face=False)
    t.done.connect(lambda d: results.append(d))
    t.run()
    assert captured["flags"] == (False, True, False)
    assert results and results[0]["satisfied"] is False


def test_evaluate_thread_stops_on_error():
    from upscaler.ui.llm_worker import LLMEvaluateThread

    class _Advisor:
        def evaluate(self, *a, **k):
            raise RuntimeError("boom")

    results = []
    t = LLMEvaluateThread(_Advisor(), None, {})
    t.done.connect(lambda d: results.append(d))
    t.run()
    assert results and results[0] == {"satisfied": True, "config": None}


def test_refine_thread_forwards_allow_face():
    captured = {}

    class _Advisor:
        def refine(self, image, analysis, base, allow_deblur=True,
                   allow_icedit=True, allow_face=True, blend_enabled=False):
            captured["allow_face"] = allow_face
            return base

    from upscaler.ui.llm_worker import LLMRefineThread
    t = LLMRefineThread(_Advisor(), None, {}, {"x": 1}, allow_face=False)
    t.run()
    assert captured["allow_face"] is False


def test_refine_thread_forwards_blend_enabled():
    captured = {}

    class _Advisor:
        def refine(self, image, analysis, base, allow_deblur=True,
                   allow_icedit=True, allow_face=True, blend_enabled=False):
            captured["blend_enabled"] = blend_enabled
            return base

    from upscaler.ui.llm_worker import LLMRefineThread
    t = LLMRefineThread(_Advisor(), None, {}, {"x": 1}, blend_enabled=True)
    t.run()
    assert captured["blend_enabled"] is True
