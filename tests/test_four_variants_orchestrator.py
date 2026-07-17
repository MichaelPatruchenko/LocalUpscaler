"""FourVariantsOrchestrator: раунды, выбор на каждой итерации, изоляция
доработки по направлениям (спек 2026-07-17, Часть 2)."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from upscaler.ui.four_variants_orchestrator import FourVariantsOrchestrator

_app = QApplication.instance() or QApplication([])


def _variants():
    from upscaler.engine.four_variants import VARIANT_DIRECTIONS
    return [
        {"id": d["id"], "name_key": d["name_key"],
         "style_directive": d["style_directive"],
         "config": {"scale": 4, "tag": d["id"]}}
        for d in VARIANT_DIRECTIONS
    ]


def _img(v=0):
    return np.full((8, 8, 3), v, np.uint8)


class _Harness:
    """Синхронные фейки submit/refine/evaluate/analyze + журнал вызовов."""

    def __init__(self, ai=False, max_iter=1, verdicts=None,
                 fail_ids=frozenset()):
        self.submitted = []       # (image, config)
        self.refined = []         # directives
        self.evaluated = []       # directives
        self.rounds = []          # списки кандидатов из round_ready
        self.done = 0
        self.failed = 0
        self.cancelled = 0
        self._fail_ids = set(fail_ids)
        # verdicts: dict id -> список вердиктов по раундам (2-й раунд и далее)
        self._verdicts = verdicts or {}
        self._result_counter = 0

        self.orch = FourVariantsOrchestrator(
            _variants(), _img(0), {"a": 1},
            submit=self._submit, analyze=lambda img: {"re": 1},
            refine=self._refine if ai else None,
            evaluate=self._evaluate if ai else None,
            ai_enabled=ai, max_iter=max_iter)
        self.orch.round_ready.connect(self.rounds.append)
        self.orch.session_done.connect(self._on_done)
        self.orch.session_failed.connect(self._on_failed)
        self.orch.cancelled.connect(self._on_cancelled)

    def _on_done(self):
        self.done += 1

    def _on_failed(self):
        self.failed += 1

    def _on_cancelled(self):
        self.cancelled += 1

    def _submit(self, image, config):
        self.submitted.append((image, config))
        tag = config.get("tag", "")
        if tag in self._fail_ids:
            self.orch.handle_error("stage", "boom")
        else:
            self._result_counter += 1
            self.orch.handle_complete(_img(self._result_counter),
                                      {"brisque": 30.0 + self._result_counter})

    def _refine(self, image, analysis, config, directive, cb):
        self.refined.append(directive)
        cb(dict(config, refined=True))

    def _evaluate(self, image, analysis, directive, cb):
        self.evaluated.append(directive)
        word = directive.split("priority:")[1].split()[0].strip(".").lower()
        matched = None
        for v in _variants():
            if word in v["style_directive"].lower().split("priority:")[1]:
                matched = v["id"]
                break
        queue = self._verdicts.get(matched, [])
        verdict = queue.pop(0) if queue else {"satisfied": True, "config": None}
        cb(verdict)


# ─── Раунд 1 ───

def test_round1_runs_four_candidates_sequentially_ai_off():
    h = _Harness(ai=False, max_iter=3)
    h.orch.start()
    assert len(h.submitted) == 4
    assert [c["tag"] for _, c in h.submitted] == ["natural", "sharp",
                                                  "clean", "vivid"]
    # вход каждого кандидата — исходное изображение
    assert all((img == _img(0)).all() for img, _ in h.submitted)
    assert len(h.rounds) == 1
    assert [c["status"] for c in h.rounds[0]] == ["done"] * 4
    assert h.refined == [] and h.evaluated == []


def test_ai_off_single_round_choose_finishes():
    h = _Harness(ai=False, max_iter=3)
    h.orch.start()
    h.orch.choose(1)
    # ИИ off -> без переоценки, сессия из одного раунда
    assert h.done == 1
    assert not h.orch.active
    assert (h.orch.current_image == h.rounds[0][1]["result_image"]).all()


def test_round1_ai_on_refines_each_direction_with_its_directive():
    h = _Harness(ai=True, max_iter=1)
    h.orch.start()
    assert len(h.refined) == 4
    assert [d.split("priority:")[1].split()[0] for d in h.refined] == \
        ["NATURAL", "MAXIMUM", "CLEAN,", "VIVID,"]
    # refine дошёл до submit
    assert all(c.get("refined") for _, c in h.submitted)


def test_failed_candidate_does_not_break_round():
    h = _Harness(ai=False, fail_ids={"sharp"})
    h.orch.start()
    assert len(h.rounds) == 1
    statuses = {c["id"]: c["status"] for c in h.rounds[0]}
    assert statuses["sharp"] == "failed"
    assert [statuses[i] for i in ("natural", "clean", "vivid")] == ["done"] * 3


def test_all_failed_round1_emits_session_failed():
    h = _Harness(ai=False, fail_ids={"natural", "sharp", "clean", "vivid"})
    h.orch.start()
    assert h.failed == 1
    assert h.rounds == []
    assert not h.orch.active


# ─── Раунды 2+ (выбор на каждой итерации) ───

def test_choice_seeds_next_round_and_gallery_shown_each_iteration():
    verdicts = {vid: [{"satisfied": False, "config": {"enhance_only": True,
                                                      "tag": vid}}]
                for vid in ("natural", "sharp", "clean", "vivid")}
    h = _Harness(ai=True, max_iter=2, verdicts=verdicts)
    h.orch.start()
    assert len(h.rounds) == 1
    chosen_img = h.rounds[0][2]["result_image"]  # выбираем «clean»
    h.orch.choose(2)
    # раунд 2: 4 оценки со своими директивами, вход — выбранное изображение
    assert len(h.evaluated) == 4
    round2_inputs = [img for img, _ in h.submitted[4:]]
    assert all((img == chosen_img).all() for img in round2_inputs)
    assert len(h.rounds) == 2
    # выбор в последнем раунде завершает сессию (max_iter=2)
    h.orch.choose(0)
    assert h.done == 1


def test_satisfied_directions_do_not_produce_candidates():
    verdicts = {"sharp": [{"satisfied": False,
                           "config": {"enhance_only": True, "tag": "sharp"}}]}
    h = _Harness(ai=True, max_iter=3, verdicts=verdicts)
    h.orch.start()
    h.orch.choose(0)
    statuses = {c["id"]: c["status"] for c in h.rounds[1]}
    assert statuses["sharp"] == "done"
    assert [statuses[i] for i in ("natural", "clean", "vivid")] == \
        ["satisfied"] * 3
    # submit был только у sharp (4 в раунде 1 + 1 во втором)
    assert len(h.submitted) == 5


def test_all_satisfied_ends_session_without_gallery():
    h = _Harness(ai=True, max_iter=3)  # все вердикты satisfied по умолчанию
    h.orch.start()
    h.orch.choose(0)
    assert len(h.rounds) == 1  # второй галереи нет
    assert h.done == 1
    assert not h.orch.active


def test_iterations_capped_by_max_iter():
    verdicts = {vid: [{"satisfied": False, "config": {"enhance_only": True}},
                      {"satisfied": False, "config": {"enhance_only": True}}]
                for vid in ("natural", "sharp", "clean", "vivid")}
    h = _Harness(ai=True, max_iter=2, verdicts=verdicts)
    h.orch.start()
    h.orch.choose(0)
    h.orch.choose(0)
    # раунда 3 нет, несмотря на "не satisfied" вердикты
    assert len(h.rounds) == 2
    assert h.done == 1


def test_choose_failed_candidate_ignored():
    h = _Harness(ai=False, fail_ids={"sharp"})
    h.orch.start()
    h.orch.choose(1)  # sharp -> failed, выбор игнорируется
    assert h.orch.awaiting_choice
    h.orch.choose(0)
    assert h.done == 1


def test_finish_without_choice_keeps_session_closed():
    h = _Harness(ai=True, max_iter=3)
    h.orch.start()
    h.orch.finish()
    assert h.done == 1
    assert not h.orch.active


def test_cancel_aborts_session():
    submitted = []

    def submit(image, config):
        submitted.append(config)  # задание «повисло» — эмулируем отмену

    orch = FourVariantsOrchestrator(
        _variants(), _img(), {}, submit=submit, analyze=lambda i: {},
        ai_enabled=False, max_iter=1)
    got = []
    orch.cancelled.connect(lambda: got.append(True))
    orch.start()
    orch.handle_cancelled()
    assert got == [True]
    assert not orch.active
    assert len(submitted) == 1  # второй кандидат не запускался
