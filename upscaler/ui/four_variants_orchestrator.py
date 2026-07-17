"""Оркестратор сессии «4 варианта обработки»: раунды с выбором пользователя.

Раунд = до четырёх кандидатов (по направлениям из engine/four_variants.py),
обработанных последовательно одним воркером. После раунда MainWindow открывает
галерею; выбранное пользователем изображение становится входом следующего
раунда (при включённом ИИ и оставшихся итерациях). Итеративная доработка
работает обособленно для каждого направления: evaluate/refine выполняются со
стилевой директивой направления.

Оркестратор не трогает диск/контроллер/LLM напрямую — все внешние операции
инъецируются вызовами, поэтому класс полностью юнит-тестируется синхронными
фейками:

    submit(image, config)                    -> запись tmp + submit_job
    refine(image, analysis, config, directive, cb)   -> cb(refined_config)
    evaluate(image, analysis, directive, cb)         -> cb(verdict)
    analyze(image)                           -> dict (SourceAnalyzer)

События завершения задания MainWindow транслирует в handle_complete /
handle_error / handle_cancelled.
"""
import logging

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class FourVariantsOrchestrator(QObject):
    # (key, kwargs) для MainWindow._set_status — строки остаются в i18n.
    status_changed = Signal(str, dict)
    # Раунд готов: список кандидатов со status="done" (галерею открывает UI).
    round_ready = Signal(list)
    # Сессия завершена (выбор сделан/итерации исчерпаны/все satisfied).
    session_done = Signal()
    # Раунд 1 не дал ни одного кандидата.
    session_failed = Signal()
    # Сессия прервана отменой задания.
    cancelled = Signal()

    def __init__(self, variants: list[dict], source_image, analysis: dict,
                 submit, analyze, refine=None, evaluate=None,
                 ai_enabled: bool = False, max_iter: int = 1, parent=None):
        super().__init__(parent)
        self._directions = [
            {"id": v["id"], "name_key": v["name_key"],
             "style_directive": v["style_directive"]}
            for v in variants
        ]
        self._initial_configs = {v["id"]: v["config"] for v in variants}
        self._submit = submit
        self._analyze = analyze
        self._refine = refine
        self._evaluate = evaluate
        self._ai = bool(ai_enabled) and refine is not None \
            and evaluate is not None
        self._max_iter = max(1, int(max_iter))

        self.iteration = 1
        self.current_image = source_image
        self._analysis = analysis      # анализ current_image текущего раунда
        self._candidates: list[dict] = []
        self._idx = 0                  # позиция в текущем раунде
        self._active = True
        self._awaiting_choice = False

    # ─── Публичный жизненный цикл ───

    def start(self) -> None:
        """Запустить раунд 1 по исходному изображению."""
        self._start_round([
            dict(d, config=self._initial_configs[d["id"]], result_image=None,
                 metrics=None, status="pending")
            for d in self._directions
        ], evaluate_first=False)

    def choose(self, index: int) -> None:
        """Пользователь выбрал кандидата (индекс в списке из round_ready)."""
        if not (self._active and self._awaiting_choice):
            return
        if not (0 <= index < len(self._candidates)):
            return
        chosen = self._candidates[index]
        if chosen["status"] != "done":
            return
        self._awaiting_choice = False
        self.current_image = chosen["result_image"]
        if not self._ai or self.iteration >= self._max_iter:
            self._finish()
            return
        self.iteration += 1
        self._analysis = None  # пересчитывается лениво в _start_round
        self._start_round([
            dict(d, config=None, result_image=None, metrics=None,
                 status="pending")
            for d in self._directions
        ], evaluate_first=True)

    def finish(self) -> None:
        """Завершить сессию (галерея закрыта без выбора)."""
        if self._active:
            self._finish()

    # ─── События контроллера (транслирует MainWindow) ───

    def handle_complete(self, result_image, metrics: dict) -> None:
        cand = self._current_candidate()
        if cand is None:
            return
        cand["result_image"] = result_image
        cand["metrics"] = dict(metrics or {})
        cand["status"] = "done"
        self._advance()

    def handle_error(self, stage: str, error: str) -> None:
        cand = self._current_candidate()
        if cand is None:
            return
        log.warning("Variant '%s' failed at %s: %s", cand["id"], stage, error)
        cand["status"] = "failed"
        self._advance()

    def handle_cancelled(self) -> None:
        if not self._active:
            return
        self._active = False
        self.cancelled.emit()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def awaiting_choice(self) -> bool:
        return self._awaiting_choice

    # ─── Внутреннее ───

    def _current_candidate(self):
        if not self._active or self._awaiting_choice:
            return None
        if 0 <= self._idx < len(self._candidates):
            cand = self._candidates[self._idx]
            return cand if cand["status"] == "running" else None
        return None

    def _start_round(self, candidates: list[dict], evaluate_first: bool) -> None:
        self._candidates = candidates
        self._idx = -1
        self._round_evaluates = evaluate_first
        if evaluate_first and self._analysis is None:
            self._analysis = self._analyze(self.current_image)
        self._advance()

    def _advance(self) -> None:
        """Перейти к следующему кандидату раунда либо закрыть раунд."""
        if not self._active:
            return
        self._idx += 1
        if self._idx >= len(self._candidates):
            self._close_round()
            return
        cand = self._candidates[self._idx]
        if self._round_evaluates:
            self._evaluate_candidate(cand)
        else:
            self._launch_candidate(cand)

    def _launch_candidate(self, cand: dict) -> None:
        """Первичный запуск кандидата раунда 1 (refine при ИИ + submit)."""
        cand["status"] = "running"
        n = self._idx + 1
        if self._ai:
            self.status_changed.emit("status.variant_refining", {
                "iter": self.iteration, "max": self._max_iter,
                "name_key": cand["name_key"]})

            def _refined(config):
                if not self._active:
                    return
                cand["config"] = config
                self._submit_candidate(cand, n)

            self._refine(self.current_image, self._analysis, cand["config"],
                         cand["style_directive"], _refined)
        else:
            self._submit_candidate(cand, n)

    def _evaluate_candidate(self, cand: dict) -> None:
        """Раунд 2+: оценка выбранного изображения в характере направления."""
        from upscaler.engine.llm_advisor import should_continue_refinement
        cand["status"] = "running"
        n = self._idx + 1
        self.status_changed.emit("status.variant_evaluating", {
            "iter": self.iteration, "max": self._max_iter,
            "name_key": cand["name_key"]})

        def _verdict(verdict):
            if not self._active:
                return
            cont = should_continue_refinement(
                self.iteration - 1, self._max_iter,
                (verdict or {}).get("satisfied", True))
            config = (verdict or {}).get("config")
            if cont and config:
                cand["config"] = config
                self._submit_candidate(cand, n)
            else:
                cand["status"] = "satisfied"
                self._advance()

        self._evaluate(self.current_image, self._analysis,
                       cand["style_directive"], _verdict)

    def _submit_candidate(self, cand: dict, n: int) -> None:
        self.status_changed.emit("status.variant_processing", {
            "iter": self.iteration, "max": self._max_iter,
            "n": n, "total": len(self._candidates),
            "name_key": cand["name_key"]})
        self._submit(self.current_image, cand["config"])

    def _close_round(self) -> None:
        done = [c for c in self._candidates if c["status"] == "done"]
        if done:
            self._awaiting_choice = True
            self.round_ready.emit(list(self._candidates))
            return
        # Ни одного кандидата: в раунде 1 это провал сессии, в раундах 2+
        # либо все satisfied (изображение финально), либо все с ошибкой —
        # остаёмся на выбранном изображении.
        if self.iteration == 1:
            self._active = False
            self.session_failed.emit()
        else:
            self._finish()

    def _finish(self) -> None:
        self._active = False
        self._awaiting_choice = False
        self.session_done.emit()
