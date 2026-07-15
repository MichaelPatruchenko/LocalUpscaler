"""Pipeline executor: analyze > [ordered processing steps] > validate.

The processing steps between analysis and validation run in a configurable
order. ``config["order"]`` (when present) is a list of step tokens that defines
the exact sequence; any configured step omitted from it is appended in its
default position so nothing is silently dropped. Without ``order`` the legacy
fixed sequence is used, so existing configs behave identically.
"""

import logging
import threading
from typing import Callable, Optional
import numpy as np
from upscaler.engine.analyzer import SourceAnalyzer
from upscaler.engine.validator import ResultValidator
from upscaler.plugins.registry import PluginRegistry

log = logging.getLogger(__name__)

# Токен шага -> i18n-ключ (slug) его подписи. Значения — НЕ текст, а ключи
# таблиц _STRINGS["ru"/"en"] в upscaler/ui/i18n.py (step.<token>); единый
# источник для UI-списка «Порядок обработки» и describe(). Используйте
# step_label(token) для получения переведённого текста.
STEP_LABELS = {
    "predownscale": "step.predownscale",
    "denoise": "step.denoise", "deblur": "step.deblur",
    "icedit": "step.icedit",
    "auto_color": "step.auto_color", "auto_tone": "step.auto_tone",
    "auto_contrast": "step.auto_contrast",
    "brightness": "step.brightness", "contrast": "step.contrast",
    "saturation": "step.saturation", "sharpness": "step.sharpness",
    "adjust": "step.adjust",
    "upscale": "step.upscale", "face_restore": "step.face_restore",
    "refocus": "step.refocus", "sharpen": "step.sharpen",
    "colorize": "step.colorize", "blend": "step.blend",
    "auto_levels": "step.auto_levels", "shadows_highlights": "step.shadows_highlights",
    "clarity": "step.clarity", "dehaze": "step.dehaze",
    "vibrance": "step.vibrance", "white_balance": "step.white_balance",
    "optics": "step.optics", "dodge_burn": "step.dodge_burn",
    "split_tone": "step.split_tone", "skin_smooth": "step.skin_smooth",
}


def step_label(token: str) -> str:
    """Переведённая подпись шага конвейера для текущего языка UI."""
    from upscaler.ui.i18n import tr
    key = STEP_LABELS.get(token, token)
    return tr(key)


def resolve_step_order(order) -> list:
    """Итоговая последовательность шагов для config["order"].

    Известные токены в заданном порядке (без дублей), затем недостающие
    дефолтные, чтобы ни один настроенный шаг не потерялся. Не-список ->
    копия DEFAULT_ORDER.
    """
    default = PipelineExecutor.DEFAULT_ORDER
    if not isinstance(order, list):
        return list(default)
    seq = []
    for token in order:
        if token in default and token not in seq:
            seq.append(token)
    for token in default:
        if token not in seq:
            seq.append(token)
    return seq


class PipelineCancelled(Exception):
    pass


class PipelineExecutor:
    # Individual-operation order tokens between ANALYZE and VALIDATE. The model
    # may reorder these via ``config["order"]``. The adjuster tokens map 1:1 to
    # the override keys the model already uses (auto_tone, saturation, ...), so
    # each method can be both chosen, parameterised and positioned. The generic
    # "adjust" token runs any remaining adjuster (e.g. Refocus) not covered by a
    # specific token, preserving legacy behaviour.
    _ADJUST_TOKEN_PLUGIN = {
        "auto_tone": "Auto Tone", "auto_contrast": "Auto Contrast",
        "auto_color": "Auto Color", "brightness": "Brightness",
        "contrast": "Contrast", "saturation": "Saturation",
        "sharpness": "Sharpness",
        "auto_levels": "Auto Levels",
        "shadows_highlights": "Shadows/Highlights",
        "clarity": "Clarity", "dehaze": "Dehaze",
        "vibrance": "Vibrance", "white_balance": "White Balance",
        "optics": "Optics", "dodge_burn": "Dodge & Burn",
        "split_tone": "Split Toning", "skin_smooth": "Skin Smooth",
    }
    DEFAULT_ORDER = [
        "predownscale",
        "denoise", "deblur", "icedit", "optics",
        "white_balance", "dehaze",
        "auto_color", "auto_levels", "auto_tone", "auto_contrast",
        "brightness", "contrast", "shadows_highlights", "dodge_burn",
        "saturation", "vibrance", "split_tone", "clarity", "sharpness",
        "adjust",
        "upscale", "face_restore", "skin_smooth",
        "refocus", "sharpen", "colorize", "blend",
    ]

    # Progress-bar stage label shown for each step.
    _STEP_STAGE = {
        "predownscale": "ПОДГОТОВКА",
        "denoise": "ПОДГОТОВКА", "deblur": "ПОДГОТОВКА",
        "icedit": "РЕДАКТИРОВАНИЕ",
        "face_restore": "ЛИЦА",
        "adjust": "ПОДГОТОВКА",
        "upscale": "МАСШТАБ", "refocus": "ПОСТОБРАБОТКА",
        "sharpen": "ПОСТОБРАБОТКА", "colorize": "РАСКРАШИВАНИЕ",
        "blend": "СМЕШИВАНИЕ",
        **{tok: "ПОДГОТОВКА" for tok in _ADJUST_TOKEN_PLUGIN},
    }

    _SNAPSHOT_TOKENS = frozenset(
        {"denoise", "deblur", "upscale", "face_restore", "colorize",
         "sharpen"})

    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.analyzer = SourceAnalyzer()
        self.validator = ResultValidator()

    def resolve_order(self, order) -> list:
        """Return the step sequence to run.

        Keeps only known tokens from *order* (in the given sequence, de-duped),
        then appends any default steps not mentioned so every configured step
        still runs. Falls back to the default order when *order* is missing or
        not a list.
        """
        return resolve_step_order(order)

    def execute(
        self,
        image: np.ndarray,
        config: dict,
        meta: dict,
        cancel_event: Optional[threading.Event] = None,
        progress_cb: Optional[Callable[[str, int, str], None]] = None,
        device: str = "cpu",
    ) -> dict:
        from upscaler.engine.denoise_routing import (
            cuda_available, substitute_gpu_denoise,
        )
        config = dict(config)  # don't mutate the caller's dict
        config["denoise"] = substitute_gpu_denoise(
            config.get("denoise", {}),
            cuda_available(device),
            config.get("prefer_gpu_denoise", True),
        )

        def _check_cancel():
            if cancel_event and cancel_event.is_set():
                raise PipelineCancelled("Pipeline cancelled by user")

        def _progress(stage: str, pct: int, msg: str = ""):
            if progress_cb:
                progress_cb(stage, pct, msg)

        # Stage 1: ANALYZE
        _check_cancel()
        _progress("АНАЛИЗ", 0, "Анализ исходного изображения...")
        analysis = self.analyzer.analyze(image, meta)
        _progress("АНАЛИЗ", 100, "Анализ завершён")

        # Mutable image state shared by the ordered step closures below.
        state = {"img": image.copy()}
        blend_cfg = config.get("blend")
        blend_enabled = isinstance(blend_cfg, dict) and \
            bool(blend_cfg.get("enabled"))
        state["snapshots"] = {}
        _SNAPSHOT_LIMIT = 6

        def _maybe_snapshot(token, before):
            """Сохранить снимок после снимкового шага, если картинка менялась."""
            if not blend_enabled or token not in self._SNAPSHOT_TOKENS:
                return
            img = state["img"]
            if len(state["snapshots"]) >= _SNAPSHOT_LIMIT:
                return
            if before is not None and before.shape == img.shape \
                    and np.array_equal(before, img):
                return
            snap = img if img.dtype == np.uint8 else \
                np.clip(img * 255.0, 0, 255).astype(np.uint8)
            state["snapshots"][token] = snap.copy()

        # Блок G: упорядоченный список снимков "вариантов" — по одному на
        # каждый шаг, изменивший картинку. Собирается ВСЕГДА (не зависит от
        # blend_enabled/_SNAPSHOT_TOKENS) — отдельный от blend-снимкового
        # механизма выше путь для UI-показа промежуточных результатов.
        variants = []
        _VARIANTS_LIMIT = 12

        def _maybe_collect_variant(token, before):
            if len(variants) >= _VARIANTS_LIMIT:
                return
            img = state["img"]
            if before is not None and before.shape == img.shape \
                    and np.array_equal(before, img):
                return
            snap = img if img.dtype == np.uint8 else \
                np.clip(img * 255.0, 0, 255).astype(np.uint8)
            variants.append({"label": step_label(token), "image": snap.copy()})

        enhance_only = config.get("enhance_only", False)
        scale = config.get("scale", 2)
        upscale_cfg = config.get("upscale", {})
        upscaler_name = upscale_cfg.get("plugin") if isinstance(upscale_cfg, dict) else None
        post_cfg = config.get("post", {})

        def _run_plugin(plugin_name, img, params):
            plugin_cls = self.registry.get(plugin_name)
            if not plugin_cls:
                return img
            plugin = plugin_cls()
            plugin.initialize(device)
            try:
                return plugin.process(img, params)
            finally:
                plugin.cleanup()

        # ─── Ordered processing steps ───
        def _step_predownscale(stage):
            pcfg = config.get("predownscale")
            if not isinstance(pcfg, dict):
                return
            if not config.get("predownscale_enabled", True):
                return
            try:
                factor = float(pcfg.get("factor", 1.0))
            except (TypeError, ValueError):
                return
            if not 0.0 < factor < 1.0:
                return
            import cv2
            h, w = state["img"].shape[:2]
            nw = max(int(round(w * factor)), 8)
            nh = max(int(round(h * factor)), 8)
            _check_cancel()
            _progress(stage, 10,
                      f"Предуменьшение до {int(round(factor * 100))}%...")
            state["img"] = cv2.resize(state["img"], (nw, nh),
                                      interpolation=cv2.INTER_AREA)
            log.info("Predownscale: %dx%d -> %dx%d (factor %.2f)",
                     w, h, nw, nh, factor)

        def _step_denoise(stage):
            for plugin_name, params in config.get("denoise", {}).items():
                _check_cancel()
                _progress(stage, 30, f"Шумоподавление: {plugin_name}...")
                p = params if isinstance(params, dict) else {"strength": params}
                state["img"] = _run_plugin(plugin_name, state["img"], p)

        def _step_deblur(stage):
            deblur_cfg = config.get("deblur", {})
            if not deblur_cfg or (isinstance(deblur_cfg, dict)
                                  and not deblur_cfg.get("enabled", True)):
                return
            if not self.registry.get("SmartDeblur"):
                return
            _check_cancel()
            _progress(stage, 45, "Деблюр (SmartDeblur)...")
            params = dict(deblur_cfg) if isinstance(deblur_cfg, dict) else {}
            params["_cancel_cb"] = (
                lambda: bool(cancel_event and cancel_event.is_set()))
            params["_progress_cb"] = (
                lambda pct: _progress(stage, 45, f"Деблюр {pct}%..."))
            state["img"] = _run_plugin("SmartDeblur", state["img"], params)

        def _step_icedit(stage):
            icfg = config.get("icedit", {})
            if not isinstance(icfg, dict) or not icfg.get("enabled", True):
                return
            if not (icfg.get("instruction") or "").strip():
                return
            if not self.registry.get("ICEdit"):
                return
            _check_cancel()
            _progress(stage, 50, "Редактирование (ICEdit)...")
            params = dict(icfg)
            params["_cancel_cb"] = (
                lambda: bool(cancel_event and cancel_event.is_set()))
            params["_progress_cb"] = (
                lambda pct: _progress(stage, 50, f"ICEdit {pct}%..."))
            state["img"] = _run_plugin("ICEdit", state["img"], params)

        def _step_face_restore(stage):
            face_cfg = config.get("face", {})
            if not face_cfg or not isinstance(face_cfg, dict) \
                    or not face_cfg.get("enabled", True):
                return
            if not self.registry.get("CodeFormer"):
                return
            _check_cancel()
            _progress(stage, 50, "Восстановление лиц (CodeFormer)...")
            params = dict(face_cfg)
            params["_cancel_cb"] = (
                lambda: bool(cancel_event and cancel_event.is_set()))
            params["_progress_cb"] = (
                lambda pct: _progress(stage, 50, f"Лица {pct}%..."))
            state["img"] = _run_plugin("CodeFormer", state["img"], params)

        def _run_adjuster(plugin_name, params, stage):
            _check_cancel()
            _progress(stage, 60, f"Коррекция: {plugin_name}...")
            p = params if isinstance(params, dict) else {}
            state["img"] = _run_plugin(plugin_name, state["img"], p)

        def _make_adjust_step(plugin_name):
            # One specific adjuster, positioned by its own order token.
            def _step(stage):
                adjust_cfg = config.get("adjust", {})
                if plugin_name in adjust_cfg:
                    _run_adjuster(plugin_name, adjust_cfg[plugin_name], stage)
            return _step

        def _step_adjust_rest(stage):
            # Any adjuster without a dedicated token (e.g. Refocus, custom ones).
            tokenized = set(self._ADJUST_TOKEN_PLUGIN.values())
            for plugin_name, params in config.get("adjust", {}).items():
                if plugin_name not in tokenized:
                    _run_adjuster(plugin_name, params, stage)

        def _step_upscale(stage):
            if not upscaler_name:
                return
            import cv2
            if enhance_only:
                plugin_cls = self.registry.get(upscaler_name)
                if not plugin_cls:
                    return
                supported = sorted(plugin_cls.supported_scales)
                min_scale = supported[0] if supported else 2
                orig_h, orig_w = state["img"].shape[:2]
                _progress(stage, 0,
                          f"Улучшение: {upscaler_name} ({min_scale}x + уменьшение)...")
                state["img"] = _run_plugin(
                    upscaler_name, state["img"], {**upscale_cfg, "scale": min_scale})
                state["img"] = cv2.resize(state["img"], (orig_w, orig_h),
                                          interpolation=cv2.INTER_LANCZOS4)
                _progress(stage, 100, "Улучшение завершено")
                return
            passes = self._plan_passes(upscaler_name, scale)
            for i, (name, pass_scale) in enumerate(passes):
                _check_cancel()
                pct = int((i / len(passes)) * 100)
                _progress(stage, pct,
                          f"Увеличение {pass_scale}x: {name} "
                          f"(проход {i+1}/{len(passes)})...")
                state["img"] = _run_plugin(
                    name, state["img"], {**upscale_cfg, "scale": pass_scale})
                if i < len(passes) - 1:
                    state["img"] = cv2.bilateralFilter(state["img"], 5, 30, 30)
            if passes:
                _progress(stage, 100, "Масштабирование завершено")

        def _step_refocus(stage):
            if not post_cfg.get("refocus") or not self.registry.get("Refocus"):
                return
            _check_cancel()
            _progress(stage, 30, "Рефокусировка...")
            rfp = post_cfg["refocus"] if isinstance(post_cfg["refocus"], dict) \
                else {"strength": 0.5}
            state["img"] = _run_plugin("Refocus", state["img"], rfp)

        def _step_sharpen(stage):
            if not post_cfg.get("sharpen") or not self.registry.get("Sharpness"):
                return
            _check_cancel()
            _progress(stage, 70, "Повышение резкости...")
            amount = post_cfg["sharpen"] \
                if isinstance(post_cfg["sharpen"], (int, float)) else 0.5
            state["img"] = _run_plugin("Sharpness", state["img"], {"amount": amount})

        def _step_colorize(stage):
            for plugin_name, params in config.get("colorize", {}).items():
                _check_cancel()
                _progress(stage, 50, f"Раскрашивание: {plugin_name}...")
                p = params if isinstance(params, dict) else {"strength": 1.0}
                state["img"] = _run_plugin(plugin_name, state["img"], p)

        def _step_blend(stage):
            if not blend_enabled:
                return
            from upscaler.engine import blend_search as bs
            snaps = dict(state["snapshots"])
            cur = state["img"]
            cur_u8 = cur if cur.dtype == np.uint8 else \
                np.clip(cur * 255.0, 0, 255).astype(np.uint8)
            snaps["result"] = cur_u8
            if len(snaps) < 2:
                return
            _check_cancel()
            _progress(stage, 10, "Смешивание вариантов: подбор рецепта...")
            h, w = cur_u8.shape[:2]
            proxies = {}
            import cv2
            for key, img_ in snaps.items():
                p = img_ if img_.shape[:2] == (h, w) else cv2.resize(
                    img_, (w, h), interpolation=cv2.INTER_LANCZOS4)
                proxies[key] = bs.make_proxy(p)
            recipe = bs.greedy_blend_search(proxies)
            if not recipe.get("layers"):
                log.info("Blend: рецепт без слоёв, результат без изменений")
                return
            _progress(stage, 70, "Смешивание вариантов: композиция...")
            out = bs.apply_recipe(recipe, snaps)
            state["img"] = out if cur.dtype == np.uint8 else \
                out.astype(np.float32) / 255.0
            log.info("Blend applied: base=%s layers=%s score=%.4f",
                     recipe["base"], recipe["layers"], recipe["score"])

        steps = {
            "predownscale": _step_predownscale,
            "denoise": _step_denoise, "deblur": _step_deblur,
            "icedit": _step_icedit, "face_restore": _step_face_restore,
            "adjust": _step_adjust_rest, "upscale": _step_upscale,
            "refocus": _step_refocus, "sharpen": _step_sharpen,
            "colorize": _step_colorize, "blend": _step_blend,
        }
        # Per-adjuster steps so the model can position each one individually.
        for _token, _pname in self._ADJUST_TOKEN_PLUGIN.items():
            steps[_token] = _make_adjust_step(_pname)

        # Stages 2-5: run the configured/ordered processing steps.
        sequence = self.resolve_order(config.get("order"))
        log.info("Pipeline step order: %s", " > ".join(sequence))
        for token in sequence:
            _check_cancel()
            step = steps.get(token)
            if step:
                before = state["img"]
                step(self._STEP_STAGE.get(token, "ОБРАБОТКА"))
                _maybe_snapshot(
                    token, before if blend_enabled and
                    token in self._SNAPSHOT_TOKENS else None)
                _maybe_collect_variant(token, before)
        current = state["img"]

        # Stage 6: VALIDATE
        _check_cancel()
        _progress("ПРОВЕРКА", 0, "Проверка результата...")
        metrics = self.validator.validate(current, image)
        _progress("ПРОВЕРКА", 100, "Проверка завершена")

        return {"image": current, "metrics": metrics, "variants": variants}

    def _plan_passes(self, upscaler_name: str, target_scale: int) -> list:
        plugin_cls = self.registry.get(upscaler_name)
        if not plugin_cls:
            return []
        supported = sorted(plugin_cls.supported_scales, reverse=True)
        if not supported:
            return [(upscaler_name, target_scale)]
        passes = []
        remaining = target_scale
        while remaining > 1:
            chosen = None
            for s in supported:
                if remaining >= s:
                    chosen = s
                    break
            if chosen is None:
                chosen = min(supported)
            passes.append((upscaler_name, chosen))
            remaining = remaining // chosen
        return passes
