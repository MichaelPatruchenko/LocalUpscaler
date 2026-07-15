"""ICEdit plugin: instruction-based editing via quantized FLUX-Fill + LoRA."""
import logging

import cv2
import numpy as np

from upscaler.plugins.base import BasePlugin
from upscaler.plugins.icedit.incontext import (
    resize_to_width, build_diptych, build_instruction_prompt, extract_right_half,
)
from upscaler.plugins.icedit.pipeline_loader import load_flux_fill
from upscaler.models.manager import ModelManager
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)

_WORK_WIDTH = 512  # ICEdit reference width per diptych side


class ICEditPlugin(BasePlugin):
    name = "ICEdit"
    category = "icedit"
    supported_scales = []
    gpu_memory_mb = 8000
    supports_video = False
    params_schema = {
        "instruction": {"type": "string", "default": ""},
        "variant": {
            "type": "string", "ui": "combo", "default": "normal",
            "options": ["normal", "moe"],
            "labels": {"normal": "Normal LoRA (рекомендуется)",
                       "moe": "MoE-LoRA (экспериментально)"},
        },
        "steps": {"type": "integer", "minimum": 8, "maximum": 50, "default": 28},
        "guidance": {"type": "number", "minimum": 1.0, "maximum": 60.0,
                     "default": 50.0},
        "seed": {"type": "integer", "minimum": -1, "maximum": 2**31 - 1,
                 "default": -1},
        "quant": {
            "type": "string", "ui": "combo", "default": "q4",
            "options": ["q4", "q5", "nf4"],
        },
        "offload": {
            "type": "string", "ui": "combo", "default": "model",
            "options": ["model", "none", "sequential"],
        },
    }

    def __init__(self):
        self.device = "cpu"
        self._pipe = None
        self._pipe_key = None
        self.model_manager = ModelManager(MODELS_DIR)

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def _get_pipe(self, variant, quant, offload):
        key = (variant, quant, offload)
        if self._pipe is not None and self._pipe_key == key:
            return self._pipe
        self._pipe = load_flux_fill(variant, quant, offload, self.device,
                                    self.model_manager)
        self._pipe_key = key
        return self._pipe

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        instruction = (params.get("instruction") or "").strip()
        if not instruction:
            return image

        try:
            pipe = self._get_pipe(params.get("variant", "normal"),
                                  params.get("quant", "q4"),
                                  params.get("offload", "model"))
        except Exception as exc:
            log.error("ICEdit недоступен (%s); шаг пропущен — правка НЕ "
                      "применена", exc)
            return image

        is_uint8 = image.dtype == np.uint8
        rgb = image[:, :, :3]
        rgb = rgb if is_uint8 else np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        orig_h, orig_w = rgb.shape[:2]

        small = resize_to_width(rgb, _WORK_WIDTH)
        canvas, mask = build_diptych(small)
        prompt = build_instruction_prompt(instruction)

        try:
            from PIL import Image
            generator = self._make_generator(params.get("seed", -1))
            callback = self._make_callback(params)
            result = pipe(
                prompt=prompt,
                image=Image.fromarray(canvas),
                mask_image=Image.fromarray(mask),
                height=canvas.shape[0], width=canvas.shape[1],
                num_inference_steps=int(params.get("steps", 28)),
                guidance_scale=float(params.get("guidance", 50.0)),
                generator=generator,
                callback_on_step_end=callback,
            )
            edited_canvas = np.asarray(result.images[0])
        except Exception as exc:
            log.warning("ICEdit inference failed (%s); skipping edit", exc)
            return image

        edited_small = extract_right_half(edited_canvas)
        edited = cv2.resize(edited_small, (orig_w, orig_h),
                            interpolation=cv2.INTER_LANCZOS4)
        if is_uint8:
            return edited.astype(np.uint8)
        return (edited.astype(np.float32) / 255.0)

    def _make_generator(self, seed):
        try:
            import torch
            if seed is None or int(seed) < 0:
                return None
            return torch.Generator(device="cpu").manual_seed(int(seed))
        except Exception:
            return None

    def _make_callback(self, params):
        cancel_cb = params.get("_cancel_cb")
        progress_cb = params.get("_progress_cb")
        total = int(params.get("steps", 28))

        def _cb(pipe, step, timestep, kwargs):
            if cancel_cb is not None and cancel_cb():
                pipe._interrupt = True
            if progress_cb is not None:
                progress_cb(int(100 * (step + 1) / max(total, 1)))
            return kwargs

        return _cb

    def cleanup(self) -> None:
        self._pipe = None
        self._pipe_key = None
        try:
            self.model_manager.unload_current()
        except Exception:
            pass
