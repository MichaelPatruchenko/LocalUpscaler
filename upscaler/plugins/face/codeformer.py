"""CodeFormer face restoration via ONNX, with YuNet detection + alignment."""
import logging

import cv2
import numpy as np

from upscaler.plugins.base import BasePlugin
from upscaler.plugins.face import facedet
from upscaler.plugins.face.align import align_face, paste_back
from upscaler.plugins.face.pipeline_loader import load_session
from upscaler.models.manager import ModelManager
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)

_CODEFORMER_MODEL = "CodeFormer-ONNX"
_YUNET_MODEL = "YuNet"


class CodeFormerPlugin(BasePlugin):
    name = "CodeFormer"
    category = "face"
    supported_scales = []
    gpu_memory_mb = 1500
    supports_video = False
    params_schema = {
        "fidelity": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.7, "ui": "slider"},
        "upscale_background": {"type": "boolean", "default": False},
        "min_face_px": {"type": "integer", "minimum": 16, "maximum": 256,
                        "default": 32, "ui": "slider"},
        "regions": {"type": "array", "default": []},
    }

    def __init__(self):
        self.device = "cpu"
        self._session = None
        self._session_path = None
        self.model_manager = ModelManager(MODELS_DIR)

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def _get_session(self):
        if self._session is not None:
            return self._session
        path = self.model_manager.get_model_path(_CODEFORMER_MODEL)
        if not path.exists():
            self.model_manager.download(_CODEFORMER_MODEL)
        self._session = load_session(str(path), self.device)
        self._session_path = str(path)
        return self._session

    def _yunet_path(self) -> str:
        path = self.model_manager.get_model_path(_YUNET_MODEL)
        if not path.exists():
            self.model_manager.download(_YUNET_MODEL)
        return str(path)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        if params.get("enabled") is False:
            return image

        is_uint8 = image.dtype == np.uint8
        alpha = image[:, :, 3:4] if (image.ndim == 3 and image.shape[2] == 4) else None
        rgb = image[:, :, :3]
        rgb = rgb if is_uint8 else np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        regions = params.get("regions") or []
        if regions:
            # Ручные зоны заменяют автодетекцию (спек, этап 3).
            faces = self._faces_from_regions(rgb, regions)
        else:
            try:
                faces = facedet.detect_faces(rgb, self._yunet_path())
            except Exception as exc:
                log.warning("Face detect failed (%s); skipping", exc)
                return image
        if not faces:
            return image

        try:
            session = self._get_session()
        except Exception as exc:
            log.warning("CodeFormer unavailable (%s); skipping face restore", exc)
            return image

        fidelity = float(params.get("fidelity", 0.7))
        min_px = int(params.get("min_face_px", 32))
        cancel_cb = params.get("_cancel_cb")

        out = rgb.copy()
        for face in faces:
            if cancel_cb is not None and cancel_cb():
                break
            _, _, fw, fh = face.bbox
            # Ручные зоны — явное намерение пользователя: не фильтруем по
            # min_face_px (он для отсева мелких АВТО-детекций).
            if not regions and min(fw, fh) < min_px:
                continue
            crop, affine_m = align_face(out, face.landmarks)
            try:
                restored = self._infer(session, crop, fidelity)
            except Exception as exc:
                log.warning("CodeFormer inference failed (%s); skip face", exc)
                continue
            out = paste_back(out, restored, affine_m)

        if is_uint8:
            if alpha is not None:
                return np.concatenate([out, alpha], axis=2)
            return out
        out_f = out.astype(np.float32) / 255.0
        if alpha is not None:
            alpha_f = alpha.astype(np.float32) / 255.0
            return np.concatenate([out_f, alpha_f], axis=2)
        return out_f

    def _faces_from_regions(self, rgb: np.ndarray, regions: list) -> list:
        """Face-объекты из ручных зон: YuNet внутри зоны либо синтетические
        лендмарки. bbox — НЕрасширенная зона в пикселях."""
        from upscaler.plugins.face.regions import (
            denormalize_rect, expand_rect, synthetic_landmarks,
        )
        img_h, img_w = rgb.shape[:2]
        out = []
        for rect in regions:
            angle = float(rect[4]) if len(rect) > 4 else 0.0
            try:
                x, y, w, h = denormalize_rect(rect, img_w, img_h)
            except Exception:
                continue
            if w < 8 or h < 8:
                continue
            ex, ey, ew, eh = expand_rect(x, y, w, h, img_w, img_h)
            lms = None
            try:
                crop = rgb[ey:ey + eh, ex:ex + ew]
                found = facedet.detect_faces(crop, self._yunet_path())
                if found:
                    best = max(found, key=lambda f: f.score)
                    lms = best.landmarks + np.array([ex, ey], np.float32)
            except Exception as exc:
                log.debug("Region detect failed (%s); synthetic landmarks", exc)
            if lms is None:
                lms = synthetic_landmarks(x, y, w, h, angle)
            out.append(facedet.Face(bbox=(x, y, w, h), landmarks=lms,
                                    score=1.0))
        return out

    def _infer(self, session, crop_rgb: np.ndarray, fidelity: float) -> np.ndarray:
        """Run CodeFormer on a 512x512 RGB crop. Returns a 512x512 RGB uint8."""
        x = crop_rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5                       # -> [-1, 1]
        x = np.transpose(x, (2, 0, 1))[None]      # NCHW
        inputs = session.get_inputs()
        feeds = {inputs[0].name: x.astype(np.float32)}
        # CodeFormer ONNX exports that expose a fidelity weight take a 2nd input.
        if len(inputs) > 1:
            want = getattr(inputs[1], "type", "") or ""
            dt = np.float64 if "double" in want else np.float32
            feeds[inputs[1].name] = np.array([fidelity], dtype=dt)
        result = session.run(None, feeds)[0]
        y = result[0]                             # CHW, [-1, 1]
        y = np.transpose(y, (1, 2, 0))
        y = np.clip((y * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        if y.shape[:2] != (512, 512):
            y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_LINEAR)
        return y

    def cleanup(self) -> None:
        self._session = None
        self._session_path = None
