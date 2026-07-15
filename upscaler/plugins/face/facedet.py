"""YuNet face detection via cv2.FaceDetectorYN (OpenCV >= 4.8).

Returns bounding boxes + 5 landmarks. Any failure (missing model / no YuNet)
yields an empty list so callers degrade gracefully.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Face:
    bbox: tuple  # (x, y, w, h)
    landmarks: np.ndarray  # (5, 2) float32
    score: float


def detect_faces(image_rgb: np.ndarray, model_path: str,
                 score_threshold: float = 0.6) -> list[Face]:
    """Detect faces. image_rgb is HxWx3 RGB uint8. Returns list[Face]."""
    try:
        if not hasattr(cv2, "FaceDetectorYN"):
            return []
        h, w = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (w, h),
            score_threshold, 0.3, 5000)
        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)
        if faces is None:
            return []
        out = []
        for row in faces:
            x, y, fw, fh = row[0:4]
            lms = row[4:14].reshape(5, 2).astype(np.float32)
            out.append(Face(bbox=(int(x), int(y), int(fw), int(fh)),
                            landmarks=lms, score=float(row[14])))
        return out
    except Exception as exc:
        log.warning("Face detection failed (%s); returning none", exc)
        return []
