"""YuNet face detection via cv2.FaceDetectorYN (OpenCV >= 4.8).

Returns bounding boxes + 5 landmarks. Any failure (missing model / no YuNet)
yields an empty list so callers degrade gracefully.

Detection runs on a copy normalized to ~DETECT_SIZE on the long side (YuNet is
trained around 320-640px and produces garbage boxes on multi-megapixel
inputs); coordinates are scaled back to the original image. Implausible
detections (bad aspect ratio, inverted/degenerate landmark geometry) are
filtered out.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Long-side target for the detection copy. YuNet's training resolution.
DETECT_SIZE = 640

# Landmark row order produced by YuNet.
_RIGHT_EYE, _LEFT_EYE, _NOSE, _MOUTH_R, _MOUTH_L = range(5)


@dataclass
class Face:
    bbox: tuple  # (x, y, w, h)
    landmarks: np.ndarray  # (5, 2) float32
    score: float


def plausible_face(face: Face) -> bool:
    """Geometric sanity check that a detection actually looks like a face.

    Rejects NaN/inf coordinates, extreme bbox aspect ratios and landmark
    layouts no upright face can produce (eyes below nose/mouth, degenerate or
    out-of-box landmarks). Keeps texture/noise false positives out of the
    restoration path.
    """
    x, y, w, h = face.bbox
    lms = face.landmarks
    if not np.all(np.isfinite(lms)) or not all(
            np.isfinite(v) for v in (x, y, w, h)):
        return False
    if w <= 0 or h <= 0:
        return False
    aspect = w / h
    if not (0.5 <= aspect <= 2.0):
        return False

    eyes_y = lms[[_RIGHT_EYE, _LEFT_EYE], 1]
    lower_y = lms[[_NOSE, _MOUTH_R, _MOUTH_L], 1]
    if eyes_y.max() >= lower_y.min():
        return False

    interocular = float(np.linalg.norm(lms[_RIGHT_EYE] - lms[_LEFT_EYE]))
    if not (0.2 * w <= interocular <= 0.9 * w):
        return False

    # All 5 points must sit inside the bbox expanded by ~30% per side.
    mx, my = 0.3 * w, 0.3 * h
    if (lms[:, 0].min() < x - mx or lms[:, 0].max() > x + w + mx
            or lms[:, 1].min() < y - my or lms[:, 1].max() > y + h + my):
        return False
    return True


def detect_faces(image_rgb: np.ndarray, model_path: str,
                 score_threshold: float = 0.6,
                 detect_size: int = DETECT_SIZE) -> list[Face]:
    """Detect faces. image_rgb is HxWx3 RGB uint8.

    Returns list[Face] in ORIGINAL image coordinates, sorted by descending
    score, implausible detections filtered out.
    """
    try:
        if not hasattr(cv2, "FaceDetectorYN"):
            return []
        h, w = image_rgb.shape[:2]
        scale = min(1.0, detect_size / max(h, w))
        if scale < 1.0:
            det_img = cv2.resize(
                image_rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA)
        else:
            det_img = image_rgb
        dh, dw = det_img.shape[:2]
        bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
        detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (dw, dh),
            score_threshold, 0.3, 50)
        detector.setInputSize((dw, dh))
        _, faces = detector.detect(bgr)
        if faces is None:
            return []
        inv = 1.0 / scale
        out = []
        for row in faces:
            x, y, fw, fh = (float(v) * inv for v in row[0:4])
            lms = row[4:14].reshape(5, 2).astype(np.float32) * np.float32(inv)
            face = Face(bbox=(int(round(x)), int(round(y)),
                              int(round(fw)), int(round(fh))),
                        landmarks=lms, score=float(row[14]))
            if plausible_face(face):
                out.append(face)
            else:
                log.debug("Face detection rejected as implausible: bbox=%s",
                          face.bbox)
        out.sort(key=lambda f: f.score, reverse=True)
        return out
    except Exception as exc:
        log.warning("Face detection failed (%s); returning none", exc)
        return []
