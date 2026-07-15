"""5-point face alignment to the FFHQ-512 template and soft paste-back.

Pure numpy/cv2 — no model weights — so it is unit-testable without downloads.
"""
import cv2
import numpy as np

# Canonical 5-point template (right eye, left eye, nose, right mouth, left
# mouth) for a 512x512 aligned face, as used by FFHQ/CodeFormer/facexlib.
FFHQ_512_TEMPLATE = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)

_FACE_SIZE = 512


def align_face(image_rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp the face onto a 512x512 canvas. Returns (crop, affine_M 2x3)."""
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    affine_m, _ = cv2.estimateAffinePartial2D(
        src, FFHQ_512_TEMPLATE, method=cv2.LMEDS)
    if affine_m is None:
        # Degenerate landmarks: fall back to identity-ish centered crop.
        affine_m = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)
    crop = cv2.warpAffine(image_rgb, affine_m, (_FACE_SIZE, _FACE_SIZE),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return crop, affine_m


def paste_back(image_rgb: np.ndarray, restored_512: np.ndarray,
               affine_m: np.ndarray, feather: int = 9) -> np.ndarray:
    """Warp the restored 512 face back and alpha-blend it with a soft mask."""
    h, w = image_rgb.shape[:2]
    inv_m = cv2.invertAffineTransform(affine_m)

    warped = cv2.warpAffine(restored_512, inv_m, (w, h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask = np.full((_FACE_SIZE, _FACE_SIZE), 255, dtype=np.uint8)
    mask = cv2.warpAffine(mask, inv_m, (w, h), flags=cv2.INTER_LINEAR)
    # Shrink + feather the seam so the paste is invisible.
    k = max(3, int(feather) | 1)
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]

    base = image_rgb.astype(np.float32)
    blended = warped.astype(np.float32) * alpha + base * (1.0 - alpha)
    return blended.astype(image_rgb.dtype)
