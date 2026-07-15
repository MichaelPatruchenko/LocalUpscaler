"""Pure helpers for the ICEdit in-context (diptych inpaint) recipe.

No torch/diffusers here — only numpy + cv2 so these are unit-testable without
any model. The plugin composes these with a FLUX-Fill pipeline.
"""
import cv2
import numpy as np

_PROMPT_TEMPLATE = (
    "A diptych with two side-by-side images of the same scene. On the right, "
    "the scene is exactly the same as on the left but {instruction}"
)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize an RGB uint8 image to *width*, keeping aspect; force even height."""
    h, w = image.shape[:2]
    if w == 0:
        return image
    new_h = max(2, int(round(h * width / float(w))))
    new_h += new_h % 2  # even height keeps FLUX VAE happy
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LANCZOS4
    return cv2.resize(image, (width, new_h), interpolation=interp)


def build_diptych(image: np.ndarray) -> tuple:
    """Return (canvas Hx(2W)x3 uint8, mask Hx(2W) uint8).

    Left half is the source image, right half is a copy of it (FLUX-Fill noises
    the masked region anyway); mask is 0 on the left, 255 on the right.
    """
    h, w = image.shape[:2]
    canvas = np.concatenate([image, image], axis=1)
    mask = np.zeros((h, 2 * w), dtype=np.uint8)
    mask[:, w:] = 255
    return canvas, mask


def build_instruction_prompt(instruction: str) -> str:
    return _PROMPT_TEMPLATE.format(instruction=instruction.strip())


def extract_right_half(canvas: np.ndarray) -> np.ndarray:
    """Return the right half (the edited side) of a diptych canvas."""
    w = canvas.shape[1] // 2
    return canvas[:, w:]
