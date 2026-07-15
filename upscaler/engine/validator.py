"""Result validation: quality metrics and artifact detection."""
import cv2
import numpy as np

from upscaler.utils.metrics import compute_brisque, compute_niqe, histogram_similarity, detect_artifacts


class ResultValidator:
    def validate(self, result: np.ndarray, original: np.ndarray) -> dict:
        if result.shape != original.shape:
            original_resized = cv2.resize(original, (result.shape[1], result.shape[0]))
        else:
            original_resized = original

        result_8bit = self._to_uint8(result)
        original_8bit = self._to_uint8(original_resized)

        return {
            "brisque": compute_brisque(result_8bit),
            "niqe": compute_niqe(result_8bit),
            "histogram_similarity": histogram_similarity(result_8bit, original_8bit),
            "artifacts": detect_artifacts(result_8bit),
        }

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        if img.dtype in (np.float32, np.float64):
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.dtype == np.uint16:
            return (img / 257).astype(np.uint8)
        return img.astype(np.uint8)
