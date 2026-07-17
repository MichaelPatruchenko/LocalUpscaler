"""Comprehensive source image analysis.

Computes statistical, histogram, texture, spatial, color, frequency-domain,
and quality metrics for an input image.
"""

import logging
import math

import cv2
import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class SourceAnalyzer:
    """Analyzes a source image and returns a rich feature dictionary."""

    # Maximum dimension (width or height) for expensive operations.
    _MAX_DIM_EXPENSIVE = 512

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def analyze(self, image: np.ndarray, meta: dict,
                detect_faces: bool = True) -> dict:
        h, w = image.shape[:2]
        is_color = image.ndim == 3 and image.shape[2] >= 3
        is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if is_color else (
            image[:, :, 0] if image.ndim == 3 else image
        )
        gray_f = gray.astype(np.float64)

        # Down-sampled copies for expensive ops
        gray_small = self._downsample(gray, self._MAX_DIM_EXPENSIVE)
        image_small = self._downsample(image, self._MAX_DIM_EXPENSIVE)

        # --- Original metrics (preserved) --------------------------------
        noise_level = self._estimate_noise(gray_f)
        blur_score = self._estimate_blur(gray_f)
        brightness = self._estimate_brightness(gray)
        contrast = self._estimate_contrast(gray)
        detail_level = self._estimate_detail(gray_f)
        color_cast = self._detect_color_cast(image) if is_color else (0.0, 0.0, 0.0)

        result: dict = {
            # Legacy keys
            "noise_level": noise_level,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "detail_level": detail_level,
            "color_cast": color_cast,
            "color_space": "sRGB",
            "bit_depth": meta.get("bit_depth", 8),
            "resolution": (w, h),
            "megapixels": (w * h) / 1e6,
            "has_icc": meta.get("icc_profile") is not None,
            "is_grayscale": is_grayscale,
            "dynamic_range": float(image.max()) - float(image.min()),
            "recommended_denoise_strength": min(noise_level / 50.0, 1.0),
            **(self._detect_faces_metric(image) if detect_faces
               else {"has_faces": False, "face_count": 0}),
        }

        # --- New metric groups -------------------------------------------
        result.update(self._compute_statistical(gray, image if is_color else None))
        result.update(self._compute_histogram_features(gray))
        result.update(self._compute_glcm_texture(gray_small))
        result.update(self._compute_spatial(gray_small))
        result.update(
            self._compute_color_metrics(image_small) if is_color else {}
        )
        result.update(self._compute_frequency(gray_small))
        result.update(self._compute_wavelet_quality(gray_small))
        result.update(self._compute_photo_flaws(image_small, gray))

        # Blur assessment for SmartDeblur triggering / parameter selection.
        try:
            from upscaler.engine.blur_estimator import BlurEstimator
            result["blur_assessment"] = BlurEstimator().assess(gray_small)
        except Exception:
            logger.debug("blur assessment failed", exc_info=True)

        # Эффективное разрешение (для авто-предуменьшения). Считается по
        # ПОЛНОразмерному gray: ресемплинг уничтожил бы сигнатуру апсемпла.
        try:
            from upscaler.engine.effective_resolution import (
                effective_downscale_factor,
            )
            result["effective_downscale_factor"] = (
                effective_downscale_factor(gray))
        except Exception:
            logger.debug("effective resolution estimation failed",
                         exc_info=True)
            result["effective_downscale_factor"] = 1.0

        return result

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _downsample(img: np.ndarray, max_dim: int) -> np.ndarray:
        """Return a copy resized so the largest side <= *max_dim*."""
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _detect_faces_metric(self, image: np.ndarray) -> dict:
        """Lightweight face presence/count for auto-config (YuNet, guarded)."""
        try:
            if image.ndim != 3 or image.shape[2] < 3:
                return {"has_faces": False, "face_count": 0}
            from upscaler.plugins.face import facedet
            from upscaler.models.manager import ModelManager
            from upscaler.config import MODELS_DIR

            rgb = image[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

            mm = ModelManager(MODELS_DIR)
            path = mm.get_model_path("YuNet")
            if not path.exists():
                try:
                    mm.download("YuNet")
                except Exception:
                    return {"has_faces": False, "face_count": 0}
            # facedet владеет нормализацией разрешения и возвращает лица в
            # координатах оригинала — тот же путь, что у face_restore, поэтому
            # has_faces/face_count совпадают с тем, что увидит восстановление.
            faces = facedet.detect_faces(rgb, str(path))
            result = {"has_faces": bool(faces), "face_count": len(faces)}
            if faces:
                result.update(self._face_quality_metrics(rgb, faces, 1.0))
            return result
        except Exception:
            return {"has_faces": False, "face_count": 0}

    @staticmethod
    def _face_quality_metrics(rgb: np.ndarray, faces: list,
                              inv_scale: float) -> dict:
        """Резкость/шум/размер лицевых кропов для авто-подбора CodeFormer.

        rgb — изображение (uint8), faces — list[Face] в его координатах,
        inv_scale — множитель пересчёта размеров в пиксели оригинала (1.0,
        когда rgb уже полноразмерный). Пустой список лиц -> {}.
        """
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)
        sharpness_vals, noise_vals, min_sides = [], [], []
        for face in faces:
            x, y, fw, fh = face.bbox
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = gray[y0:y1, x0:x1].astype(np.float64)
            lap = cv2.Laplacian(crop, cv2.CV_64F)
            sharpness_vals.append(float(lap.var()))
            noise_vals.append(
                float(np.sqrt(np.pi / 2) * np.mean(np.abs(lap)) / 6.0))
            min_sides.append(min(x1 - x0, y1 - y0))
        if not sharpness_vals:
            return {}
        # Дисперсия Лапласиана ~300+ на резком лице; нормируем с капом 1.0.
        sharpness = min(float(np.mean(sharpness_vals)) / 300.0, 1.0)
        return {
            "face_sharpness": round(sharpness, 3),
            "face_noise": round(float(np.mean(noise_vals)), 2),
            "face_min_px": int(round(min(min_sides) * inv_scale)),
        }

    # ------------------------------------------------------------------ #
    #  Original estimators (kept as-is)                                    #
    # ------------------------------------------------------------------ #

    def _estimate_noise(self, gray_f: np.ndarray) -> float:
        """Laplacian-based noise estimation."""
        laplacian = cv2.Laplacian(gray_f, cv2.CV_64F)
        sigma = np.sqrt(np.pi / 2) * np.mean(np.abs(laplacian)) * (1.0 / 6.0)
        return float(sigma)

    def _estimate_blur(self, gray_f: np.ndarray) -> float:
        """Laplacian variance — lower = more blur. Range ~0-2000+."""
        laplacian = cv2.Laplacian(gray_f, cv2.CV_64F)
        return float(laplacian.var())

    def _estimate_brightness(self, gray: np.ndarray) -> float:
        """Average brightness 0-255."""
        return float(gray.mean())

    def _estimate_contrast(self, gray: np.ndarray) -> float:
        """Standard deviation of pixel values. Low = flat, high = contrasty."""
        return float(gray.std())

    def _estimate_detail(self, gray_f: np.ndarray) -> float:
        """High-frequency energy — edges and texture amount."""
        sobelx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return float(magnitude.mean())

    def _detect_color_cast(self, image: np.ndarray) -> tuple:
        """Deviation of each channel mean from neutral gray. (R, G, B) offsets."""
        means = image.mean(axis=(0, 1)).astype(float)
        gray_mean = means.mean()
        return tuple(float(m - gray_mean) for m in means[:3])

    # ------------------------------------------------------------------ #
    #  1. Statistical metrics                                              #
    # ------------------------------------------------------------------ #

    def _compute_statistical(self, gray: np.ndarray, color: np.ndarray | None) -> dict:
        """Per-channel and grayscale basic statistics."""
        result: dict = {}

        def _stats_for(arr: np.ndarray, prefix: str) -> None:
            flat = arr.ravel().astype(np.float64)
            result[f"{prefix}_mean"] = float(np.mean(flat))
            result[f"{prefix}_variance"] = float(np.var(flat))
            result[f"{prefix}_std"] = float(np.std(flat))
            result[f"{prefix}_skewness"] = float(sp_stats.skew(flat))
            result[f"{prefix}_kurtosis"] = float(sp_stats.kurtosis(flat))
            result[f"{prefix}_median"] = float(np.median(flat))
            result[f"{prefix}_min"] = float(np.min(flat))
            result[f"{prefix}_max"] = float(np.max(flat))
            # Entropy from normalised histogram
            hist, _ = np.histogram(flat, bins=256, range=(0, 255))
            hist_norm = hist / hist.sum()
            hist_norm = hist_norm[hist_norm > 0]
            result[f"{prefix}_entropy"] = float(sp_stats.entropy(hist_norm, base=2))

        _stats_for(gray, "gray")

        if color is not None:
            channel_names = ("r", "g", "b")
            for i, name in enumerate(channel_names):
                _stats_for(color[:, :, i], name)

        return result

    # ------------------------------------------------------------------ #
    #  2. Histogram features                                               #
    # ------------------------------------------------------------------ #

    def _compute_histogram_features(self, gray: np.ndarray) -> dict:
        """Histogram shape moments (mean, variance, skewness, kurtosis)."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
        hist_norm = hist / hist.sum()
        bins = np.arange(256, dtype=np.float64)

        h_mean = float(np.sum(bins * hist_norm))
        h_var = float(np.sum(((bins - h_mean) ** 2) * hist_norm))
        h_std = math.sqrt(h_var) if h_var > 0 else 0.0

        if h_std > 0:
            h_skew = float(np.sum(((bins - h_mean) / h_std) ** 3 * hist_norm))
            h_kurt = float(np.sum(((bins - h_mean) / h_std) ** 4 * hist_norm) - 3.0)
        else:
            h_skew = 0.0
            h_kurt = 0.0

        return {
            "hist_mean": h_mean,
            "hist_variance": h_var,
            "hist_skewness": h_skew,
            "hist_kurtosis": h_kurt,
        }

    # ------------------------------------------------------------------ #
    #  3. GLCM texture features                                            #
    # ------------------------------------------------------------------ #

    def _compute_glcm_texture(self, gray_small: np.ndarray) -> dict:
        """GLCM-based texture descriptors (skimage)."""
        defaults = {
            "glcm_contrast": 0.0,
            "glcm_energy": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_correlation": 0.0,
        }
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            logger.debug("skimage not available — skipping GLCM features")
            return defaults

        try:
            # Quantise to 64 levels for speed
            levels = 64
            img_q = (gray_small.astype(np.float64) / 255.0 * (levels - 1)).astype(np.uint8)
            glcm = graycomatrix(
                img_q,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=levels,
                symmetric=True,
                normed=True,
            )
            return {
                "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
                "glcm_energy": float(graycoprops(glcm, "energy").mean()),
                "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
                "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
            }
        except Exception as exc:
            logger.debug("GLCM computation failed: %s", exc)
            return defaults

    # ------------------------------------------------------------------ #
    #  4. Spatial features                                                 #
    # ------------------------------------------------------------------ #

    def _compute_spatial(self, gray_small: np.ndarray) -> dict:
        """Edge density, corner density, fractal dimension."""
        total_pixels = gray_small.shape[0] * gray_small.shape[1]

        # Edge density (Canny)
        edges = cv2.Canny(gray_small, 100, 200)
        edge_density = float(np.count_nonzero(edges) / total_pixels)

        # Corner density (Harris)
        gray_f32 = np.float32(gray_small)
        harris = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
        corner_count = int(np.count_nonzero(harris > 0.01 * harris.max()))
        corner_density = float(corner_count / total_pixels)

        # Fractal dimension via box-counting on the edge map
        fractal_dim = self._box_counting_dimension(edges)

        return {
            "edge_density": edge_density,
            "corner_density": corner_density,
            "fractal_dimension": fractal_dim,
        }

    @staticmethod
    def _box_counting_dimension(binary: np.ndarray) -> float:
        """Estimate fractal dimension of a binary image via box-counting."""
        # Ensure binary
        Z = (binary > 0)
        if not Z.any():
            return 0.0

        # Pad to next power of 2
        h, w = Z.shape
        dim = max(h, w)
        p = int(2 ** np.ceil(np.log2(dim)))
        padded = np.zeros((p, p), dtype=bool)
        padded[:h, :w] = Z

        sizes = []
        counts = []
        size = p
        while size >= 2:
            # Reshape into boxes of (size x size) and check occupancy
            n_boxes_side = p // size
            reshaped = padded[:n_boxes_side * size, :n_boxes_side * size]
            reshaped = reshaped.reshape(n_boxes_side, size, n_boxes_side, size)
            box_has_pixel = reshaped.any(axis=(1, 3))
            count = int(box_has_pixel.sum())
            if count > 0:
                sizes.append(size)
                counts.append(count)
            size //= 2

        if len(sizes) < 2:
            return 0.0

        log_sizes = np.log(np.array(sizes, dtype=np.float64))
        log_counts = np.log(np.array(counts, dtype=np.float64))
        # Linear regression: log(count) = -D * log(size) + c
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(-coeffs[0])

    # ------------------------------------------------------------------ #
    #  5. Color metrics (3-channel only)                                   #
    # ------------------------------------------------------------------ #

    def _compute_color_metrics(self, image_small: np.ndarray) -> dict:
        """Color-specific features. Only call for 3-channel images."""
        if image_small.ndim != 3 or image_small.shape[2] < 3:
            return {}

        rgb = image_small[:, :, :3].astype(np.float64)
        r_mean = rgb[:, :, 0].mean()
        g_mean = rgb[:, :, 1].mean()
        b_mean = rgb[:, :, 2].mean()
        overall_mean = rgb.mean()

        # Average color
        avg_color = (float(r_mean), float(g_mean), float(b_mean))

        # Channel balance
        if overall_mean > 0:
            channel_balance = (
                float(r_mean / overall_mean),
                float(g_mean / overall_mean),
                float(b_mean / overall_mean),
            )
        else:
            channel_balance = (0.0, 0.0, 0.0)

        # Color temperature proxy (R/B ratio)
        color_temperature = float(r_mean / b_mean) if b_mean > 0 else 0.0

        # HSV saturation stats
        hsv = cv2.cvtColor(image_small[:, :, :3], cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float64)
        sat_mean = float(sat.mean())
        sat_std = float(sat.std())

        # Hue stats (circular mean on non-zero saturation pixels)
        hue = hsv[:, :, 0].astype(np.float64)  # 0-179 in OpenCV
        hue_mean = float(hue.mean())

        # Color harmony: entropy of hue histogram (higher = more varied)
        hue_hist, _ = np.histogram(hue.ravel(), bins=180, range=(0, 180))
        hue_hist_norm = hue_hist.astype(np.float64)
        hue_sum = hue_hist_norm.sum()
        if hue_sum > 0:
            hue_hist_norm = hue_hist_norm / hue_sum
            hue_hist_nonzero = hue_hist_norm[hue_hist_norm > 0]
            color_harmony = float(sp_stats.entropy(hue_hist_nonzero, base=2))
        else:
            color_harmony = 0.0

        # Dominant colors via k-means (k=3)
        dominant_colors = self._dominant_colors(image_small[:, :, :3], k=3)

        return {
            "average_color": avg_color,
            "channel_balance": channel_balance,
            "color_temperature": color_temperature,
            "saturation_mean": sat_mean,
            "saturation_std": sat_std,
            "hue_mean": hue_mean,
            "color_harmony": color_harmony,
            "dominant_colors": dominant_colors,
        }

    @staticmethod
    def _dominant_colors(image_rgb: np.ndarray, k: int = 3) -> list[tuple[float, float, float]]:
        """K-means dominant colour extraction. Returns list of (R, G, B) tuples."""
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            # Sort by cluster population (most dominant first)
            _, counts = np.unique(labels, return_counts=True)
            order = np.argsort(-counts)
            centers = centers[order]
            return [tuple(float(c) for c in center) for center in centers]
        except Exception as exc:
            logger.debug("K-means failed: %s", exc)
            return [(0.0, 0.0, 0.0)] * k

    # ------------------------------------------------------------------ #
    #  6. Frequency domain features                                        #
    # ------------------------------------------------------------------ #

    def _compute_frequency(self, gray_small: np.ndarray) -> dict:
        """FFT-based low / mid / high frequency energy ratios."""
        f = np.fft.fft2(gray_small.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift) ** 2  # power spectrum

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = math.sqrt(cy ** 2 + cx ** 2)

        # Build radial distance map
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        # Three equal bands
        r1 = max_radius / 3.0
        r2 = 2.0 * max_radius / 3.0

        total_energy = magnitude.sum()
        if total_energy == 0:
            return {
                "fft_low_energy_ratio": 0.0,
                "fft_mid_energy_ratio": 0.0,
                "fft_high_energy_ratio": 0.0,
            }

        low = float(magnitude[r <= r1].sum() / total_energy)
        mid = float(magnitude[(r > r1) & (r <= r2)].sum() / total_energy)
        high = float(magnitude[r > r2].sum() / total_energy)

        return {
            "fft_low_energy_ratio": low,
            "fft_mid_energy_ratio": mid,
            "fft_high_energy_ratio": high,
        }

    def _compute_photo_flaws(self, image_small: np.ndarray,
                             gray: np.ndarray) -> dict:
        """Метрики дефектов для авто-правил: дымка, клиппинг, виньетка."""
        g = gray.astype(np.float64)
        total = g.size
        shadow_clip = float((g <= 5).sum() / total)
        highlight_clip = float((g >= 250).sum() / total)
        shadow_mass = float((g < 60).sum() / total)

        # Дымка: средний dark channel (min по каналам + эрозия) на копии.
        if image_small.ndim == 3 and image_small.shape[2] >= 3:
            rgb = image_small[:, :, :3].astype(np.float64) / 255.0
        else:
            rgb = np.stack([image_small.astype(np.float64) / 255.0] * 3,
                           axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(rgb.min(axis=2), kernel)
        haze_level = float(np.clip(dark.mean(), 0.0, 1.0))

        # Виньетка: перепад средней яркости центр -> углы.
        h, w = g.shape
        ch, cw = max(int(h * 0.3), 1), max(int(w * 0.3), 1)
        cy0, cx0 = (h - ch) // 2, (w - cw) // 2
        center = g[cy0:cy0 + ch, cx0:cx0 + cw].mean()
        kh, kw = max(int(h * 0.15), 1), max(int(w * 0.15), 1)
        corners = np.mean([g[:kh, :kw].mean(), g[:kh, -kw:].mean(),
                           g[-kh:, :kw].mean(), g[-kh:, -kw:].mean()])
        if center > 1.0:
            vignette = float(np.clip(1.0 - corners / center, 0.0, 1.0))
        else:
            vignette = 0.0

        return {
            "haze_level": round(haze_level, 3),
            "shadow_clip": round(shadow_clip, 4),
            "highlight_clip": round(highlight_clip, 4),
            "shadow_mass": round(shadow_mass, 4),
            "vignette_strength": round(vignette, 3),
        }

    # ------------------------------------------------------------------ #
    #  7. Wavelet quality                                                  #
    # ------------------------------------------------------------------ #

    def _compute_wavelet_quality(self, gray_small: np.ndarray) -> dict:
        """Haar wavelet detail subband energy (proxy for sharpness/quality)."""
        img = gray_small.astype(np.float64)
        # Ensure even dimensions
        h, w = img.shape
        h, w = h - h % 2, w - w % 2
        img = img[:h, :w]

        # One-level Haar DWT implemented manually (no pywt dependency)
        # Row-wise transform
        row_low = (img[:, 0::2] + img[:, 1::2]) / 2.0
        row_high = (img[:, 0::2] - img[:, 1::2]) / 2.0

        # Column-wise on row_low -> LL, LH
        ll = (row_low[0::2, :] + row_low[1::2, :]) / 2.0
        lh = (row_low[0::2, :] - row_low[1::2, :]) / 2.0

        # Column-wise on row_high -> HL, HH
        hl = (row_high[0::2, :] + row_high[1::2, :]) / 2.0
        hh = (row_high[0::2, :] - row_high[1::2, :]) / 2.0

        detail_energy = float(np.sum(lh ** 2) + np.sum(hl ** 2) + np.sum(hh ** 2))
        total_energy = float(np.sum(ll ** 2)) + detail_energy

        detail_ratio = detail_energy / total_energy if total_energy > 0 else 0.0

        return {
            "wavelet_detail_energy": detail_energy,
            "wavelet_detail_ratio": detail_ratio,
        }
