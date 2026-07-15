"""Blind blur parameter estimation for SmartDeblur auto mode.

Estimates blur type (motion/focus/gaussian), radius (or half motion length),
motion angle, and a regularization 'smooth' value from a grayscale image.
"""
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class BlurEstimator:
    """Estimates deconvolution parameters directly from a blurred image."""

    _MAX_DIM = 512

    # A disk (out-of-focus) PSF has spectral nulls, so its radial power
    # spectrum dips and rebounds; a gaussian falls off monotonically. The
    # rebound depth is measured in nats (log-power); 0.5 was chosen
    # empirically to separate disk from gaussian on synthetic kernels.
    _DISK_DIP_THRESHOLD = 0.5

    # Sharpness ("kernel metric" analog) and need-detection thresholds.
    _SHARPNESS_CUTOFF_FRAC = 0.25   # radial cutoff (fraction of rmax) = "high freq"
    _SHARPNESS_FULL_SCALE = 0.30    # high-freq energy fraction mapped to sharpness 1.0
    _SHARPNESS_NEEDS_DEBLUR = 0.5   # below this (and a kernel detected) -> deblur
    _KERNEL_MIN_RADIUS = 1.5        # plausible blur kernel must be at least this big
    _NOISE_HIGH = 12.0              # noise sigma above which Tikhonov is preferred
    _STRONG_BLUR_RADIUS = 8.0       # radius above which Total Variation is preferred

    def estimate(self, gray: np.ndarray) -> dict:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        gray_u8 = self._to_uint8(gray)
        g = self._downsample(gray_u8).astype(np.float64)

        smooth = self._estimate_smooth(g)
        anisotropy, angle = self._spectral_anisotropy(g)
        motion_len = self._motion_length_cepstrum(g, angle)

        if anisotropy > 1.8 and motion_len >= 3:
            return {
                "blur_type": "motion",
                "radius": round(motion_len / 2.0, 2),
                "angle": round(angle, 1),
                "smooth": smooth,
            }

        blur_type, radius = self._classify_defocus(g)
        log.debug("BlurEstimator: type=%s radius=%.2f angle=%.1f smooth=%.1f "
                  "(anisotropy=%.2f motion_len=%.1f)", blur_type, radius, 0.0,
                  smooth, anisotropy, motion_len)
        return {
            "blur_type": blur_type,
            "radius": round(radius, 2),
            "angle": 0.0,
            "smooth": smooth,
        }

    def _sharpness(self, g: np.ndarray) -> float:
        """Kernel-metric analog: fraction of radial power-spectrum energy above a
        mid-frequency cutoff, scaled to 0..1 (1 = sharp). Blur removes high
        frequencies, so a blurred image scores low."""
        h, w = g.shape
        win = g * np.outer(np.hanning(h), np.hanning(w))
        power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        rmax = min(cy, cx)
        if rmax < 4:
            return 1.0
        radial = np.array([power[r == i].sum() for i in range(rmax)])
        total = radial[1:].sum()  # exclude DC
        cutoff = max(1, int(rmax * self._SHARPNESS_CUTOFF_FRAC))
        high = radial[cutoff:].sum()
        frac = high / (total + 1e-12)
        return float(np.clip(frac / self._SHARPNESS_FULL_SCALE, 0.0, 1.0))

    def assess(self, gray: np.ndarray) -> dict:
        """Full assessment: estimate() fields + sharpness, needs_deblur, method."""
        base = self.estimate(gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        g = self._downsample(self._to_uint8(gray)).astype(np.float64)

        sharpness = self._sharpness(g)
        sigma = self._noise_sigma(g)
        radius = float(base["radius"])

        if sigma >= self._NOISE_HIGH:
            method = "tikhonov"
        elif radius >= self._STRONG_BLUR_RADIUS:
            method = "tv"
        else:
            method = "wiener"

        # A crisp low-detail image stays protected here: sharp edges -> high
        # edge strength -> small estimated radius -> kernel_detected False. A
        # degenerate near-flat image can still slip this gate (flat -> large
        # radius), but the plugin's quality safeguard reverts it (a flat->flat
        # deconvolution yields no sharpness gain), so deblur never degrades it.
        kernel_detected = radius >= self._KERNEL_MIN_RADIUS
        needs_deblur = bool(sharpness < self._SHARPNESS_NEEDS_DEBLUR
                            and kernel_detected)

        signal_std = float(g.std())
        snr = signal_std / (sigma + 1e-6)
        dip = self._radial_spectrum_dip(g) if base["blur_type"] == "focus" else 0.0
        extras = self._extra_params(base["blur_type"], radius, method,
                                    float(base["smooth"]), snr, dip)

        return {
            **base,
            **extras,
            "sharpness": round(sharpness, 3),
            "needs_deblur": needs_deblur,
            "method": method,
        }

    @staticmethod
    def _extra_params(blur_type: str, radius: float, method: str,
                      smooth: float, snr: float, dip: float) -> dict:
        """Производные параметры деконволюции по первичным оценкам.

        snr — отношение std сигнала к сигме шума; dip — глубина спектрального
        провала диск-ядра (0 = нет провала). Возвращаемые ключи совпадают со
        схемой SmartDeblurPlugin.
        """
        # Шумный кадр (низкий SNR) требует сильнее регуляризацию.
        out = {
            "smooth": round(float(np.clip(
                smooth + 10.0 * (1.0 - min(snr / 20.0, 1.0)), 20.0, 60.0)), 1),
            "edge_taper": True,
        }
        if method in ("tv", "rl"):
            out["tv_iterations"] = int(np.clip(radius * 20.0, 60, 400))
        if blur_type == "focus":
            d = min(max(dip, 0.0), 1.0)
            # Чёткий провал = честное диск-ядро: края резкие (меньше feather),
            # коррекция краёв ядра сильнее.
            out["edge_feather"] = round(float(np.clip(25.0 - 20.0 * d,
                                                      5.0, 25.0)), 1)
            out["correction_strength"] = round(float(np.clip(d * 30.0,
                                                             0.0, 50.0)), 1)
        return out

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """Normalize any numeric image to uint8 [0, 255] for cv2 ops."""
        if img.dtype == np.uint8:
            return img
        a = img.astype(np.float64)
        mx = float(a.max()) if a.size else 0.0
        if mx <= 1.0:
            a = a * 255.0
        elif mx > 255.0:
            a = a / mx * 255.0
        return np.clip(a, 0, 255).astype(np.uint8)

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= self._MAX_DIM:
            return img
        scale = self._MAX_DIM / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _noise_sigma(self, g: np.ndarray) -> float:
        """Estimate noise sigma from the mean absolute Laplacian (MAD-style)."""
        lap = cv2.Laplacian(g, cv2.CV_64F)
        return float(np.sqrt(np.pi / 2) * np.mean(np.abs(lap)) / 6.0)

    def _estimate_smooth(self, g: np.ndarray) -> float:
        # Map estimated noise sigma (0..30) linearly to SmartDeblur's smooth
        # regularization range (20..60): noisier image -> stronger smoothing.
        sigma = self._noise_sigma(g)
        smooth = 20.0 + min(sigma, 30.0) * (40.0 / 30.0)
        return round(float(smooth), 1)

    def _log_spectrum(self, g: np.ndarray) -> np.ndarray:
        windowed = g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
        f = np.fft.fftshift(np.fft.fft2(windowed))
        return np.log1p(np.abs(f))

    def _spectral_anisotropy(self, g: np.ndarray):
        spec = self._log_spectrum(g)
        h, w = spec.shape
        cy, cx = h // 2, w // 2
        spec = spec - spec.mean()
        y, x = np.nonzero(spec > spec.std())
        if len(x) < 10:
            return 1.0, 0.0
        xs = x - cx
        ys = y - cy
        cov = np.cov(np.vstack([xs, ys]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]
        ratio = float(max(eigvals) / (min(eigvals) + 1e-6))
        angle = (np.degrees(np.arctan2(major[1], major[0])) + 90.0) % 180.0
        return ratio, angle

    def _motion_length_cepstrum(self, g: np.ndarray, angle_deg: float) -> float:
        spec = self._log_spectrum(g)
        cep = np.abs(np.fft.fftshift(np.fft.ifft2(spec)))
        h, w = cep.shape
        cy, cx = h // 2, w // 2
        cep[cy, cx] = 0
        theta = np.deg2rad(angle_deg)
        best = 0.0
        best_r = 0
        for r in range(3, min(cy, cx)):
            xr = int(round(cx + r * np.cos(theta)))
            yr = int(round(cy + r * np.sin(theta)))
            if 0 <= xr < w and 0 <= yr < h:
                if cep[yr, xr] > best:
                    best = cep[yr, xr]
                    best_r = r
        return float(best_r) if best > 0 else 0.0

    def _classify_defocus(self, g: np.ndarray):
        """Distinguish disk (focus) from gaussian blur; return (type, radius)."""
        radius = self._defocus_radius(g)
        dip = self._radial_spectrum_dip(g)
        blur_type = "focus" if dip > self._DISK_DIP_THRESHOLD else "gaussian"
        return blur_type, radius

    def _radial_spectrum_dip(self, g: np.ndarray) -> float:
        """Largest rebound after a local minimum in the radial power spectrum.

        Disk PSFs produce spectral nulls (a dip followed by a rebound); a pure
        gaussian falls off monotonically and yields a near-zero value.
        """
        h, w = g.shape
        win = g * np.outer(np.hanning(h), np.hanning(w))
        power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
        rmax = min(cy, cx)
        radial = np.array([power[r == i].mean() for i in range(rmax)])
        prof = np.convolve(np.log1p(radial), np.ones(5) / 5.0, mode="same")

        lo = max(3, int(rmax * 0.04))
        hi = int(rmax * 0.6)
        band = prof[lo:hi]
        best_dip = 0.0
        for i in range(1, len(band) - 1):
            if band[i] < band[i - 1] and band[i] <= band[i + 1]:
                rebound = float(band[i:].max() - band[i])
                best_dip = max(best_dip, rebound)
        return best_dip

    def _defocus_radius(self, g: np.ndarray) -> float:
        edges = cv2.Canny(g.astype(np.uint8), 50, 150)
        grad = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        grad += cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.abs(grad)
        edge_strength = mag[edges > 0].mean() if np.any(edges) else mag.mean()
        radius = float(np.clip(60.0 / (edge_strength + 1.0), 1.0, 15.0))
        return radius
