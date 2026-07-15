"""FFT-based deconvolution ported from SmartDeblur's DeconvolutionTool.cpp.

Operates on single-channel float64 images in [0, 1]. Three methods:
Wiener (fast), Tikhonov (Laplacian-regularized), Total Variation (iterative).
"""
from typing import Callable, Optional

import logging

import numpy as np

try:  # torch is optional; deconvolution must work without it.
    import torch
except ImportError:  # pragma: no cover - exercised on torch-less installs
    torch = None

log = logging.getLogger(__name__)

# Base of the exponential that maps the user-facing ``smooth`` value to the
# internal regularization weight K. Ported verbatim from SmartDeblur so the
# slider behaves identically: higher smooth -> stronger noise suppression.
_SMOOTH_BASE = 1.07


def _kernel_fft(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """Pad kernel to image shape and center its origin for circular conv."""
    kh, kw = kernel.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.rfft2(padded)


def edge_taper(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blend image borders toward a circularly-blurred copy to suppress ringing.

    FFT deconvolution assumes a periodic image; the discontinuity at the wrap
    boundary produces ringing. Tapering the borders toward the blurred version
    (which is consistent with the periodic model) removes that discontinuity.
    This mirrors the purpose of MATLAB's ``edgetaper``.
    """
    img = channel.astype(np.float64)
    h, w = img.shape
    kh, kw = kernel.shape
    blurred = np.fft.irfft2(
        np.fft.rfft2(img) * _kernel_fft(kernel, img.shape), s=img.shape)

    def _ramp(length: int, border: int) -> np.ndarray:
        weight = np.ones(length)
        border = min(border, length // 2)
        if border > 0:
            r = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, border))
            weight[:border] = r
            weight[-border:] = r[::-1]
        return weight

    alpha = np.outer(_ramp(h, kh), _ramp(w, kw))  # ~0 at borders, 1 interior
    return alpha * img + (1.0 - alpha) * blurred


def wiener_deconvolve(channel: np.ndarray, kernel: np.ndarray,
                      smooth: float) -> np.ndarray:
    K = (_SMOOTH_BASE ** smooth) / 10000.0
    H = _kernel_fft(kernel, channel.shape)
    G = np.fft.rfft2(channel.astype(np.float64))
    F = (np.conj(H) * G) / (np.abs(H) ** 2 + K)
    return np.fft.irfft2(F, s=channel.shape)


def richardson_lucy(channel: np.ndarray, kernel: np.ndarray,
                    iterations: int = 30,
                    progress_cb: Optional[Callable[[int], None]] = None,
                    cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Iterative Richardson-Lucy deconvolution (Poisson-noise MLE).

    Often recovers detail better than Wiener on real photographs at the cost of
    several FFT pairs per iteration.
    """
    img = np.clip(channel.astype(np.float64), 1e-6, None)
    H = _kernel_fft(kernel, img.shape)
    Hc = np.conj(H)  # adjoint = flipped PSF for circular convolution
    est = np.full(img.shape, 0.5, dtype=np.float64)
    for i in range(int(iterations)):
        if cancel_cb is not None and cancel_cb():
            break
        conv = np.fft.irfft2(np.fft.rfft2(est) * H, s=img.shape)
        conv = np.clip(conv, 1e-6, None)
        ratio = img / conv
        correction = np.fft.irfft2(np.fft.rfft2(ratio) * Hc, s=img.shape)
        est = np.clip(est * correction, 0.0, 1.0)
        if progress_cb is not None and i % 5 == 0:
            progress_cb(int(100 * i / max(iterations, 1)))
    return est


def tikhonov_deconvolve(channel: np.ndarray, kernel: np.ndarray,
                        smooth: float) -> np.ndarray:
    K = (_SMOOTH_BASE ** smooth) / 1000.0
    H = _kernel_fft(kernel, channel.shape)
    lap = np.zeros(channel.shape, dtype=np.float64)
    lap[0, 0] = 4.0
    lap[0, 1] = -1.0
    lap[1, 0] = -1.0
    lap[0, -1] = -1.0
    lap[-1, 0] = -1.0
    L = np.fft.rfft2(lap)
    G = np.fft.rfft2(channel.astype(np.float64))
    F = (np.conj(H) * G) / (np.abs(H) ** 2 + K * np.abs(L) ** 2)
    return np.fft.irfft2(F, s=channel.shape)


def tv_deconvolve(channel: np.ndarray, kernel: np.ndarray, smooth: float,
                  iterations: int = 500,
                  progress_cb: Optional[Callable[[int], None]] = None,
                  cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Total-Variation-regularized deconvolution via gradient descent."""
    eps = 0.004
    lam = (_SMOOTH_BASE ** smooth) / 100000.0
    tau = 1.9 / (1.0 + lam * 8.0 / eps)

    y = channel.astype(np.float64)
    H = _kernel_fft(kernel, y.shape)
    HtH = np.abs(H) ** 2
    Hty = np.fft.irfft2(np.conj(H) * np.fft.rfft2(y), s=y.shape)

    f = y.copy()
    for it in range(int(iterations)):
        if cancel_cb is not None and cancel_cb():
            break
        Hf = np.fft.irfft2(HtH * np.fft.rfft2(f), s=y.shape)
        data_grad = Hf - Hty

        # TV regularizer uses forward differences with an implicit zero
        # (Neumann) boundary, while the data term uses circular FFT
        # convolution. This mixed boundary only affects a 1px rim and matches
        # the original SmartDeblur behavior.
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gy[:-1, :] = f[1:, :] - f[:-1, :]
        gx[:, :-1] = f[:, 1:] - f[:, :-1]
        norm = 1.0 / np.sqrt(eps ** 2 + gx ** 2 + gy ** 2)
        gx *= norm
        gy *= norm

        fx = np.zeros_like(f)
        fy = np.zeros_like(f)
        fy[1:, :] = gy[1:, :] - gy[:-1, :]
        fx[:, 1:] = gx[:, 1:] - gx[:, :-1]
        divergence = -(fx + fy)

        f = f - tau * (data_grad + lam * divergence)

        if progress_cb is not None and it % 10 == 0:
            progress_cb(int(100 * it / max(iterations, 1)))

    return f


def deconvolve(channel: np.ndarray, kernel: np.ndarray, method: str = "wiener",
               smooth: float = 30.0, iterations: int = 500,
               taper: bool = True,
               progress_cb: Optional[Callable[[int], None]] = None,
               cancel_cb: Optional[Callable[[], bool]] = None,
               device: str = "cpu") -> np.ndarray:
    """Dispatch to the requested single-channel deconvolution method.

    On CUDA (``device`` starting with ``"cuda"`` and a working torch install)
    the GPU mirror is used; any failure falls back to the NumPy path below.
    ``taper`` applies edge-taper border preprocessing to suppress ringing for
    the frequency-domain methods (Wiener/Tikhonov). TV has its own border
    handling, so tapering is skipped there.
    """
    if (isinstance(device, str) and device.startswith("cuda")
            and torch is not None and torch.cuda.is_available()):
        try:
            from upscaler.engine.deconvolution_gpu import deconvolve_torch
            return deconvolve_torch(channel, kernel, method=method,
                                    smooth=smooth, iterations=iterations,
                                    taper=taper, progress_cb=progress_cb,
                                    cancel_cb=cancel_cb, device=device)
        except Exception as e:  # OOM, unsupported op, numerical issue, etc.
            log.warning("GPU deconvolution failed (%s); falling back to CPU", e)

    if method == "tv":
        # TV already pre-convolves borders internally.
        return tv_deconvolve(channel, kernel, smooth, iterations,
                             progress_cb, cancel_cb)

    work = edge_taper(channel, kernel) if taper else channel
    if method == "tikhonov":
        return tikhonov_deconvolve(work, kernel, smooth)
    if method == "rl":
        rl_iters = max(10, int(iterations) // 10)
        return richardson_lucy(work, kernel, rl_iters, progress_cb, cancel_cb)
    return wiener_deconvolve(work, kernel, smooth)
