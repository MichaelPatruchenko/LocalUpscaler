"""Torch (GPU) mirror of the NumPy deconvolution kernels in deconvolution.py.

Each function reproduces the math of its NumPy counterpart so results match
within floating-point tolerance. Used as the CUDA fast-path; the NumPy module
remains the reference and CPU/fallback implementation. ``device="cpu"`` is
valid and exercised by tests so this path runs in CI without a GPU.
"""
import math
from typing import Callable, Optional

import numpy as np
import torch

# Must stay identical to deconvolution._SMOOTH_BASE.
_SMOOTH_BASE = 1.07


def _kernel_fft_t(kernel: torch.Tensor, shape, device) -> torch.Tensor:
    kh, kw = kernel.shape
    padded = torch.zeros(tuple(shape), dtype=torch.float64, device=device)
    padded[:kh, :kw] = kernel
    padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
    return torch.fft.rfft2(padded)


def _edge_taper_t(img: torch.Tensor, kernel: torch.Tensor, device) -> torch.Tensor:
    h, w = img.shape
    kh, kw = kernel.shape
    blurred = torch.fft.irfft2(
        torch.fft.rfft2(img) * _kernel_fft_t(kernel, (h, w), device), s=(h, w))

    def _ramp(length: int, border: int) -> torch.Tensor:
        weight = torch.ones(length, dtype=torch.float64, device=device)
        border = min(border, length // 2)
        if border > 0:
            r = 0.5 - 0.5 * torch.cos(
                torch.linspace(0.0, math.pi, border, dtype=torch.float64, device=device))
            weight[:border] = r
            weight[-border:] = torch.flip(r, dims=[0])
        return weight

    alpha = torch.outer(_ramp(h, kh), _ramp(w, kw))
    return alpha * img + (1.0 - alpha) * blurred


def _wiener_t(channel, kernel, smooth, device):
    K = (_SMOOTH_BASE ** smooth) / 10000.0
    H = _kernel_fft_t(kernel, channel.shape, device)
    G = torch.fft.rfft2(channel)
    F = (torch.conj(H) * G) / (torch.abs(H) ** 2 + K)
    return torch.fft.irfft2(F, s=channel.shape)


def _tikhonov_t(channel, kernel, smooth, device):
    K = (_SMOOTH_BASE ** smooth) / 1000.0
    H = _kernel_fft_t(kernel, channel.shape, device)
    lap = torch.zeros(channel.shape, dtype=torch.float64, device=device)
    lap[0, 0] = 4.0
    lap[0, 1] = -1.0
    lap[1, 0] = -1.0
    lap[0, -1] = -1.0
    lap[-1, 0] = -1.0
    L = torch.fft.rfft2(lap)
    G = torch.fft.rfft2(channel)
    F = (torch.conj(H) * G) / (torch.abs(H) ** 2 + K * torch.abs(L) ** 2)
    return torch.fft.irfft2(F, s=channel.shape)


def _richardson_lucy_t(channel, kernel, iterations, progress_cb, cancel_cb, device):
    img = torch.clamp(channel, min=1e-6)
    H = _kernel_fft_t(kernel, img.shape, device)
    Hc = torch.conj(H)
    est = torch.full(img.shape, 0.5, dtype=torch.float64, device=device)
    for i in range(int(iterations)):
        if cancel_cb is not None and cancel_cb():
            break
        conv = torch.fft.irfft2(torch.fft.rfft2(est) * H, s=img.shape)
        conv = torch.clamp(conv, min=1e-6)
        ratio = img / conv
        correction = torch.fft.irfft2(torch.fft.rfft2(ratio) * Hc, s=img.shape)
        est = torch.clamp(est * correction, 0.0, 1.0)
        if progress_cb is not None and i % 5 == 0:
            progress_cb(int(100 * i / max(iterations, 1)))
    return est


def _tv_t(channel, kernel, smooth, iterations, progress_cb, cancel_cb, device):
    eps = 0.004
    lam = (_SMOOTH_BASE ** smooth) / 100000.0
    tau = 1.9 / (1.0 + lam * 8.0 / eps)

    y = channel
    H = _kernel_fft_t(kernel, y.shape, device)
    HtH = torch.abs(H) ** 2
    Hty = torch.fft.irfft2(torch.conj(H) * torch.fft.rfft2(y), s=y.shape)

    f = y.clone()
    for it in range(int(iterations)):
        if cancel_cb is not None and cancel_cb():
            break
        Hf = torch.fft.irfft2(HtH * torch.fft.rfft2(f), s=y.shape)
        data_grad = Hf - Hty

        gx = torch.zeros_like(f)
        gy = torch.zeros_like(f)
        gy[:-1, :] = f[1:, :] - f[:-1, :]
        gx[:, :-1] = f[:, 1:] - f[:, :-1]
        norm = 1.0 / torch.sqrt(eps ** 2 + gx ** 2 + gy ** 2)
        gx = gx * norm
        gy = gy * norm

        fx = torch.zeros_like(f)
        fy = torch.zeros_like(f)
        fy[1:, :] = gy[1:, :] - gy[:-1, :]
        fx[:, 1:] = gx[:, 1:] - gx[:, :-1]
        divergence = -(fx + fy)

        f = f - tau * (data_grad + lam * divergence)

        if progress_cb is not None and it % 10 == 0:
            progress_cb(int(100 * it / max(iterations, 1)))
    return f


def deconvolve_torch(channel: np.ndarray, kernel: np.ndarray,
                     method: str = "wiener", smooth: float = 30.0,
                     iterations: int = 500, taper: bool = True,
                     progress_cb: Optional[Callable[[int], None]] = None,
                     cancel_cb: Optional[Callable[[], bool]] = None,
                     device: str = "cuda") -> np.ndarray:
    """Torch mirror of deconvolution.deconvolve. Returns a float64 ndarray."""
    dev = torch.device(device)
    ch = torch.from_numpy(np.ascontiguousarray(channel, dtype=np.float64)).to(dev)
    ker = torch.from_numpy(np.ascontiguousarray(kernel, dtype=np.float64)).to(dev)

    if method == "tv":
        out = _tv_t(ch, ker, smooth, iterations, progress_cb, cancel_cb, dev)
    else:
        work = _edge_taper_t(ch, ker, dev) if taper else ch
        if method == "tikhonov":
            out = _tikhonov_t(work, ker, smooth, dev)
        elif method == "rl":
            rl_iters = max(10, int(iterations) // 10)
            out = _richardson_lucy_t(work, ker, rl_iters, progress_cb, cancel_cb, dev)
        else:
            out = _wiener_t(work, ker, smooth, dev)

    return out.detach().cpu().numpy().astype(np.float64)
