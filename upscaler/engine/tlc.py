"""Test-time Local Converter (TLC) for NAFNet-style global pooling.

NAFNet's Simplified Channel Attention uses a global average pool. Its statistics
are computed over the whole input, so they shift between training (small fixed
patches) and inference (large/native-resolution tiles) — causing instability and
out-of-range "garbage" output on some tiles. TLC replaces the global pool with a
LOCAL windowed average sized to the training patch, removing the train/test gap.

Reference: Chu et al., "Improving Image Restoration by Revisiting Global
Information Aggregation" (TLC), and the official NAFNet ``local_arch.py``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAvgPool2d(nn.Module):
    """Drop-in replacement for ``nn.AdaptiveAvgPool2d(1)`` that averages over a
    local window proportional to the training patch size instead of globally.

    The window is a FIXED size (``base_size``, ≈1.5× the training patch),
    independent of the input size — matching the official NAFNetLocal, which
    freezes the kernel from a training-size dummy forward. When the window
    covers the whole input it degenerates to a global average (the original
    behaviour), so small inputs are unaffected.
    """

    def __init__(self, base_size, train_size=(1, 3, 256, 256)):
        super().__init__()
        if isinstance(base_size, int):
            base_size = (base_size, base_size)
        self.base_size = base_size
        self.train_size = train_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        # Fixed local window (≈train patch), clamped to the input size.
        k1 = min(max(1, int(self.base_size[0])), h)
        k2 = min(max(1, int(self.base_size[1])), w)
        if k1 >= h and k2 >= w:
            return F.adaptive_avg_pool2d(x, 1)
        # Local mean via an integral image (summed-area table) — O(N).
        s = x.cumsum(dim=-1).cumsum(dim=-2)
        s = F.pad(s, (1, 0, 1, 0))
        s1 = s[:, :, :-k1, :-k2]
        s2 = s[:, :, :-k1, k2:]
        s3 = s[:, :, k1:, :-k2]
        s4 = s[:, :, k1:, k2:]
        out = (s4 + s1 - s2 - s3) / (k1 * k2)
        # Pad back to the input size so it multiplies the features per-pixel.
        _h, _w = out.shape[-2], out.shape[-1]
        out = F.pad(
            out,
            ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2),
            mode="replicate",
        )
        return out


def apply_tlc(model: nn.Module, base_size=384, train_size=(1, 3, 256, 256)) -> nn.Module:
    """Recursively replace every ``nn.AdaptiveAvgPool2d`` in ``model`` with a
    :class:`LocalAvgPool2d`. Returns the (in-place) modified model.

    ``base_size`` defaults to 1.5× the NAFNet training patch (256) per the
    official NAFNetLocal configuration.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(model, name, LocalAvgPool2d(base_size, train_size))
        else:
            apply_tlc(child, base_size, train_size)
    return model
