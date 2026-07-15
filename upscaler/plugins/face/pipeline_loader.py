"""Isolated, guarded ONNX runtime loader for CodeFormer.

All onnxruntime imports live here so the rest of the app runs without it.
"""
import logging

log = logging.getLogger(__name__)


def load_session(model_path: str, device: str = "auto"):
    """Build an onnxruntime InferenceSession.

    Uses the CUDA execution provider when the resolved ``device`` is not CPU and
    onnxruntime exposes it (requires the ``onnxruntime-gpu`` package); otherwise
    CPU only. CPU is always appended as a fallback provider.
    """
    import onnxruntime as ort  # raises if not installed -> caller skips face step
    want_cuda = not (isinstance(device, str) and device.startswith("cpu"))
    providers = []
    available = ort.get_available_providers()
    if want_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    log.info("Loading CodeFormer ONNX with providers=%s (device=%s)", providers, device)
    return ort.InferenceSession(str(model_path), providers=providers)
