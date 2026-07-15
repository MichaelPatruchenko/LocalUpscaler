"""Assemble a quantized FLUX.1-Fill-dev + ICEdit-LoRA diffusers pipeline.

All heavy imports happen inside ``load_flux_fill`` so importing this module is
cheap and dependency-free.

The pipeline is assembled from individual NON-GATED components so the gated
``black-forest-labs/FLUX.1-Fill-dev`` repo (and its HF token / licence) is never
required:

* transformer - local quantized GGUF (downloaded via ``ModelManager``); diffusers
  infers the ``flux-fill`` architecture straight from the checkpoint.
* VAE - ``ae.safetensors`` from the Apache-2.0 ``FLUX.1-schnell`` repo
  (downloaded via ``ModelManager``).
* CLIP-L text encoder + tokenizer - ``openai/clip-vit-large-patch14`` (HF hub).
* T5-XXL text encoder - quantized GGUF from ``city96/t5-v1_1-xxl-encoder-gguf``
  (HF hub); tokenizer from ``google/t5-v1_1-xxl``.
* scheduler - constructed inline with the FLUX flow-match configuration.

Verified against diffusers 0.38 / transformers 5.12. The function raises on any
missing dependency or weight; ``ICEditPlugin`` catches that and skips editing.
On a machine without CUDA the pipeline runs on the CPU (offload settings, which
require a GPU, are ignored).
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Bundled diffusers-format configs so the components load without the gated
# FLUX.1-Fill-dev repo. The Fill transformer config (in_channels=384,
# out_channels=64, guidance_embeds=true) exists in no ungated repo, so we ship
# it; the VAE config is the standard FLUX autoencoder.
_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_TRANSFORMER_CONFIG_DIR = str(_CONFIG_DIR / "transformer")
_VAE_CONFIG_DIR = str(_CONFIG_DIR / "vae")

# Quantized FLUX-Fill transformer registry keys, by requested quant level.
_TRANSFORMER_KEYS = {
    "q5": "FLUX-Fill-GGUF-Q5",
    "q4": "FLUX-Fill-GGUF-Q4",
    "nf4": "FLUX-Fill-GGUF-Q4",
}
# When resolving the transformer, prefer a file that is already present so a
# user who downloaded one quant level is not forced to fetch another.
_TRANSFORMER_PREFER = ["FLUX-Fill-GGUF-Q5", "FLUX-Fill-GGUF-Q4"]
_LORA_KEY = {"moe": "ICEdit-MoE-LoRA", "normal": "ICEdit-normal-LoRA"}

# Non-gated text-encoder sources (fetched + cached via the HF hub).
_CLIP_REPO = "openai/clip-vit-large-patch14"
_T5_GGUF_REPO = "city96/t5-v1_1-xxl-encoder-gguf"
_T5_GGUF_FILE = "t5-v1_1-xxl-encoder-Q5_K_M.gguf"
_T5_TOKENIZER_REPO = "google/t5-v1_1-xxl"

# FLUX flow-match scheduler configuration (matches FLUX.1-Fill-dev).
_FLUX_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}


def _resolve_transformer_key(model_manager, quant: str) -> str:
    """Prefer an already-present transformer; else the requested quant; else Q5."""
    for key in _TRANSFORMER_PREFER:
        if model_manager.is_downloaded(key):
            return key
    return _TRANSFORMER_KEYS.get(quant, "FLUX-Fill-GGUF-Q5")


def _ensure(model_manager, key: str) -> str:
    """Return the local path to a registry weight, downloading it if absent."""
    if not model_manager.is_downloaded(key):
        log.info("ICEdit: downloading %s", key)
        model_manager.download(key)
    return str(model_manager.get_model_path(key))


def _make_gguf_version_detectable() -> None:
    """Work around a transformers/gguf packaging mismatch.

    gguf >= 0.19 dropped its ``__version__`` attribute and is missing from
    transformers' package-distribution map, so transformers reports its version
    as ``'N/A'`` and ``is_gguf_available()`` crashes on ``version.parse('N/A')``
    while loading the GGUF T5 encoder. Setting ``gguf.__version__`` from the
    installed package metadata makes transformers' fallback detection succeed.
    """
    try:
        import gguf
        if not getattr(gguf, "__version__", None):
            import importlib.metadata as md
            try:
                gguf.__version__ = md.version("gguf")
            except Exception:
                gguf.__version__ = "0.10.0"
        from transformers.utils import import_utils
        if hasattr(import_utils.is_gguf_available, "cache_clear"):
            import_utils.is_gguf_available.cache_clear()
    except Exception as exc:  # never block loading on the shim itself
        log.debug("gguf version shim skipped: %s", exc)


def _resolve_lora_key(variant: str) -> str:
    """Ключ реестра LoRA по варианту; незнакомое -> Normal (не MoE)."""
    return _LORA_KEY.get(variant, "ICEdit-normal-LoRA")


def _assert_lora_applied(pipe, lora_key: str) -> int:
    """Проверить, что LoRA реально навесилась на трансформер.

    Стандартный load_lora_weights молча игнорирует несовпавшие ключи
    (например, MoE-LoRA с кастомным гейтингом) — тогда пайплайн работает
    БЕЗ правок и «не следует промту». Возвращает число LoRA-параметров;
    0 -> RuntimeError.
    """
    n_lora = sum(1 for name, _ in pipe.transformer.named_parameters()
                 if "lora" in name.lower())
    if n_lora == 0:
        raise RuntimeError(
            f"ICEdit LoRA '{lora_key}' не применилась к трансформеру "
            "(несовместимый формат; MoE-LoRA требует кастомного кода "
            "ICEdit и не поддерживается стандартной загрузкой)")
    log.info("ICEdit: LoRA '%s' применена (%d lora-параметров)",
             lora_key, n_lora)
    return n_lora


def load_flux_fill(variant: str, quant: str, offload: str, device: str,
                   model_manager) -> object:
    """Build a FLUX-Fill pipeline (local GGUF transformer/VAE + HF text encoders).

    Raises on any missing dependency or weight so the caller can skip ICEdit.
    """
    # Validate cheap preconditions before importing heavy optional deps, so the
    # contract "no model_manager -> RuntimeError" holds even on installs without
    # diffusers/transformers (where the import would otherwise raise first).
    if model_manager is None:
        raise RuntimeError("model_manager is required to locate ICEdit weights")

    import torch
    from diffusers import (
        FluxFillPipeline, FluxTransformer2DModel, AutoencoderKL,
        FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
    )
    from transformers import (
        CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
    )

    use_cuda = bool(device and str(device).startswith("cuda")
                    and torch.cuda.is_available())
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32

    # 1. Transformer - local quantized GGUF + bundled Fill config (no gated repo).
    transformer_key = _resolve_transformer_key(model_manager, quant)
    transformer = FluxTransformer2DModel.from_single_file(
        _ensure(model_manager, transformer_key),
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        config=_TRANSFORMER_CONFIG_DIR, subfolder="",
        torch_dtype=compute_dtype,
    )

    # 2. VAE - local single file (Apache-2.0 FLUX.1-schnell autoencoder) +
    #    bundled config.
    vae = AutoencoderKL.from_single_file(
        _ensure(model_manager, "FLUX-VAE"),
        config=_VAE_CONFIG_DIR, subfolder="",
        torch_dtype=compute_dtype)

    # 3. Text encoders + tokenizers - non-gated HF hub.
    text_encoder = CLIPTextModel.from_pretrained(
        _CLIP_REPO, torch_dtype=compute_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(_CLIP_REPO)
    _make_gguf_version_detectable()
    text_encoder_2 = T5EncoderModel.from_pretrained(
        _T5_GGUF_REPO, gguf_file=_T5_GGUF_FILE, torch_dtype=compute_dtype)
    tokenizer_2 = T5TokenizerFast.from_pretrained(_T5_TOKENIZER_REPO)

    # 4. Scheduler.
    scheduler = FlowMatchEulerDiscreteScheduler(**_FLUX_SCHEDULER_CONFIG)

    pipe = FluxFillPipeline(
        scheduler=scheduler, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        transformer=transformer,
    )

    # 5. ICEdit LoRA (с верификацией: тихая работа без LoRA недопустима).
    lora_key = _resolve_lora_key(variant)
    lora_path = _ensure(model_manager, lora_key)
    try:
        pipe.load_lora_weights(lora_path)
    except Exception as exc:
        raise RuntimeError(
            f"ICEdit LoRA '{lora_key}' не загрузилась: {exc}") from exc
    _assert_lora_applied(pipe, lora_key)

    # 6. Placement / offload. CPU-only leaves the pipeline where it was
    #    assembled (already on CPU); calling .to("cpu") raises a meta-tensor
    #    error on the GGUF-quantized transformer. Offload modes require CUDA.
    if use_cuda:
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "none":
            pipe.to(device)
        else:  # "model" (default)
            pipe.enable_model_cpu_offload()
    return pipe
