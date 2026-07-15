"""Model weight management: download, cache, checksum, load/unload.

All models are stored in a single local directory (upscaler/models/models/).
On first use, models with a URL are downloaded there automatically.
For PyInstaller builds, bundle this directory with --add-data.
If the local dir is read-only (packaged app), downloads go to ~/.upscaler/models/.
"""

import hashlib
import logging
from pathlib import Path
from urllib.request import urlopen, Request

import torch

log = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "Real-ESRGAN-x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "sha256": "",
        "size_mb": 65,
    },
    "Real-ESRGAN-x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "sha256": "",
        "size_mb": 65,
    },
    "SwinIR-x2": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
        "filename": "SwinIR_x2.pth",
        "sha256": "",
        "size_mb": 50,
    },
    "SwinIR-x4": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        "filename": "SwinIR_x4.pth",
        "sha256": "",
        "size_mb": 50,
    },
    "HAT-S-x4": {
        "url": "https://huggingface.co/Phips/4xNomos8kSCHAT-S/resolve/main/4xNomos8kSCHAT-S.safetensors",
        "filename": "HAT-S_x4.safetensors",
        "sha256": "",
        "size_mb": 40,
    },
    "OmniSR-x2": {
        "url": "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors",
        "filename": "OmniSR_X2_DIV2K.safetensors",
        "sha256": "",
        "size_mb": 2,
    },
    "OmniSR-x4": {
        "url": "https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors",
        "filename": "OmniSR_X4_DIV2K.safetensors",
        "sha256": "",
        "size_mb": 2,
    },
    "DAT-x2": {
        "url": "https://github.com/zhengchen1999/DAT/releases/download/v1.0/DAT_x2.pth",
        "filename": "DAT_x2.pth",
        "sha256": "",
        "size_mb": 45,
    },
    "DAT-x4": {
        "url": "https://huggingface.co/Phips/4xNomos8kDAT/resolve/main/4xNomos8kDAT.safetensors",
        "filename": "4xNomos8kDAT.safetensors",
        "sha256": "",
        "size_mb": 154,
    },
    "SCUNet": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "filename": "scunet_color_real_psnr.pth",
        "sha256": "",
        "size_mb": 25,
    },
    "NAFNet-SIDD": {
        "url": "",
        "filename": "NAFNet-SIDD-width64.pth",
        "sha256": "",
        "size_mb": 464,
    },
    "NAFNet-GoPro": {
        "url": "",
        "filename": "NAFNet-GoPro-width64.pth",
        "sha256": "",
        "size_mb": 272,
    },
    "DDColor-artistic": {
        "url": "https://huggingface.co/piddnad/ddcolor_artistic/resolve/main/pytorch_model.bin",
        "filename": "ddcolor_artistic.pth",
        "sha256": "",
        "size_mb": 912,
    },
    "DDColor-modelscope": {
        "url": "https://huggingface.co/piddnad/ddcolor_modelscope/resolve/main/pytorch_model.bin",
        "filename": "ddcolor_modelscope.pth",
        "sha256": "",
        "size_mb": 912,
    },
    "DeOldify-stable": {
        "url": "",
        "filename": "ColorizeStable_gen.pth",
        "sha256": "",
        "size_mb": 874,
    },
    "DeOldify-artistic": {
        "url": "",
        "filename": "ColorizeArtistic_gen.pth",
        "sha256": "",
        "size_mb": 255,
    },
    "DeOldify-video": {
        "url": "",
        "filename": "ColorizeVideo_gen.pth",
        "sha256": "",
        "size_mb": 874,
    },
    "ColorMNet": {
        "url": "",
        "filename": "DINOv2FeatureV6_LocalAtten_s2_154000.pth",
        "sha256": "",
        "size_mb": 495,
    },
    # --- ICEdit: quantized FLUX.1-Fill-dev components + LoRAs ---
    "FLUX-Fill-GGUF-Q4": {
        "url": "https://huggingface.co/city96/FLUX.1-Fill-dev-gguf/resolve/main/flux1-fill-dev-Q4_K_S.gguf",
        "filename": "flux1-fill-dev-Q4_K_S.gguf",
        "sha256": "",
        "size_mb": 6900,
    },
    "FLUX-Fill-GGUF-Q5": {
        "url": "https://huggingface.co/city96/FLUX.1-Fill-dev-gguf/resolve/main/flux1-fill-dev-Q5_K_S.gguf",
        "filename": "flux1-fill-dev-Q5_K_S.gguf",
        "sha256": "",
        "size_mb": 8400,
    },
    "FLUX-T5-GGUF": {
        "url": "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_M.gguf",
        "filename": "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
        "sha256": "",
        "size_mb": 3400,
    },
    "FLUX-CLIP-L": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "filename": "clip_l.safetensors",
        "sha256": "",
        "size_mb": 246,
    },
    "FLUX-VAE": {
        # Ungated mirror of the FLUX autoencoder (the gated black-forest-labs
        # repos return 401 even for the Apache-2.0 schnell VAE).
        "url": "https://huggingface.co/second-state/FLUX.1-schnell-GGUF/resolve/main/ae.safetensors",
        "filename": "flux_ae.safetensors",
        "sha256": "",
        "size_mb": 335,
    },
    "ICEdit-MoE-LoRA": {
        "url": "https://huggingface.co/sanaka87/ICEdit-MoE-LoRA/resolve/main/pytorch_lora_weights.safetensors",
        "filename": "ICEdit-MoE-LoRA.safetensors",
        "sha256": "",
        "size_mb": 280,
    },
    "ICEdit-normal-LoRA": {
        "url": "https://huggingface.co/RiverZ/normal-lora/resolve/main/pytorch_lora_weights.safetensors",
        "filename": "ICEdit-normal-LoRA.safetensors",
        "sha256": "",
        "size_mb": 280,
    },
    # --- Face restoration ---
    "CodeFormer-ONNX": {
        # TODO: set verified non-gated CodeFormer .onnx URL
        # All known HuggingFace redistributions require auth (401).
        # Set this to a publicly accessible .onnx mirror when one is confirmed.
        "url": "",
        "filename": "codeformer.onnx",
        "sha256": "",
        "size_mb": 360,
    },
    "YuNet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "sha256": "",
        "size_mb": 1,
    },
}


# Primary model storage: project-local directory (bundled with PyInstaller)
LOCAL_MODELS_DIR = Path(__file__).parent / "models"


class ModelManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_model = None

    def list_models(self) -> list[str]:
        return list(MODEL_REGISTRY.keys())

    def get_model_info(self, name: str) -> dict | None:
        return MODEL_REGISTRY.get(name)

    def get_model_path(self, name: str) -> Path:
        """Find model file. Checks local dir first, then user cache dir."""
        info = MODEL_REGISTRY.get(name, {})
        filename = info.get("filename", f"{name}.pth")
        # Check project-local directory first (works both in dev and packaged app)
        local_path = LOCAL_MODELS_DIR / filename
        if local_path.exists():
            return local_path
        # Fallback: user cache directory (~/.upscaler/models/)
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            return cache_path
        # Not found yet — return local dir as preferred download target
        return local_path

    def _download_path(self, name: str) -> Path:
        """Choose where to download: local dir if writable, else cache dir."""
        info = MODEL_REGISTRY.get(name, {})
        filename = info.get("filename", f"{name}.pth")
        local_path = LOCAL_MODELS_DIR / filename
        # Try local dir first (so model stays in project for PyInstaller builds)
        try:
            LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = LOCAL_MODELS_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
            return local_path
        except OSError:
            # Local dir is read-only (packaged app) — use cache dir
            return self.cache_dir / filename

    def is_downloaded(self, name: str) -> bool:
        return self.get_model_path(name).exists()

    def download(self, name: str, progress_cb=None, max_retries: int = 3) -> Path:
        info = MODEL_REGISTRY.get(name)
        if not info:
            raise ValueError(f"Unknown model: {name}")
        # Check if already exists anywhere
        existing = self.get_model_path(name)
        if existing.exists():
            return existing
        if not info.get("url"):
            raise FileNotFoundError(
                f"Модель '{name}' не найдена локально и не имеет URL для загрузки. "
                f"Ожидаемый путь: {existing}"
            )

        # Download to the best writable location
        path = self._download_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Downloading {name} to {path}")

        url = info["url"]
        size_bytes = info.get("size_mb", 0) * 1024 * 1024
        last_error = None

        for attempt in range(max_retries):
            try:
                start_byte = 0
                if path.with_suffix(".part").exists():
                    start_byte = path.with_suffix(".part").stat().st_size
                req = Request(url)
                req.add_header("User-Agent", "Upscaler/1.0 (Python)")
                if start_byte > 0:
                    req.add_header("Range", f"bytes={start_byte}-")
                response = urlopen(req, timeout=60)
                total = int(response.headers.get("Content-Length", size_bytes)) + start_byte
                mode = "ab" if start_byte > 0 else "wb"
                with open(path.with_suffix(".part"), mode) as f:
                    downloaded = start_byte
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            pct = int(downloaded / total * 100) if total > 0 else 0
                            progress_cb(pct, downloaded, total)

                expected_sha = info.get("sha256", "")
                if expected_sha:
                    actual_sha = self._compute_sha256(path.with_suffix(".part"))
                    if actual_sha != expected_sha:
                        path.with_suffix(".part").unlink()
                        raise ValueError(f"Checksum mismatch for {name}")

                path.with_suffix(".part").rename(path)
                log.info(f"Downloaded {name} → {path}")
                return path
            except Exception as e:
                last_error = e
                log.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Не удалось скачать модель '{name}' после {max_retries} попыток: {last_error}\n"
            f"Вы можете скачать файл вручную:\n"
            f"  URL: {url}\n"
            f"  Сохраните как: {path}"
        )

    def load_weights(self, name: str, device: str = "auto") -> dict:
        path = self.get_model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Model not downloaded: {name}")
        dev = self.get_device(device)
        self.unload_current()
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(path), device=dev)
        else:
            state_dict = torch.load(path, map_location=dev, weights_only=True)
        self._loaded_model = name
        return state_dict

    def unload_current(self):
        if self._loaded_model:
            self._loaded_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_device(self, preference: str = "auto") -> str:
        if preference == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if preference.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return preference

    def _compute_sha256(self, path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
