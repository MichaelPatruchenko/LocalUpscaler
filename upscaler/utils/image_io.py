"""Image reading, writing, and thumbnail generation.

Supports: PNG, JPEG, TIFF, BMP, WebP, OpenEXR, HDR, RAW (CR2/NEF/ARW).
All images returned as RGB numpy arrays (HxWxC).

Uses cv2.imdecode/imencode with raw bytes to handle non-ASCII file paths on
Windows (cv2.imread/imwrite silently fail when the path contains characters
outside the system code page).
"""

import cv2
import numpy as np
from pathlib import Path

RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng", ".raf", ".orf"}
EXR_EXTENSIONS = {".exr"}
HDR_EXTENSIONS = {".hdr"}


class ImageReadError(Exception):
    pass


class ImageWriteError(Exception):
    pass


def read_image(path: Path) -> tuple[np.ndarray, dict]:
    """Read an image file, return (rgb_array, metadata_dict).

    Metadata keys: format, bit_depth, width, height, channels, icc_profile (bytes|None).
    """
    path = Path(path)
    if not path.exists():
        raise ImageReadError(f"File not found: {path}")

    ext = path.suffix.lower()
    icc_profile = None

    try:
        if ext in RAW_EXTENSIONS:
            img, icc_profile = _read_raw(path)
        elif ext in EXR_EXTENSIONS:
            img = _read_exr(path)
        elif ext in HDR_EXTENSIONS:
            img = _read_hdr(path)
        else:
            img = _read_standard(path)
            icc_profile = _extract_icc(path)
    except ImageReadError:
        raise
    except Exception as e:
        raise ImageReadError(f"Failed to read {path}: {e}") from e

    if img is None:
        raise ImageReadError(f"Failed to decode: {path}")

    bit_depth = _detect_bit_depth(img)

    meta = {
        "format": ext,
        "bit_depth": bit_depth,
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if img.ndim == 3 else 1,
        "icc_profile": icc_profile,
    }
    return img, meta


def write_image(image: np.ndarray, path: Path, format: str | None = None,
                quality: int = 95):
    """Write image array to disk. Format inferred from extension if not given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = format or path.suffix.lstrip(".").lower()

    if ext == "exr":
        _write_exr(image, path)
        return

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image

    if ext in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == "png":
        ok, buf = cv2.imencode(".png", bgr)
    elif ext in ("tiff", "tif"):
        ok, buf = cv2.imencode(".tiff", bgr)
    else:
        dot_ext = "." + ext if not ext.startswith(".") else ext
        ok, buf = cv2.imencode(dot_ext, bgr)

    if not ok or buf is None:
        raise ImageWriteError(f"Failed to encode image for: {path}")

    path.write_bytes(buf.tobytes())


def make_thumbnail(image: np.ndarray, max_size: int = 150) -> np.ndarray:
    """Resize image so the longest side equals max_size, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_standard(path: Path) -> np.ndarray:
    """Read PNG/JPEG/TIFF/BMP/WebP via OpenCV using imdecode (handles non-ASCII paths)."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ImageReadError(f"OpenCV failed to decode: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_hdr(path: Path) -> np.ndarray:
    """Read Radiance HDR files via OpenCV."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ImageReadError(f"Failed to read HDR: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_raw(path: Path) -> tuple[np.ndarray, bytes | None]:
    """Read RAW camera files via rawpy."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(output_bps=16)
    return rgb, None


def _read_exr(path: Path) -> np.ndarray:
    """Read OpenEXR files."""
    import OpenEXR
    import Imath
    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        dw = header["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = []
        for ch in ("R", "G", "B"):
            raw = exr.channel(ch, pt)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
            channels.append(arr)
        return np.stack(channels, axis=-1)
    finally:
        exr.close()


def _write_exr(image: np.ndarray, path: Path):
    """Write float32 RGB array as OpenEXR."""
    import OpenEXR
    import Imath
    h, w = image.shape[:2]
    img = image.astype(np.float32)
    header = OpenEXR.Header(w, h)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        "R": Imath.Channel(pixel_type),
        "G": Imath.Channel(pixel_type),
        "B": Imath.Channel(pixel_type),
    }
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels({
            "R": img[:, :, 0].tobytes(),
            "G": img[:, :, 1].tobytes(),
            "B": img[:, :, 2].tobytes(),
        })
    finally:
        out.close()


def _extract_icc(path: Path) -> bytes | None:
    """Extract ICC profile from image using Pillow."""
    try:
        from PIL import Image
        with Image.open(path) as pil_img:
            return pil_img.info.get("icc_profile")
    except Exception:
        return None


def _detect_bit_depth(img: np.ndarray) -> int:
    if img.dtype == np.uint8:
        return 8
    elif img.dtype == np.uint16:
        return 16
    elif img.dtype in (np.float32, np.float64):
        return 32
    return 8
