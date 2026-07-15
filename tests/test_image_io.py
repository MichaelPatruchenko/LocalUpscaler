import numpy as np
import pytest
from pathlib import Path


class TestReadImage:
    def test_read_png(self, tmp_path, sample_rgb_uint8):
        import cv2
        p = tmp_path / "test.png"
        cv2.imwrite(str(p), cv2.cvtColor(sample_rgb_uint8, cv2.COLOR_RGB2BGR))
        from upscaler.utils.image_io import read_image
        img, meta = read_image(p)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert meta["bit_depth"] == 8

    def test_read_jpeg(self, tmp_path, sample_rgb_uint8):
        import cv2
        p = tmp_path / "test.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(sample_rgb_uint8, cv2.COLOR_RGB2BGR))
        from upscaler.utils.image_io import read_image
        img, meta = read_image(p)
        assert img.shape == (64, 64, 3)
        assert meta["format"] == ".jpg"

    def test_read_tiff_16bit(self, tmp_path):
        import cv2
        img16 = np.random.default_rng(42).integers(0, 65536, (32, 32, 3), dtype=np.uint16)
        p = tmp_path / "test.tiff"
        cv2.imwrite(str(p), img16)
        from upscaler.utils.image_io import read_image
        img, meta = read_image(p)
        assert img.dtype == np.uint16
        assert meta["bit_depth"] == 16

    def test_read_nonexistent_raises(self):
        from upscaler.utils.image_io import read_image, ImageReadError
        with pytest.raises(ImageReadError):
            read_image(Path("/nonexistent/image.png"))

    def test_read_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "corrupt.png"
        p.write_bytes(b"not an image")
        from upscaler.utils.image_io import read_image, ImageReadError
        with pytest.raises(ImageReadError):
            read_image(p)


class TestWriteImage:
    def test_write_png_uint8(self, tmp_path, sample_rgb_uint8):
        from upscaler.utils.image_io import write_image, read_image
        out = tmp_path / "out.png"
        write_image(sample_rgb_uint8, out)
        assert out.exists()
        img, meta = read_image(out)
        assert img.dtype == np.uint8

    def test_write_tiff_16bit(self, tmp_path):
        from upscaler.utils.image_io import write_image, read_image
        img16 = np.random.default_rng(42).integers(0, 65536, (32, 32, 3), dtype=np.uint16)
        out = tmp_path / "out.tiff"
        write_image(img16, out, format="tiff")
        img, meta = read_image(out)
        assert img.dtype == np.uint16

    def test_write_auto_format_from_extension(self, tmp_path, sample_rgb_uint8):
        from upscaler.utils.image_io import write_image
        out = tmp_path / "out.jpg"
        write_image(sample_rgb_uint8, out)
        assert out.exists()


class TestMakeThumbnail:
    def test_thumbnail_size(self, sample_rgb_uint8):
        from upscaler.utils.image_io import make_thumbnail
        thumb = make_thumbnail(sample_rgb_uint8, max_size=32)
        assert max(thumb.shape[:2]) == 32

    def test_thumbnail_preserves_aspect(self):
        from upscaler.utils.image_io import make_thumbnail
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        thumb = make_thumbnail(img, max_size=50)
        assert thumb.shape == (25, 50, 3)
