"""Comprehensive unit tests for ImageService.

Covers:
- Valid JPEG, PNG, WEBP processing.
- Magic bytes and file signature verification.
- Content-type / MIME consistency checks.
- Corrupted and empty files.
- Oversized upload protection.
- Decompression bomb detection.
- Dimension boundaries.
- EXIF orientation transposition.
- In-memory stream reuse (to_pil, to_rgb_pil).
- Path traversal and filename sanitization.
- Memory cleanup.
"""

import io
import pytest
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ImageValidationError
from app.services.image_service import ImageService, ProcessedImage


# ==============================================================================
# Helper Fixtures
# ==============================================================================

@pytest.fixture
def make_image_bytes():
    """Factory fixture to create in-memory image bytes of any format and size."""
    def _create(fmt="JPEG", size=(100, 80), color="blue", mode="RGB", exif_orientation=None):
        buf = io.BytesIO()
        img = Image.new(mode, size, color=color)
        save_kwargs = {"format": fmt}
        if exif_orientation is not None:
            exif = img.getexif()
            exif[0x0112] = exif_orientation  # 0x0112 is EXIF Orientation tag
            save_kwargs["exif"] = exif
        img.save(buf, **save_kwargs)
        return buf.getvalue()
    return _create


# ==============================================================================
# 1. Valid Formats & Metadata Extraction
# ==============================================================================

def test_process_valid_jpeg(make_image_bytes):
    """Verify JPEG validation and metadata extraction."""
    data = make_image_bytes(fmt="JPEG", size=(120, 90))
    processed = ImageService.process_image(data, "garment.jpg", "image/jpeg")

    assert processed.metadata.format == "JPEG"
    assert processed.metadata.width == 120
    assert processed.metadata.height == 90
    assert processed.metadata.size_bytes == len(data)
    assert processed.metadata.mime_type == "image/jpeg"
    assert processed.metadata.filename == "garment.jpg"


def test_process_valid_png(make_image_bytes):
    """Verify PNG validation and metadata extraction."""
    data = make_image_bytes(fmt="PNG", size=(64, 64))
    processed = ImageService.process_image(data, "label.png", "image/png")

    assert processed.metadata.format == "PNG"
    assert processed.metadata.width == 64
    assert processed.metadata.height == 64
    assert processed.metadata.mime_type == "image/png"


def test_process_valid_webp(make_image_bytes):
    """Verify WEBP validation and metadata extraction."""
    data = make_image_bytes(fmt="WEBP", size=(80, 80))
    processed = ImageService.process_image(data, "fabric.webp", "image/webp")

    assert processed.metadata.format == "WEBP"
    assert processed.metadata.width == 80
    assert processed.metadata.height == 80
    assert processed.metadata.mime_type == "image/webp"


# ==============================================================================
# 2. Security: Path Traversal & Filename Sanitization
# ==============================================================================

def test_filename_sanitization_path_traversal():
    """Verify path traversal patterns (../) are stripped."""
    sanitized = ImageService.sanitize_filename("../../../etc/passwd_garment.jpg")
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert sanitized == "passwd_garment.jpg"


def test_filename_sanitization_special_characters():
    """Verify unsafe shell/script characters are replaced with underscores."""
    sanitized = ImageService.sanitize_filename("shirt<script>alert(1).jpg")
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert "(" not in sanitized
    assert ")" not in sanitized
    assert sanitized.endswith(".jpg")


def test_filename_sanitization_empty_or_none():
    """Verify empty filename raises ImageValidationError."""
    with pytest.raises(ImageValidationError) as exc:
        ImageService.sanitize_filename("")
    assert exc.value.status_code == 400

    with pytest.raises(ImageValidationError):
        ImageService.sanitize_filename("   ")


# ==============================================================================
# 3. Security: Magic Bytes & MIME Consistency
# ==============================================================================

def test_magic_bytes_spoofed_extension():
    """Verify a text file renamed to .jpg is rejected."""
    fake_data = b"This is plain text disguised as an image file."
    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(fake_data, "fake.jpg", "image/jpeg")
    assert exc.value.status_code == 400


def test_mime_type_mismatch(make_image_bytes):
    """Verify mismatch between declared content-type and real format is caught."""
    jpeg_data = make_image_bytes(fmt="JPEG")
    # Declaring image/png for actual JPEG bytes
    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(jpeg_data, "garment.jpg", content_type="image/png")
    assert exc.value.status_code == 415
    assert "MIME type mismatch" in exc.value.message


def test_unsupported_image_format():
    """Verify non-allowed image formats (like GIF or BMP) are rejected."""
    buf = io.BytesIO()
    img = Image.new("RGB", (20, 20), color="red")
    img.save(buf, format="GIF")
    gif_data = buf.getvalue()

    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(gif_data, "animated.gif", "image/gif")
    assert exc.value.status_code == 415


# ==============================================================================
# 4. Error Handling: Empty, Corrupted, and Oversized Files
# ==============================================================================

def test_empty_image_data():
    """Verify empty bytes are rejected."""
    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(b"", "empty.jpg")
    assert exc.value.status_code == 400
    assert "empty" in exc.value.message.lower()


def test_corrupted_image_bytes():
    """Verify corrupted image with valid header but truncated payload is caught."""
    corrupted_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # Incomplete JPEG header
    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(corrupted_data, "broken.jpg")
    assert exc.value.status_code == 400
    assert "corrupted" in exc.value.message.lower()


def test_oversized_image_upload(monkeypatch, make_image_bytes):
    """Verify file exceeding MAX_UPLOAD_SIZE_BYTES is rejected with HTTP 413."""
    # Temporarily set limit to 1000 bytes for testing
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 500)
    data = make_image_bytes(fmt="JPEG", size=(100, 100))
    assert len(data) > 500

    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(data, "large.jpg")
    assert exc.value.status_code == 413
    assert "exceed 10 MB" in exc.value.message or "10" in exc.value.message


# ==============================================================================
# 5. Security: Decompression Bomb & Dimension Boundaries
# ==============================================================================

def test_decompression_bomb_protection(monkeypatch, make_image_bytes):
    """Verify image exceeding MAX_IMAGE_PIXELS raises bomb protection error."""
    # Set max pixels to 500
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 500)
    data = make_image_bytes(fmt="PNG", size=(50, 50))  # 2500 pixels > 500

    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(data, "bomb.png")
    assert exc.value.status_code == 400
    assert "decompression bomb" in exc.value.message.lower()


def test_excessive_dimension_limit(monkeypatch, make_image_bytes):
    """Verify single dimension exceeding MAX_IMAGE_DIMENSION is rejected."""
    monkeypatch.setattr(settings, "MAX_IMAGE_DIMENSION", 100)
    data = make_image_bytes(fmt="JPEG", size=(150, 20))  # width 150 > 100

    with pytest.raises(ImageValidationError) as exc:
        ImageService.process_image(data, "wide.jpg")
    assert exc.value.status_code == 400
    assert "exceed maximum allowed dimension" in exc.value.message


# ==============================================================================
# 6. Processing: EXIF Orientation & In-Memory Reuse
# ==============================================================================

def test_exif_orientation_transposition(make_image_bytes):
    """Verify that EXIF orientation tag 6 (90° CW rotation) transposes dimensions properly."""
    # Create image with width 100, height 50 and EXIF orientation 6
    data = make_image_bytes(fmt="JPEG", size=(100, 50), exif_orientation=6)
    processed = ImageService.process_image(data, "rotated.jpg")

    # to_pil should apply exif_transpose, swapping width and height to 50x100
    pil_img = processed.to_pil()
    assert pil_img.width == 50
    assert pil_img.height == 100


def test_in_memory_stream_reuse(make_image_bytes):
    """Verify that multiple services can access clean streams and PIL instances repeatedly."""
    data = make_image_bytes(fmt="PNG", size=(80, 80))
    processed = ImageService.process_image(data, "multi_use.png")

    # First stream usage (e.g., OCR)
    stream1 = processed.get_stream()
    assert len(stream1.read()) == len(data)

    # Second stream usage (e.g., Computer Vision)
    stream2 = processed.get_stream()
    assert len(stream2.read()) == len(data)

    # PIL instances can be generated independently
    img1 = processed.to_pil()
    img2 = processed.to_pil()
    assert img1.size == (80, 80)
    assert img2.size == (80, 80)


def test_to_rgb_conversion(make_image_bytes):
    """Verify that RGBA images are normalized to RGB for CV models."""
    data = make_image_bytes(fmt="PNG", size=(40, 40), mode="RGBA", color=(255, 0, 0, 128))
    processed = ImageService.process_image(data, "alpha.png")

    rgb_img = processed.to_rgb_pil()
    assert rgb_img.mode == "RGB"


def test_processed_image_cleanup(make_image_bytes):
    """Verify memory cleanup clears internal byte buffer."""
    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "clean.jpg")
    assert len(processed.raw_bytes) > 0

    processed.cleanup()
    assert len(processed.raw_bytes) == 0


def test_multiple_sequential_image_processing(make_image_bytes):
    """Verify handling multiple uploads concurrently or in sequence without state leakage."""
    images = [
        ImageService.process_image(make_image_bytes(fmt="JPEG", size=(50, 50)), f"img_{i}.jpg")
        for i in range(5)
    ]
    assert len(images) == 5
    for i, img in enumerate(images):
        assert img.metadata.filename == f"img_{i}.jpg"
        assert img.metadata.width == 50
