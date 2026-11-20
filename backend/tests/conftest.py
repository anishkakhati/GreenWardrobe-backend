"""Pytest fixtures and configuration for GreenWardrobe tests."""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    buffer = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate in-memory valid PNG image bytes."""
    buffer = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Generate arbitrary invalid bytes simulating corrupt file."""
    return b"NOT_A_REAL_IMAGE_HEADER_DATA_123456"


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
