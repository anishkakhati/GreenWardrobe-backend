"""UNKNOWN classification must stop before OCR and computer vision run."""

import io
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.normalization_service import normalization_service


def test_blank_unknown_image_returns_422_without_running_analysis_pipeline(monkeypatch):
    image_buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color="white").save(image_buffer, format="JPEG")
    pipeline = AsyncMock()
    monkeypatch.setattr(normalization_service, "execute_pipeline", pipeline)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"image": ("blank.jpg", image_buffer.getvalue(), "image/jpeg")},
        )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["details"]["image_type"] == "UNKNOWN"
    assert error["details"]["confidence"] < error["details"]["threshold"]
    pipeline.assert_not_awaited()
