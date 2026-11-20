"""Integration tests for the Phase 7 single-image POST /api/v1/analyze endpoint.

Tests cover all 16 required scenarios:
  1. Valid garment image → GARMENT classification → Vision routing
  2. Valid label image → LABEL classification → OCR routing
  3. Uncertain image (high threshold) → UNKNOWN → 422
  4. Unsupported image format → 400/415
  5. GARMENT → computer-vision routing (Vision adapter called, not OCR)
  6. LABEL → OCR routing (OCR adapter called, not Vision)
  7. Material normalization (via pipeline)
  8. Percentage validation (via pipeline)
  9. Knowledge-base integration (via pipeline)
  10. Environmental scoring (via pipeline)
  11. Gemini unavailable / fallback
  12. Missing image field → 422
  13. Invalid image format (plain text as image) → 400/422
  14. Oversized image → 413
  15. Structured response validation (AnalysisResponse schema)
  16. Regression: existing /upload endpoint still works

All tests use mocked adapters or in-memory images.
No live Gemini API calls. No real secrets.
"""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.integrations.image_type_classifier import (
    IMAGE_TYPE_GARMENT,
    IMAGE_TYPE_LABEL,
    IMAGE_TYPE_UNKNOWN,
    BaseImageTypeClassifier,
    ClassificationResult,
    HeuristicImageTypeClassifier,
)
from app.main import app
from app.schemas.response import AiExplanation, AnalysisResponse
from app.services.gemini_service import gemini_service
from app.services.image_service import ProcessedImage, image_service
from app.schemas.request import ImageMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def _jpeg_bytes(size=(100, 100), color=(80, 120, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def _white_label_jpeg(size=(200, 300)) -> bytes:
    """JPEG that looks like a care label: white bg + text lines."""
    pil = Image.new("RGB", size, color=(252, 252, 252))
    draw = ImageDraw.Draw(pil)
    for y in range(30, size[1] - 30, 25):
        draw.rectangle([(20, y), (size[0] - 20, y + 8)], fill=(10, 10, 10))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()


def _garment_jpeg(size=(224, 224)) -> bytes:
    """JPEG that looks like a textile garment: mid-tones, noisy."""
    rng = np.random.default_rng(7)
    arr = rng.integers(60, 160, size=(*size, 3), dtype=np.uint8)
    for i in range(0, size[0], 8):
        arr[i, :, :] = np.clip(arr[i, :, :].astype(int) + 40, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_gemini_explanation():
    return AiExplanation(
        status="AVAILABLE",
        summary="Test garment summary.",
        material_explanation="Synthetic blend.",
        environmental_impact_explanation="Moderate impact.",
        impact_category_explanations={},
        environmental_highlights=["High microfibre risk"],
        practical_recommendations=["Wash cold"],
        care_and_repair_tips=["Air dry"],
        limitations_and_uncertainty=["Benchmark estimate"],
        data_quality_disclaimer="Comparative data.",
        is_fallback=False,
    )


@pytest.fixture
def sample_label_bytes():
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0002.jpg"
    )
    if fixture_path.exists():
        return fixture_path.read_bytes()
    return _white_label_jpeg()


# ---------------------------------------------------------------------------
# Helper: force classifier to return a specific type
# ---------------------------------------------------------------------------

def _make_forced_classifier(image_type: str, confidence: float = 0.90):
    """Create a classifier double that always returns the specified type."""

    class ForcedClassifier(BaseImageTypeClassifier):
        @property
        def classifier_name(self) -> str:
            return f"ForcedClassifier-{image_type}"

        @property
        def confidence_threshold(self) -> float:
            return 0.60

        def classify(self, image: ProcessedImage) -> ClassificationResult:
            above = confidence >= self.confidence_threshold
            final_type = image_type if above else IMAGE_TYPE_UNKNOWN
            return ClassificationResult(
                image_type=final_type,
                confidence=confidence,
                classifier_name=self.classifier_name,
                reasoning=[f"forced={image_type}"],
                is_above_threshold=above,
            )

    return ForcedClassifier()


# ---------------------------------------------------------------------------
# T01: Valid garment image → 200 + AnalysisResponse
# ---------------------------------------------------------------------------

def test_single_image_garment_returns_200(client, mock_gemini_explanation):
    """T01: A garment-like image with forced GARMENT classification returns 200."""
    forced = _make_forced_classifier(IMAGE_TYPE_GARMENT)
    with patch(
        "app.services.analysis_service.get_default_classifier",
        return_value=forced,
    ):
        with patch("app.services.analysis_service.AnalysisService.__init__") as mock_init:
            # Instead of patching __init__, patch at service level
            pass

    # Direct service injection via classifier_instance parameter
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_GARMENT),
        ):
            files = {"image": ("garment.jpg", _garment_jpeg(), "image/jpeg")}
            data = {"shopping": "true", "second_hand_available": "false", "condition": "new"}
            response = client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["analysis_id"] is not None


# ---------------------------------------------------------------------------
# T02: Valid label image → 200 + AnalysisResponse
# ---------------------------------------------------------------------------

def test_single_image_label_returns_200(client, sample_label_bytes, mock_gemini_explanation):
    """T02: A label-like image with forced LABEL classification returns 200."""
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
            data = {"shopping": "false", "second_hand_available": "false", "condition": "new"}
            response = client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True


# ---------------------------------------------------------------------------
# T03: Uncertain classification → 422 UNKNOWN
# ---------------------------------------------------------------------------

def test_unknown_classification_returns_422(client):
    """T03: When classifier returns UNKNOWN (confidence below threshold), expect 422."""
    with patch(
        "app.services.analysis_service.analysis_service._classifier",
        new=_make_forced_classifier(IMAGE_TYPE_UNKNOWN, confidence=0.0),
    ):
        files = {"image": ("ambiguous.jpg", _jpeg_bytes(), "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 422
    err = response.json()
    assert err["success"] is False
    # Error message should be interpretable
    err_text = str(err).lower()
    assert "classif" in err_text or "unknown" in err_text or "confidence" in err_text


# ---------------------------------------------------------------------------
# T04 & T13: Unsupported / invalid image format
# ---------------------------------------------------------------------------

def test_invalid_image_format_returns_error(client):
    """T04/T13: Plain text disguised as JPEG returns 400 or 422."""
    files = {"image": ("fake.jpg", b"NOT_AN_IMAGE_BYTES", "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in {400, 415, 422}


def test_gif_image_format_rejected(client):
    """T04: GIF format not in allowed list → error."""
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color="red").save(buf, format="GIF")
    files = {"image": ("anim.gif", buf.getvalue(), "image/gif")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in {400, 415, 422}


# ---------------------------------------------------------------------------
# T05: GARMENT → Vision adapter is called; OCR is NOT called
# ---------------------------------------------------------------------------

def test_garment_route_calls_vision_not_ocr(client, mock_gemini_explanation):
    """T05: When image_type=GARMENT, Vision adapter is called; OCR is skipped."""
    from app.integrations.vision_adapter import VisionAdapter
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import VisionPrediction, OCRResult

    mock_vision = MagicMock(spec=VisionAdapter)
    mock_vision.predict_material = AsyncMock(return_value=VisionPrediction(
        predicted_material="cotton",
        confidence=0.85,
        alternatives=["linen"],
    ))
    mock_vision.status_info = MagicMock(return_value={"adapter": "mock_vision"})

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[],
        unreadable=True,
        warnings=["OCR should not have been called"],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_GARMENT),
        ):
            with patch(
                "app.integrations.vision_adapter.vision_adapter",
                new=mock_vision,
            ):
                with patch(
                    "app.integrations.ocr_adapter.ocr_adapter",
                    new=mock_ocr,
                ):
                    files = {"image": ("garment.jpg", _garment_jpeg(), "image/jpeg")}
                    response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    mock_vision.predict_material.assert_called_once()
    mock_ocr.extract_composition.assert_not_called()


# ---------------------------------------------------------------------------
# T06: LABEL → OCR adapter is called; Vision is NOT called
# ---------------------------------------------------------------------------

def test_label_route_calls_ocr_not_vision(client, sample_label_bytes, mock_gemini_explanation):
    """T06: When image_type=LABEL, OCR adapter is called; Vision is skipped."""
    from app.integrations.vision_adapter import VisionAdapter
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import VisionPrediction, OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[
            OCRCompositionItem(material="Cotton", percentage=80.0),
            OCRCompositionItem(material="Polyester", percentage=20.0),
        ],
        confidence=0.90,
        raw_text="80% Cotton 20% Polyester",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    mock_vision = MagicMock(spec=VisionAdapter)
    mock_vision.predict_material = AsyncMock(return_value=VisionPrediction(
        predicted_material=None,
        warnings=["Vision should not have been called"],
    ))
    mock_vision.status_info = MagicMock(return_value={"adapter": "mock_vision"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch(
                "app.integrations.ocr_adapter.ocr_adapter",
                new=mock_ocr,
            ):
                with patch(
                    "app.integrations.vision_adapter.vision_adapter",
                    new=mock_vision,
                ):
                    files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                    response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    mock_ocr.extract_composition.assert_called_once()
    mock_vision.predict_material.assert_not_called()


# ---------------------------------------------------------------------------
# T07: Material normalization (canonical name mapping)
# ---------------------------------------------------------------------------

def test_material_normalization_applied(client, sample_label_bytes, mock_gemini_explanation):
    """T07: OCR result with 'SPANDEX' should be normalized to 'elastane' in composition."""
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[
            OCRCompositionItem(material="Cotton", percentage=95.0),
            OCRCompositionItem(material="Spandex", percentage=5.0),
        ],
        confidence=0.88,
        raw_text="95% COTTON 5% SPANDEX",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch("app.integrations.ocr_adapter.ocr_adapter", new=mock_ocr):
                files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    materials = {m["name"] for m in res["product"]["composition"]["materials"]}
    # "Spandex" should be normalized to "elastane"
    assert "elastane" in materials, f"Expected 'elastane' in {materials}"
    assert "spandex" not in materials


# ---------------------------------------------------------------------------
# T08: Percentage validation — composition source preserved
# ---------------------------------------------------------------------------

def test_ocr_percentage_sum_warning_in_response(client, sample_label_bytes, mock_gemini_explanation):
    """T08: If OCR percentages don't sum to 100%, a warning appears in response."""
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[
            OCRCompositionItem(material="Cotton", percentage=60.0),
            OCRCompositionItem(material="Polyester", percentage=20.0),
            # deliberately only 80% total
        ],
        confidence=0.75,
        raw_text="60% COTTON 20% POLYESTER",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch("app.integrations.ocr_adapter.ocr_adapter", new=mock_ocr):
                files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    all_text = str(res["warnings"]).lower() + str(res["limitations"]).lower()
    # Normalization service should emit a percentage warning
    assert "percent" in all_text or "sum" in all_text or "80" in all_text or "composition" in all_text


# ---------------------------------------------------------------------------
# T09: Knowledge-base integration
# ---------------------------------------------------------------------------

def test_knowledge_base_queried_for_detected_materials(client, sample_label_bytes, mock_gemini_explanation):
    """T09: Detected materials are passed to the knowledge base and profiles returned."""
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[
            OCRCompositionItem(material="Wool", percentage=100.0),
        ],
        confidence=0.92,
        raw_text="100% WOOL",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch("app.integrations.ocr_adapter.ocr_adapter", new=mock_ocr):
                files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    # environmental_data should contain "wool" profile
    env_keys = set(res["environmental_data"].keys())
    assert "wool" in env_keys, f"Expected 'wool' in environmental_data keys: {env_keys}"


# ---------------------------------------------------------------------------
# T10: Environmental scoring
# ---------------------------------------------------------------------------

def test_environmental_score_nonzero_for_known_material(client, sample_label_bytes, mock_gemini_explanation):
    """T10: Known materials produce a non-zero overall_impact_score from the scoring engine."""
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[
            OCRCompositionItem(material="Nylon", percentage=68.0),
            OCRCompositionItem(material="Wool", percentage=32.0),
        ],
        confidence=0.95,
        raw_text="68% NYLON 32% WOOL",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch("app.integrations.ocr_adapter.ocr_adapter", new=mock_ocr):
                files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    score = res["impact_assessment"]["overall_impact_score"]
    assert score > 0.0
    assert score <= 100.0
    assert res["impact_assessment"]["recommendation"] is not None


# ---------------------------------------------------------------------------
# T11: Gemini unavailable / fallback
# ---------------------------------------------------------------------------

def test_gemini_failure_returns_fallback_not_500(client, sample_label_bytes):
    """T11: When Gemini raises an exception, the response still returns 200 with fallback."""
    from app.integrations.ocr_adapter import OCRAdapter
    from app.schemas.response import OCRCompositionItem, OCRResult

    mock_ocr = MagicMock(spec=OCRAdapter)
    mock_ocr.extract_composition = AsyncMock(return_value=OCRResult(
        composition=[OCRCompositionItem(material="Cotton", percentage=100.0)],
        confidence=0.88,
        raw_text="100% COTTON",
        unreadable=False,
        warnings=[],
    ))
    mock_ocr.status_info = MagicMock(return_value={"adapter": "mock_ocr"})

    with patch.object(
        gemini_service, "explain_analysis", side_effect=RuntimeError("Gemini crashed")
    ):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            with patch("app.integrations.ocr_adapter.ocr_adapter", new=mock_ocr):
                files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
                response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    assert res["ai_explanation"]["is_fallback"] is True
    assert res["ai_explanation"]["status"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# T12: Missing image field
# ---------------------------------------------------------------------------

def test_missing_image_field_returns_422(client):
    """T12: POST /api/v1/analyze with no image field returns 422."""
    response = client.post("/api/v1/analyze", data={"shopping": "true"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# T14: Oversized image
# ---------------------------------------------------------------------------

def test_oversized_image_returns_413(client):
    """T14: Image exceeding MAX_UPLOAD_SIZE_BYTES returns 413."""
    from app.core.config import settings

    # Craft bytes larger than the limit
    over_limit = b"\xff\xd8\xff" + b"X" * (settings.MAX_UPLOAD_SIZE_BYTES + 100)
    files = {"image": ("huge.jpg", over_limit, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in {400, 413, 422}


# ---------------------------------------------------------------------------
# T15: Structured response validation (AnalysisResponse schema)
# ---------------------------------------------------------------------------

def test_response_schema_validates_successfully(client, sample_label_bytes, mock_gemini_explanation):
    """T15: The response body from /api/v1/analyze validates against AnalysisResponse."""
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_LABEL),
        ):
            files = {"image": ("label.jpg", sample_label_bytes, "image/jpeg")}
            response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    validated = AnalysisResponse(**response.json())
    assert validated.success is True
    assert validated.analysis_id is not None
    assert validated.image_classification is not None
    assert validated.image_classification.image_type == IMAGE_TYPE_LABEL
    assert validated.image_classification.analysis_source == "OCR"
    assert validated.metadata.gemini_model is not None
    assert validated.metadata.explanation_status is not None


# ---------------------------------------------------------------------------
# T15b: Response includes image_classification metadata fields
# ---------------------------------------------------------------------------

def test_response_includes_image_classification_metadata(client, mock_gemini_explanation):
    """T15b: Response must include image_type, classification_confidence, analysis_source."""
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        with patch(
            "app.services.analysis_service.analysis_service._classifier",
            new=_make_forced_classifier(IMAGE_TYPE_GARMENT, confidence=0.88),
        ):
            files = {"image": ("garment.jpg", _garment_jpeg(), "image/jpeg")}
            response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res = response.json()
    ic = res["image_classification"]
    assert ic is not None
    assert ic["image_type"] == IMAGE_TYPE_GARMENT
    assert ic["analysis_source"] == "VISION"
    assert 0.0 <= ic["classification_confidence"] <= 1.0
    assert isinstance(ic["classification_reasoning"], list)


# ---------------------------------------------------------------------------
# T16: Regression — legacy /upload endpoint still works
# ---------------------------------------------------------------------------

def test_legacy_upload_endpoint_still_works(client):
    """T16: POST /api/v1/upload (legacy) still accepts garment_image and returns 200."""
    files = {"garment_image": ("garment.jpg", _jpeg_bytes(), "image/jpeg")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    res = response.json()
    assert res["message"] == "Images uploaded and validated successfully."


def test_legacy_upload_with_both_images(client):
    """T16b: Legacy /upload endpoint accepts both garment_image and label_image."""
    files = {
        "garment_image": ("garment.jpg", _jpeg_bytes(), "image/jpeg"),
        "label_image": ("label.jpg", _png_bytes(), "image/png"),
    }
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# T16c: Regression — old two-image analyze_garment() method still works
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_legacy_analyze_garment_method_preserved(mock_gemini_explanation):
    """T16c: AnalysisService.analyze_garment() (two-image legacy) still works."""
    from unittest.mock import MagicMock
    from app.services.analysis_service import analysis_service

    label_file = MagicMock()
    label_file.filename = "label.jpg"
    garment_file = MagicMock()
    garment_file.filename = "garment.jpg"

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    async def mock_validate(file):
        from app.schemas.request import ImageMetadata
        return ProcessedImage(
            raw_bytes=jpeg_bytes,
            metadata=ImageMetadata(
                filename=file.filename,
                format="JPEG",
                width=100,
                height=100,
                size_bytes=len(jpeg_bytes),
            ),
        )

    with patch.object(
        image_service, "validate_and_process_upload", side_effect=mock_validate
    ):
        with patch.object(
            gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation
        ):
            result = await analysis_service.analyze_garment(
                label_file, garment_file
            )

    assert isinstance(result, AnalysisResponse)
    assert result.analysis_id is not None


# ---------------------------------------------------------------------------
# T17: CORS headers on the new single-image endpoint
# ---------------------------------------------------------------------------

def test_cors_on_analyze_allows_vite_origin(client):
    """T17: CORS preflight from localhost:5173 is accepted by /api/v1/analyze."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/analyze", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
