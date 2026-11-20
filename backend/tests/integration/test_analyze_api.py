"""Integration tests for POST /api/v1/analyze endpoint and full Phase 6 orchestration.

All tests use real Phase 5 adapters where assets are present and mocked Gemini clients.
NO live Gemini API calls are made.
NO real secrets are used.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.response import AiExplanation, AnalysisResponse
from app.services.gemini_service import gemini_service


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_label_bytes():
    """Load real label fixture label_0002.jpg (68% Nylon, 32% Wool)."""
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0002.jpg"
    if fixture_path.exists():
        with open(fixture_path, "rb") as f:
            return f.read()
    # Fallback in-memory image
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color="white").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_gemini_explanation():
    return AiExplanation(
        status="AVAILABLE",
        summary="This garment is composed of 68% Nylon and 32% Wool.",
        material_explanation="Nylon provides weather resistance; wool delivers thermal insulation.",
        environmental_impact_explanation="Composite score is 68.3 with high microfiber shedding concerns.",
        impact_category_explanations={
            "water": "Moderate water demand.",
            "microfibres": "Critical synthetic shedding risk.",
        },
        environmental_highlights=["68% Nylon sheds microplastics", "32% Wool requires pasture land"],
        practical_recommendations=["Consider durable second-hand alternatives"],
        care_and_repair_tips=["Wash cold with a microfibre filter", "Dry flat away from direct heat"],
        limitations_and_uncertainty=["Baseline estimates based on industry benchmarks"],
        data_quality_disclaimer="Comparative literature baseline.",
        is_fallback=False,
    )


def test_analyze_api_valid_multipart_request(client, sample_label_bytes, mock_gemini_explanation):
    """Verify POST /api/v1/analyze with valid image returns 200 and AnalysisResponse structure."""
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        files = {
            "image": ("label_0002.jpg", sample_label_bytes, "image/jpeg"),
        }
        data = {
            "shopping": "true",
            "second_hand_available": "false",
            "condition": "new",
        }
        response = client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()

    # Validate against AnalysisResponse schema
    validated = AnalysisResponse(**res_json)
    assert validated.success is True
    assert validated.analysis_id is not None
    assert validated.product.composition is not None
    assert validated.impact_assessment.overall_impact_score > 0
    assert validated.ai_explanation is not None
    assert validated.ai_explanation.status == "AVAILABLE"
    assert validated.metadata.gemini_model == "gemini-3.5-flash-lite"
    assert validated.metadata.explanation_status == "AVAILABLE"


def test_analyze_api_missing_image(client):
    """Verify POST /api/v1/analyze without required image returns 422 validation error."""
    response = client.post("/api/v1/analyze", data={"shopping": "true"})
    assert response.status_code == 422


def test_analyze_api_invalid_image_format(client):
    """Verify POST /api/v1/analyze with non-image file extension/MIME returns 400 error."""
    files = {
        "image": ("label.txt", b"plain text is not an image", "text/plain"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code in [400, 422]
    res_json = response.json()
    err_text = str(res_json).lower()
    assert "format" in err_text or "unsupported" in err_text or "image" in err_text


def test_analyze_api_corrupt_image_bytes(client, corrupt_image_bytes):
    """Verify POST /api/v1/analyze with corrupt image bytes returns 400 error."""
    files = {
        "image": ("label.jpg", corrupt_image_bytes, "image/jpeg"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 400
    res_json = response.json()
    err_text = str(res_json).lower()
    assert "corrupted" in err_text or "invalid" in err_text or "image" in err_text


def test_analyze_api_oversized_image(client):
    """Verify POST /api/v1/analyze with oversized image (>10MB) returns 413 Payload Too Large."""
    # 11 MB of zero bytes
    large_bytes = b"0" * (11 * 1024 * 1024)
    files = {
        "image": ("garment_large.jpg", large_bytes, "image/jpeg"),
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 413


def test_analyze_api_optional_parameters_condition(client, sample_label_bytes, mock_gemini_explanation):
    """Verify optional parameters (e.g. condition='damaged', shopping=False) affect recommendations."""
    with patch.object(gemini_service.adapter, "_sync_generate", return_value=mock_gemini_explanation):
        files = {
            "image": ("label.jpg", sample_label_bytes, "image/jpeg"),
        }
        data = {
            "shopping": "false",
            "second_hand_available": "false",
            "condition": "damaged",
        }
        response = client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()
    # Damaged garment for existing wardrobe item triggers 'Repair' recommendation
    assert res_json["impact_assessment"]["recommendation"] in ["Repair", "Think Twice"]


def test_orchestration_phase5_succeeds_gemini_unconfigured(client, sample_label_bytes):
    """Verify that when GEMINI_API_KEY is empty, the pipeline succeeds with factual deterministic fallback."""
    # Ensure unconfigured adapter
    with patch.object(gemini_service.adapter, "_api_key", ""):
        files = {
            "image": ("label.jpg", sample_label_bytes, "image/jpeg"),
        }
        response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    # Phase 5 results remain fully populated
    assert res_json["product"]["composition"] is not None
    assert res_json["impact_assessment"]["overall_impact_score"] > 0
    # Gemini layer provides deterministic fallback
    assert res_json["ai_explanation"]["is_fallback"] is True
    assert res_json["ai_explanation"]["status"] == "UNAVAILABLE"
    assert "Nylon" in res_json["ai_explanation"]["summary"]
    assert res_json["metadata"]["explanation_status"] == "UNAVAILABLE"


def test_orchestration_scores_and_materials_unaltered_by_gemini(client, sample_label_bytes):
    """Verify Gemini explanation layer cannot alter original Phase 5 composition, scores, or recommendation."""
    # Injected rogue LLM response attempting to change fiber composition and score
    rogue_llm_json = AiExplanation(
        status="AVAILABLE",
        summary="I think this is 100% cashmere with a score of 10/100!",  # Contradicts reality
        environmental_highlights=["Fictional low impact"],
        practical_recommendations=["Buy now!"],
        care_and_repair_tips=["Dry clean only"],
        limitations_and_uncertainty=["Rogue model hallucination"],
        is_fallback=False,
    )

    with patch.object(gemini_service.adapter, "_sync_generate", return_value=rogue_llm_json):
        files = {
            "image": ("label.jpg", sample_label_bytes, "image/jpeg"),
        }
        response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res_json = response.json()

    # The actual composition in product must STILL be nylon/wool, NOT cashmere!
    materials = {m["name"] for m in res_json["product"]["composition"]["materials"]}
    assert "nylon" in materials
    assert "cashmere" not in materials

    # The actual numerical score must remain ~68.3, NOT 10!
    assert res_json["impact_assessment"]["overall_impact_score"] > 60.0


def test_orchestration_cors_headers_on_analyze(client):
    """Verify CORS preflight headers on /api/v1/analyze allow the React frontend."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/analyze", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


@pytest.mark.anyio
async def test_guaranteed_image_cleanup_when_garment_validation_fails():
    """TASK 1 & 7: Verify ProcessedImage cleanup is guaranteed when label succeeds but garment validation fails."""
    from app.core.exceptions import ImageValidationError
    from app.schemas.request import ImageMetadata
    from app.services.analysis_service import analysis_service
    from app.services.image_service import ProcessedImage, image_service

    # Create dummy upload files
    label_file = MagicMock()
    garment_file = MagicMock()

    # Create mock ProcessedImage for label with tracked cleanup
    mock_label_proc = ProcessedImage(
        raw_bytes=b"fake_label_bytes",
        metadata=ImageMetadata(filename="label.jpg", format="JPEG", width=100, height=100, size_bytes=100),
    )
    mock_label_proc.cleanup = MagicMock()

    async def mock_validate(file):
        if file == label_file:
            return mock_label_proc
        raise ImageValidationError("Corrupt garment image format", status_code=400)

    with patch.object(image_service, "validate_and_process_upload", side_effect=mock_validate):
        with pytest.raises(ImageValidationError):
            await analysis_service.analyze_garment(label_file, garment_file)

    # Label cleanup MUST have been called even though garment validation failed!
    assert mock_label_proc.cleanup.call_count == 1


@pytest.mark.anyio
async def test_guaranteed_image_cleanup_when_pipeline_raises_exception():
    """TASK 1 & 7: Verify ProcessedImage cleanup is guaranteed when Phase 5 pipeline execution raises an exception."""
    from app.schemas.request import ImageMetadata
    from app.services.analysis_service import analysis_service
    from app.services.image_service import ProcessedImage, image_service
    from app.services.normalization_service import normalization_service

    label_file = MagicMock()
    garment_file = MagicMock()

    mock_label_proc = ProcessedImage(
        raw_bytes=b"label_bytes",
        metadata=ImageMetadata(filename="label.jpg", format="JPEG", width=100, height=100, size_bytes=100),
    )
    mock_label_proc.cleanup = MagicMock()

    mock_garment_proc = ProcessedImage(
        raw_bytes=b"garment_bytes",
        metadata=ImageMetadata(filename="garment.jpg", format="JPEG", width=100, height=100, size_bytes=100),
    )
    mock_garment_proc.cleanup = MagicMock()

    async def mock_validate(file):
        if file == label_file:
            return mock_label_proc
        return mock_garment_proc

    with patch.object(image_service, "validate_and_process_upload", side_effect=mock_validate):
        with patch.object(normalization_service, "execute_pipeline", side_effect=RuntimeError("Simulated pipeline crash")):
            with pytest.raises(RuntimeError):
                await analysis_service.analyze_garment(label_file, garment_file)

    # Both images MUST have been cleaned up!
    assert mock_label_proc.cleanup.call_count == 1
    assert mock_garment_proc.cleanup.call_count == 1


def test_gemini_service_exception_does_not_destroy_phase5_result(client, sample_label_bytes):
    """TASK 5 & 7: Verify that an unexpected Gemini exception never destroys Phase 5 analytical outputs."""
    with patch.object(gemini_service, "explain_analysis", side_effect=RuntimeError("Unrecoverable Gemini crash")):
        files = {
            "image": ("label.jpg", sample_label_bytes, "image/jpeg"),
        }
        response = client.post("/api/v1/analyze", files=files)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    # Phase 5 analytical result is intact
    assert res_json["product"]["composition"] is not None
    assert res_json["impact_assessment"]["overall_impact_score"] > 0
    # Gemini layer provided safe fallback
    assert res_json["ai_explanation"]["status"] == "UNAVAILABLE"
    assert res_json["ai_explanation"]["is_fallback"] is True
    assert res_json["metadata"]["explanation_status"] == "UNAVAILABLE"
