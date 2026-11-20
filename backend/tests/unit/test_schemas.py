"""Comprehensive unit tests for GreenWardrobe Pydantic v2 schemas.

Tests validation rules, edge cases, boundaries, multi-material blends,
and JSON serialization.
"""

import json
import pytest
from pydantic import ValidationError

from app.schemas.common import (
    RecommendationTier,
    ImpactLevel,
    AnalysisSource,
    ErrorDetail,
    ErrorResponse,
)
from app.schemas.request import (
    ImageMetadata,
    AnalysisOptions,
    DirectMaterialInput,
    DirectAnalysisRequest,
)
from app.schemas.response import (
    Material,
    Composition,
    VisionResult,
    OCRCompositionItem,
    OCRResult,
    VisionPrediction,
    MaterialEnvironmentalProfile,
    KnowledgeBaseResult,
    IndicatorScore,
    ImpactScoreBreakdown,
    ScoringResult,
    AiExplanation,
    ProductAnalysis,
    AnalysisMetadata,
    AnalysisResponse,
    GreenWardrobeResponse,
)


# ==============================================================================
# 1. Material & Composition Validation Tests
# ==============================================================================

def test_material_valid_percentage():
    """Verify material accepts valid percentages."""
    mat = Material(name="cotton", percentage=100.0)
    assert mat.name == "cotton"
    assert mat.percentage == 100.0

    mat_zero = Material(name="elastane", percentage=0.0)
    assert mat_zero.percentage == 0.0


def test_material_invalid_percentage_bounds():
    """Verify material rejects percentages below 0 or above 100."""
    with pytest.raises(ValidationError):
        Material(name="cotton", percentage=-5.0)

    with pytest.raises(ValidationError):
        Material(name="cotton", percentage=105.0)


def test_composition_multi_materials_and_completeness():
    """Verify multi-material composition calculation and completeness flag."""
    comp = Composition(
        materials=[
            Material(name="polyester", percentage=65.0),
            Material(name="cotton", percentage=35.0),
        ],
        confidence=0.95,
        source=AnalysisSource.OCR_LABEL,
    )
    assert len(comp.materials) == 2
    assert comp.total_percentage == 100.0
    assert comp.is_complete is True
    assert comp.confidence == 0.95
    assert comp.source == AnalysisSource.OCR_LABEL


def test_composition_incomplete_label():
    """Verify incomplete label (e.g., missing percentage) is accepted but flagged incomplete."""
    comp = Composition(
        materials=[Material(name="wool", percentage=70.0)],
        confidence=0.80,
    )
    assert comp.total_percentage == 70.0
    assert comp.is_complete is False


def test_composition_invalid_confidence():
    """Verify confidence must stay between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        Composition(materials=[], confidence=1.5)

    with pytest.raises(ValidationError):
        Composition(materials=[], confidence=-0.1)


# ==============================================================================
# 2. Request Schema Tests
# ==============================================================================

def test_image_metadata_validation():
    """Verify ImageMetadata validates dimensions and sizes."""
    meta = ImageMetadata(
        filename="shirt.jpg",
        format="JPEG",
        width=1920,
        height=1080,
        size_bytes=204800,
    )
    assert meta.filename == "shirt.jpg"
    assert meta.width == 1920

    with pytest.raises(ValidationError):
        ImageMetadata(filename="bad.jpg", format="JPEG", width=-10, height=100, size_bytes=100)


def test_direct_material_input_normalization():
    """Verify fiber names are trimmed and converted to lowercase."""
    item = DirectMaterialInput(material="  POLYESTER  ", percentage=60.0)
    assert item.material == "polyester"
    assert item.percentage == 60.0


def test_direct_analysis_request_valid_and_total_cap():
    """Verify DirectAnalysisRequest accepts valid total percentages and rejects > 100.5%."""
    valid_req = DirectAnalysisRequest(
        materials=[
            DirectMaterialInput(material="Cotton", percentage=80.0),
            DirectMaterialInput(material="Polyester", percentage=20.0),
        ],
        garment_type="t-shirt",
    )
    assert len(valid_req.materials) == 2

    # Exceeding total percentage
    with pytest.raises(ValidationError) as exc:
        DirectAnalysisRequest(
            materials=[
                DirectMaterialInput(material="Cotton", percentage=80.0),
                DirectMaterialInput(material="Polyester", percentage=30.0),
            ]
        )
    assert "exceeds 100%" in str(exc.value)


# ==============================================================================
# 3. Teammate Module Contract Schema Tests
# ==============================================================================

def test_ocr_result_schema():
    """Verify Harshit's OCR result model handles structured compositions."""
    ocr = OCRResult(
        composition=[
            OCRCompositionItem(material="Organic Cotton", percentage=95.0),
            OCRCompositionItem(material="Elastane", percentage=5.0),
        ],
        confidence=0.96,
        raw_text="95% ORGANIC COTTON 5% ELASTANE",
        unreadable=False,
    )
    assert len(ocr.composition) == 2
    assert ocr.confidence == 0.96
    assert ocr.unreadable is False


def test_vision_prediction_schema():
    """Verify Anshika Thakur's CV result model captures predictions and alternatives."""
    cv = VisionPrediction(
        predicted_material="Linen",
        confidence=0.89,
        alternatives=["Hemp", "Raw Cotton"],
        probabilities={"Linen": 0.89, "Hemp": 0.08, "Cotton": 0.03},
    )
    assert cv.predicted_material == "Linen"
    assert cv.confidence == 0.89
    assert len(cv.alternatives) == 2


def test_knowledge_base_profile_schema():
    """Verify Ankit's Knowledge Base models environmental metrics properly."""
    profile = MaterialEnvironmentalProfile(
        material="Polyester",
        water_liters_per_kg=25.0,
        co2_kg_per_kg=9.5,
        energy_mj_per_kg=125.0,
        microplastic_shedding_risk="Severe",
        biodegradability="Non-biodegradable",
        source_reference="Synthetic Test Benchmark (illustrative fixture)",
    )
    assert profile.material == "Polyester"
    assert profile.co2_kg_per_kg == 9.5
    assert profile.microplastic_shedding_risk == "Severe"

    kb_result = KnowledgeBaseResult(
        profiles={"polyester": profile},
        unrecognized_materials=["unknown_synthetic_fiber"],
    )
    assert "polyester" in kb_result.profiles
    assert len(kb_result.unrecognized_materials) == 1


def test_scoring_result_schema():
    """Verify Maneesh's Scoring Engine output model enforces score limits."""
    score = ScoringResult(
        overall_impact_score=72.0,
        recommendation=RecommendationTier.THINK_TWICE.value,
        breakdown=ImpactScoreBreakdown(
            water=IndicatorScore(score=30.0, rating="Low"),
            carbon=IndicatorScore(score=75.0, rating="High"),
            microfibres=IndicatorScore(score=90.0, rating="Critical"),
        ),
        key_indicators={"microplastics": "Severe concern"},
    )
    assert score.overall_impact_score == 72.0
    assert score.recommendation == "Think Twice"
    assert score.breakdown.microfibres.score == 90.0

    # Test score boundary violation
    with pytest.raises(ValidationError):
        ScoringResult(overall_impact_score=105.0, recommendation="Buy")

    unavailable = ScoringResult(
        key_indicators={"status": "INSUFFICIENT_DATA", "reason": "Incomplete composition"},
        calculation_basis="unscored_due_to_invalid_composition",
    )
    assert unavailable.overall_impact_score is None
    assert unavailable.recommendation is None


# ==============================================================================
# 4. Master Unified Response & Serialization Tests
# ==============================================================================

def test_full_analysis_response_serialization_and_roundtrip():
    """Verify complete end-to-end master response schema serializes and deserializes."""
    response = AnalysisResponse(
        analysis_id="gw-test-uuid-1234",
        timestamp="2026-11-13T18:00:00Z",
        product=ProductAnalysis(
            composition=Composition(
                materials=[
                    Material(name="polyester", percentage=65.0),
                    Material(name="cotton", percentage=35.0),
                ],
                confidence=0.94,
                source=AnalysisSource.OCR_LABEL,
            ),
            visual_prediction=VisionPrediction(
                predicted_material="polyester",
                confidence=0.88,
            ),
            composition_source=AnalysisSource.OCR_LABEL,
        ),
        environmental_data={
            "polyester": MaterialEnvironmentalProfile(
                material="polyester",
                co2_kg_per_kg=9.5,
                microplastic_shedding_risk="Severe",
                source_reference="Synthetic Test Benchmark (illustrative test profile)",
            )
        },
        impact_assessment=ScoringResult(
            overall_impact_score=72.0,
            recommendation="Think Twice",
        ),
        ai_explanation=AiExplanation(
            summary="High synthetic content garment requiring microfibre care.",
            environmental_highlights=["65% polyester produces significant microplastic shedding."],
            practical_recommendations=["Use a microfibre wash bag."],
        ),
        confidence={"ocr": 0.94, "vision": 0.88},
        limitations=["Analysis based on standard domestic wash cycle estimates."],
        warnings=[],
        metadata=AnalysisMetadata(
            processing_time_ms=350.2,
            model_versions={"ocr": "TrOCR-v1", "gemini": "gemini-2.5-flash"},
        ),
    )

    json_str = response.model_dump_json(indent=2)
    assert "gw-test-uuid-1234" in json_str
    assert "Think Twice" in json_str

    # Roundtrip validation
    parsed = AnalysisResponse.model_validate_json(json_str)
    assert parsed.analysis_id == "gw-test-uuid-1234"
    assert parsed.product.composition.materials[0].name == "polyester"
    assert parsed.impact_assessment.overall_impact_score == 72.0


def test_error_response_schema():
    """Verify standardized error envelope schema."""
    err = ErrorResponse(
        success=False,
        error=ErrorDetail(
            type="ImageValidationError",
            message="Image size exceeds 10 MB.",
            details={"max_size": 10485760},
        ),
    )
    assert err.success is False
    assert err.error.type == "ImageValidationError"
    assert err.error.details["max_size"] == 10485760
