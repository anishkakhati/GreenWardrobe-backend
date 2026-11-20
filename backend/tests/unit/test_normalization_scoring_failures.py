"""Scoring failure responses must preserve analysis without fabricated results."""

import asyncio
import importlib

import pytest

from app.integrations.ocr_adapter import OCRAdapter
from app.integrations.scoring_adapter import ScoringAdapter
from app.core.exceptions import ScoringError
from app.integrations.teammate_modules.scoring.scoring import EnvironmentalScorer
from app.integrations.vision_adapter import VisionAdapter
from app.schemas.response import OCRCompositionItem, OCRResult
from app.services.normalization_service import NormalizationService


class UnusedVisionAdapter:
    """Avoid running CV for label-only pipeline cases in these tests."""

    def status_info(self):
        return {"adapter": "unused-test-vision"}


class ExplodingScorer:
    def calculate_impact(self, composition, **kwargs):
        raise RuntimeError("test scoring engine failure")

    def status_info(self):
        return {"adapter": "exploding-test-scorer"}


def make_ocr_adapter(percentages):
    result = OCRResult(
        composition=[
            OCRCompositionItem(material=material, percentage=percentage)
            for material, percentage in percentages
        ],
        confidence=0.87,
        raw_text="fixture label text",
        unreadable=not percentages,
        warnings=["OCR fixture warning"] if percentages else [],
    )
    return OCRAdapter(custom_extractor=lambda image: result)


async def run_label_pipeline(make_image_bytes, percentages, scoring_adapter=None):
    return await NormalizationService.execute_pipeline(
        label_image_bytes=make_image_bytes(fmt="JPEG"),
        label_filename="test-label.jpg",
        ocr_adapter_instance=make_ocr_adapter(percentages),
        vision_adapter_instance=UnusedVisionAdapter(),
        scoring_adapter_instance=scoring_adapter,
    )


@pytest.mark.parametrize(
    "composition",
    [
        [("Wool", 84.0), ("Acrylic", 84.0)],  # 168%, matching label_0001 OCR failure
        [("Cotton", 80.0)],  # incomplete composition rejected by the scorer
    ],
)
def test_invalid_composition_preserves_ocr_and_returns_no_score_or_recommendation(
    make_image_bytes, composition
):
    response = asyncio.run(run_label_pipeline(make_image_bytes, composition))

    assert response.success is True
    assert [(m.name, m.percentage) for m in response.product.composition.materials] == [
        (name.lower(), percentage) for name, percentage in composition
    ]
    assert response.product.composition.confidence == 0.87
    assert response.confidence["ocr"] == 0.87
    assert "OCR fixture warning" in response.warnings
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert response.impact_assessment.key_indicators["status"] == "INSUFFICIENT_DATA"
    assert response.impact_assessment.key_indicators["reason"]
    assert any("no score or recommendation was produced" in warning.lower() for warning in response.warnings)


def test_valid_100_percent_composition_receives_real_score_and_recommendation(make_image_bytes):
    response = asyncio.run(run_label_pipeline(make_image_bytes, [("Cotton", 100.0)]))

    assert response.success is True
    assert response.impact_assessment.overall_impact_score is not None
    assert response.impact_assessment.recommendation is not None
    assert response.impact_assessment.calculation_basis == "rule_based_lca_weighted_average"


def test_scoring_engine_exception_returns_null_score_and_recommendation(make_image_bytes):
    response = asyncio.run(
        run_label_pipeline(
            make_image_bytes,
            [("Cotton", 100.0)],
            scoring_adapter=ExplodingScorer(),
        )
    )

    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert response.impact_assessment.key_indicators["status"] == "ENGINE_ERROR"
    assert response.impact_assessment.key_indicators["reason"] == "The scoring service is unavailable or failed."
    assert "test scoring engine failure" not in str(response.model_dump())
    assert any("scoring service is unavailable or failed" in warning.lower() for warning in response.warnings)


class WrappedEngineErrorScorer:
    def calculate_impact(self, composition, **kwargs):
        raise ScoringError("private engine detail", status_code=500)

    def status_info(self):
        return {"adapter": "wrapped-engine-error"}


def test_wrapped_scoring_error_500_is_engine_error_and_preserves_valid_data(make_image_bytes):
    response = asyncio.run(
        run_label_pipeline(
            make_image_bytes,
            [("Cotton", 100.0)],
            scoring_adapter=WrappedEngineErrorScorer(),
        )
    )

    assert [(m.name, m.percentage) for m in response.product.composition.materials] == [("cotton", 100.0)]
    assert "cotton" in response.environmental_data
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert response.impact_assessment.key_indicators["status"] == "ENGINE_ERROR"
    assert "private engine detail" not in str(response.model_dump())
    assert any("scoring service is unavailable or failed" in w.lower() for w in response.warnings)


def test_scoring_engine_initialization_failure_is_engine_error(make_image_bytes, monkeypatch):
    def fail_initialization(*args, **kwargs):
        raise RuntimeError("private initialization detail")

    scoring_module = importlib.import_module("app.integrations.scoring_adapter")
    monkeypatch.setattr(scoring_module, "EnvironmentalScorer", fail_initialization)
    adapter = ScoringAdapter()
    response = asyncio.run(
        run_label_pipeline(make_image_bytes, [("Cotton", 100.0)], scoring_adapter=adapter)
    )

    assert response.impact_assessment.key_indicators["status"] == "ENGINE_ERROR"
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert "private initialization detail" not in str(response.model_dump())


def test_recommendation_engine_runtime_failure_is_engine_error(make_image_bytes):
    class FailingRecommendationEngine:
        def recommend(self, *args, **kwargs):
            raise ValueError("private recommendation detail")

    adapter = ScoringAdapter(
        scorer=EnvironmentalScorer(),
        recommendation_engine=FailingRecommendationEngine(),
    )
    response = asyncio.run(
        run_label_pipeline(make_image_bytes, [("Cotton", 100.0)], scoring_adapter=adapter)
    )

    assert response.impact_assessment.key_indicators["status"] == "ENGINE_ERROR"
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert "private recommendation detail" not in str(response.model_dump())


def test_unknown_material_is_insufficient_data_and_keeps_composition(make_image_bytes):
    response = asyncio.run(run_label_pipeline(make_image_bytes, [("Unobtainium", 100.0)]))

    assert [(m.name, m.percentage) for m in response.product.composition.materials] == [("unobtainium", 100.0)]
    assert response.impact_assessment.key_indicators["status"] == "INSUFFICIENT_DATA"
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None


@pytest.mark.parametrize("total", [98.0, 99.0, 99.9, 100.1, 101.0, 102.0])
def test_out_of_tolerance_composition_is_preserved_but_not_scored(make_image_bytes, total):
    composition = [("Cotton", 50.0), ("Polyester", round(total - 50.0, 2))]
    response = asyncio.run(run_label_pipeline(make_image_bytes, composition))

    assert [(m.name, m.percentage) for m in response.product.composition.materials] == [
        (name.lower(), percentage) for name, percentage in composition
    ]
    assert response.impact_assessment.key_indicators["status"] == "INSUFFICIENT_DATA"
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert any("within ±0.01%" in warning for warning in response.warnings)


@pytest.mark.parametrize("total", [99.99, 100.0, 100.01])
def test_in_tolerance_composition_is_scored_without_rewriting_percentages(make_image_bytes, total):
    composition = [("Cotton", 50.0), ("Polyester", round(total - 50.0, 2))]
    response = asyncio.run(run_label_pipeline(make_image_bytes, composition))

    assert [m.percentage for m in response.product.composition.materials] == [
        percentage for _, percentage in composition
    ]
    assert response.impact_assessment.key_indicators.get("status") is None
    assert response.impact_assessment.overall_impact_score is not None


def test_missing_composition_returns_null_score_and_recommendation(make_image_bytes):
    response = asyncio.run(run_label_pipeline(make_image_bytes, []))

    assert response.success is False
    assert response.product.composition is None
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert response.impact_assessment.key_indicators["status"] == "NO_COMPOSITION_DATA"


def test_label_pipeline_status_reporting_does_not_load_cv_model(make_image_bytes, tmp_path):
    checkpoint = tmp_path / "checkpoint-available-but-not-loaded.pth"
    checkpoint.write_bytes(b"placeholder asset; status must not load it")
    vision = VisionAdapter(checkpoint_path=checkpoint)
    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            label_filename="label.jpg",
            ocr_adapter_instance=make_ocr_adapter([("Cotton", 100.0)]),
            vision_adapter_instance=vision,
        )
    )

    assert response.product.visual_prediction is None
    assert response.confidence["vision"] is None
    assert vision._initialized is False
    assert vision._model is None


def test_duplicate_material_names_are_combined_without_rewriting_percentages():
    from app.schemas.response import OCRCompositionItem, OCRResult

    result = OCRResult(
        composition=[
            OCRCompositionItem(material="Cotton", percentage=60),
            OCRCompositionItem(material="COTTON", percentage=40),
        ],
        unreadable=False,
    )
    composition, warnings = NormalizationService.normalize_ocr_composition(result)

    assert [(item.name, item.percentage) for item in composition.materials] == [("cotton", 100.0)]
    assert not warnings
