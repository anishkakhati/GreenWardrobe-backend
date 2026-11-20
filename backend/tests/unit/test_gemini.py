"""Unit tests for GeminiAdapter and GeminiService.

All tests use mocks, test doubles, or deterministic fallbacks.
NO live Google Gemini API calls are made.
NO real API keys are used.
"""

from copy import deepcopy
from unittest.mock import MagicMock, patch
import pytest

from app.integrations.gemini_adapter import GeminiAdapter
from app.schemas.common import AnalysisSource
from app.schemas.response import (
    AiExplanation,
    Composition,
    Material,
    ProductAnalysis,
    ScoringResult,
    VisionPrediction,
)
from app.services.gemini_service import GeminiService


@pytest.fixture
def sample_analysis_data():
    """Sample structured Phase 5 data for testing Gemini grounding."""
    return {
        "product": {
            "composition": {
                "materials": [
                    {"name": "cotton", "percentage": 70.0},
                    {"name": "polyester", "percentage": 30.0},
                ],
                "confidence": 0.88,
                "source": "ocr_label",
            },
            "visual_prediction": {
                "predicted_material": "cotton",
                "confidence": 0.82,
                "alternatives": ["linen", "polyolefin"],
            },
            "composition_source": "ocr_label",
        },
        "environmental_data": {
            "cotton": {
                "material": "cotton",
                "water_liters_per_kg": 9500.0,
                "co2_kg_per_kg": 5.2,
                "source_reference": "EEA / UNEP Sustainable Textiles",
            },
            "polyester": {
                "material": "polyester",
                "water_liters_per_kg": 1500.0,
                "co2_kg_per_kg": 9.5,
                "source_reference": "Textile Exchange 2023",
            },
        },
        "impact_assessment": {
            "overall_impact_score": 68.5,
            "recommendation": "Think Twice",
            "breakdown": {
                "water": {"score": 80.0, "rating": "High"},
                "carbon": {"score": 65.0, "rating": "Moderate"},
                "microfibres": {"score": 90.0, "rating": "Critical"},
            },
            "key_indicators": {"impact_classification": "High Impact"},
        },
        "confidence": {"ocr": 0.88, "vision": 0.82},
        "warnings": ["Surface computer vision agrees with label cotton."],
        "limitations": ["LCA figures are comparative median benchmarks."],
    }


def test_gemini_adapter_status_unconfigured():
    """Verify GeminiAdapter status info when no API key is provided."""
    adapter = GeminiAdapter(api_key="")
    status = adapter.status_info()
    assert status["adapter"] == "Google-Gemini-AI-Adapter"
    assert status["api_key_configured"] is False
    assert status["status"] == "CONFIGURATION_REQUIRED"
    assert status["configured_model"] == "gemini-3.5-flash-lite"
    assert status["max_retries"] == 2
    assert adapter.is_available is False


def test_gemini_adapter_status_configured():
    """Verify GeminiAdapter status info when API key is set."""
    adapter = GeminiAdapter(api_key="mock_test_key_abc123")
    status = adapter.status_info()
    assert status["api_key_configured"] is True
    assert status["status"] == "OPERATIONAL"
    assert adapter.is_available is True
    # Ensure raw API key is never exposed in status info
    assert "mock_test_key_abc123" not in str(status)


def test_gemini_adapter_injected_generator(sample_analysis_data):
    """Verify GeminiAdapter invokes injected custom generator double."""
    mock_explanation = AiExplanation(
        status="AVAILABLE",
        summary="This shirt is 70% cotton and 30% polyester with elevated water and microfiber impacts.",
        environmental_highlights=["High water usage from cotton", "Microfiber shedding from polyester"],
        practical_recommendations=["Think twice before purchasing"],
        care_and_repair_tips=["Wash cold in a microfibre wash bag"],
        limitations_and_uncertainty=["Baseline data"],
        is_fallback=False,
    )

    adapter = GeminiAdapter(custom_generator=lambda payload: mock_explanation)
    assert adapter.is_available is True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))
    assert result.status == "AVAILABLE"
    assert result.summary == mock_explanation.summary
    assert result.is_fallback is False


def test_gemini_adapter_deterministic_fallback_unconfigured(sample_analysis_data):
    """Verify GeminiAdapter generates grounded deterministic fallback when API key is unconfigured."""
    adapter = GeminiAdapter(api_key="")
    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is True
    assert result.status == "UNAVAILABLE"
    assert "70% Cotton" in result.summary
    assert "30% Polyester" in result.summary
    assert "68.5" in result.summary
    assert "Think Twice" in result.summary
    assert len(result.environmental_highlights) > 0
    assert len(result.care_and_repair_tips) > 0
    assert result.data_quality_disclaimer is not None


def test_gemini_adapter_timeout_configuration(sample_analysis_data):
    """TASK 2 & 7: Verify GEMINI_TIMEOUT_SECONDS controls timeout in SDK client and generate call."""
    adapter = GeminiAdapter(api_key="test_key_xyz", timeout_seconds=15)
    assert adapter.timeout_seconds == 15

    mock_response = MagicMock()
    mock_response.text = '{"status": "AVAILABLE", "summary": "Valid response", "environmental_highlights": [], "practical_recommendations": [], "care_and_repair_tips": [], "limitations_and_uncertainty": [], "data_quality_disclaimer": "test", "is_fallback": false}'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.status == "AVAILABLE"
    assert mock_client.models.generate_content.called

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "config" in call_kwargs
    config = call_kwargs["config"]
    # Check that HttpOptions timeout is set to 15,000 milliseconds
    assert config.http_options.timeout == 15000


def test_gemini_adapter_timeout_fallback_behavior(sample_analysis_data):
    """TASK 2 & 7: Verify GeminiAdapter catches timeout and returns TIMEOUT status fallback."""
    adapter = GeminiAdapter(api_key="valid_looking_key", timeout_seconds=10)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Request timed out after 10000ms")
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is True
    assert result.status == "TIMEOUT"
    assert "70% Cotton" in result.summary
    assert "68.5" in result.summary
    assert any("timed out" in lim.lower() for lim in result.limitations_and_uncertainty)


def test_gemini_adapter_rate_limit_bounded_retry(sample_analysis_data):
    """TASK 3 & 7: Verify GeminiAdapter retries transient 429 rate limit with backoff and succeeds."""
    mock_sleep = MagicMock()
    adapter = GeminiAdapter(
        api_key="valid_looking_key",
        max_retries=2,
        initial_backoff=0.2,
        backoff_multiplier=2.0,
        sleep_fn=mock_sleep,
    )

    valid_json = """{
      "status": "AVAILABLE",
      "summary": "Generated successfully after transient retry.",
      "environmental_highlights": ["Recovered after rate limit"],
      "practical_recommendations": ["Buy durable pieces"],
      "care_and_repair_tips": ["Wash cold"],
      "limitations_and_uncertainty": ["Comparative benchmarks"],
      "data_quality_disclaimer": "Standard disclaimer.",
      "is_fallback": false
    }"""
    mock_response = MagicMock()
    mock_response.text = valid_json

    mock_client = MagicMock()
    # First attempt fails with 429 rate limit, second attempt succeeds
    mock_client.models.generate_content.side_effect = [
        RuntimeError("429 ResourceExhausted: Quota exceeded for model"),
        mock_response,
    ]
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    # Should succeed on 2nd attempt
    assert result.is_fallback is False
    assert result.status == "AVAILABLE"
    assert result.summary == "Generated successfully after transient retry."
    assert mock_client.models.generate_content.call_count == 2
    # Verify sleep was called once with initial backoff
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_once_with(0.2)


def test_gemini_adapter_auth_error_not_retried(sample_analysis_data):
    """TASK 3 & 7: Verify authentication errors (401/403) are NOT retried unnecessarily."""
    mock_sleep = MagicMock()
    adapter = GeminiAdapter(
        api_key="invalid_test_key",
        max_retries=2,
        sleep_fn=mock_sleep,
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("401 Invalid API Key / Permission Denied")
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is True
    assert result.status == "UNAVAILABLE"
    # Must fail fast: exactly 1 call, 0 sleep/retry attempts
    assert mock_client.models.generate_content.call_count == 1
    assert mock_sleep.call_count == 0
    assert any("permission denied" in lim.lower() or "invalid api key" in lim.lower() for lim in result.limitations_and_uncertainty)


def test_gemini_adapter_retry_exhaustion_returns_fallback(sample_analysis_data):
    """TASK 3 & 7: Verify retry exhaustion after max_retries returns RATE_LIMITED fallback."""
    mock_sleep = MagicMock()
    adapter = GeminiAdapter(
        api_key="valid_looking_key",
        max_retries=2,
        initial_backoff=0.1,
        backoff_multiplier=2.0,
        sleep_fn=mock_sleep,
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("429 ResourceExhausted: Quota exceeded")
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is True
    assert result.status == "RATE_LIMITED"
    # Total calls: 1 initial + 2 retries = 3 calls
    assert mock_client.models.generate_content.call_count == 3
    # Slept twice: 0.1s then 0.2s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(0.1)
    mock_sleep.assert_any_call(0.2)


def test_gemini_fallback_missing_category_no_fabricated_scores(sample_analysis_data):
    """TASK 4 & 7: Verify missing environmental category data does NOT produce fabricated category scores."""
    adapter = GeminiAdapter(api_key="")

    data = deepcopy(sample_analysis_data)
    # Breakdown only contains water; carbon, energy, chemicals, microfibres, waste are missing
    data["impact_assessment"]["breakdown"] = {
        "water": {"score": 80.0, "rating": "High"}
    }

    fallback = adapter.generate_fallback(data, reason="Test missing categories")

    assert fallback.is_fallback is True
    cat_expl = fallback.impact_category_explanations

    # Water should have real score
    assert "80.0/100" in cat_expl["water"]

    # Missing categories must NOT have fabricated scores like "50.0/100"
    for missing_cat in ["carbon", "energy", "chemicals", "microfibres", "waste"]:
        assert missing_cat in cat_expl
        msg = cat_expl[missing_cat]
        assert "no score assigned" in msg.lower() or "not available" in msg.lower()
        # Verify no fabricated numeric score format in missing categories
        assert "/100" not in msg


def test_gemini_fallback_preserves_actual_composition_scores_and_tier(sample_analysis_data):
    """TASK 4 & 7: Verify fallback preserves actual composition, scores, and recommendation tier."""
    adapter = GeminiAdapter(api_key="")

    fallback = adapter.generate_fallback(sample_analysis_data, reason="Preservation test")

    assert fallback.is_fallback is True
    # Preserves actual composition
    assert "70% Cotton" in fallback.summary
    assert "30% Polyester" in fallback.summary
    # Preserves actual scores
    assert "68.5" in fallback.summary
    # Preserves actual recommendation tier
    assert "Think Twice" in fallback.summary
    assert fallback.practical_recommendations[0] == "Recommended decision tier: Think Twice."
    # Ensures no calibration claims
    full_text = str(fallback.model_dump()).lower()
    assert "calibrated" not in full_text
    # Disclaimer is present
    assert fallback.data_quality_disclaimer is not None
    assert "comparative benchmarks" in fallback.data_quality_disclaimer.lower()


def test_gemini_fallback_missing_overall_impact_score(sample_analysis_data):
    """Verify fallback handles missing overall_impact_score without fabricating 50.0."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    del data["impact_assessment"]["overall_impact_score"]

    fallback = adapter.generate_fallback(data, reason="Missing overall score")

    assert fallback.is_fallback is True
    assert "score is unavailable" in fallback.summary.lower()
    assert "50.0" not in fallback.summary
    assert "Think Twice" in fallback.summary
    assert fallback.practical_recommendations[0] == "Recommended decision tier: Think Twice."
    assert "score is unavailable" in fallback.environmental_impact_explanation.lower()


def test_gemini_fallback_overall_impact_score_none(sample_analysis_data):
    """Verify fallback handles overall_impact_score explicitly set to None."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    data["impact_assessment"]["overall_impact_score"] = None

    fallback = adapter.generate_fallback(data, reason="None overall score")

    assert fallback.is_fallback is True
    assert "score is unavailable" in fallback.summary.lower()
    assert "50.0" not in fallback.summary
    assert "score is unavailable" in fallback.environmental_impact_explanation.lower()
    assert fallback.practical_recommendations[0] == "Recommended decision tier: Think Twice."


def test_gemini_fallback_missing_recommendation(sample_analysis_data):
    """Verify fallback handles missing recommendation tier without fabricating 'Think Twice'."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    del data["impact_assessment"]["recommendation"]

    fallback = adapter.generate_fallback(data, reason="Missing recommendation")

    assert fallback.is_fallback is True
    assert "no recommendation tier was produced" in fallback.summary.lower()
    assert "Think Twice" not in fallback.summary
    assert "68.5" in fallback.summary
    assert "no recommendation tier was produced" in fallback.practical_recommendations[0].lower()


def test_gemini_fallback_recommendation_none(sample_analysis_data):
    """Verify fallback handles recommendation explicitly set to None."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    data["impact_assessment"]["recommendation"] = None

    fallback = adapter.generate_fallback(data, reason="None recommendation")

    assert fallback.is_fallback is True
    assert "no recommendation tier was produced" in fallback.summary.lower()
    assert "Think Twice" not in fallback.summary
    assert "68.5" in fallback.summary
    assert "no recommendation tier was produced" in fallback.practical_recommendations[0].lower()


def test_gemini_fallback_both_score_and_recommendation_missing(sample_analysis_data):
    """Verify fallback handles both overall score and recommendation missing simultaneously."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    data["impact_assessment"]["overall_impact_score"] = None
    data["impact_assessment"]["recommendation"] = None

    fallback = adapter.generate_fallback(data, reason="Both missing")

    assert fallback.is_fallback is True
    assert "score is unavailable" in fallback.summary.lower()
    assert "no recommendation tier was produced" in fallback.summary.lower()
    assert "50.0" not in fallback.summary
    assert "Think Twice" not in fallback.summary
    assert "score and recommendation tier are unavailable" in fallback.environmental_impact_explanation.lower()
    assert "no recommendation tier was produced" in fallback.practical_recommendations[0].lower()


def test_gemini_fallback_no_unsupported_second_hand_claim(sample_analysis_data):
    """Verify recommendation tier 'Second-hand' does not make unsupported environmental claims."""
    adapter = GeminiAdapter(api_key="")
    data = deepcopy(sample_analysis_data)
    data["impact_assessment"]["recommendation"] = "Second-hand"

    fallback = adapter.generate_fallback(data, reason="Second-hand rec")

    full_text = str(fallback.model_dump()).lower()
    assert "significantly offsets" not in full_text
    assert "virgin production impacts" not in full_text
    assert any("may extend the useful life" in rec.lower() for rec in fallback.practical_recommendations)
    assert any("reduce demand for new production" in rec.lower() for rec in fallback.practical_recommendations)


def test_gemini_adapter_malformed_json_handling(sample_analysis_data):
    """Verify GeminiAdapter handles malformed non-JSON output gracefully."""
    adapter = GeminiAdapter(api_key="valid_looking_key")

    mock_response = MagicMock()
    mock_response.text = "This is not JSON at all! Just conversational text."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is True
    assert result.status == "PARSE_ERROR"
    assert "70% Cotton" in result.summary


def test_gemini_adapter_valid_json_parsing(sample_analysis_data):
    """Verify GeminiAdapter parses valid structured JSON and creates AiExplanation."""
    adapter = GeminiAdapter(api_key="valid_looking_key")

    valid_json = """
    {
      "status": "AVAILABLE",
      "summary": "This blend is mostly cotton with polyester reinforcement.",
      "material_explanation": "Cotton provides breathable comfort while polyester adds tensile strength.",
      "environmental_impact_explanation": "Moderate to high overall footprint due to synthetic microfibers.",
      "impact_category_explanations": {
        "water": "Cotton requires high irrigation volume.",
        "microfibres": "Polyester sheds non-biodegradable particles."
      },
      "environmental_highlights": ["High water demand", "Microplastic risk"],
      "practical_recommendations": ["Look for certified organic alternatives"],
      "care_and_repair_tips": ["Wash cold in a Guppyfriend bag"],
      "limitations_and_uncertainty": ["Baseline estimates"],
      "data_quality_disclaimer": "Standard disclaimer."
    }
    """
    mock_response = MagicMock()
    mock_response.text = f"```json\n{valid_json}\n```"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    adapter._client = mock_client
    adapter._client_initialized = True

    import asyncio
    result = asyncio.run(adapter.generate_explanation(sample_analysis_data))

    assert result.is_fallback is False
    assert result.status == "AVAILABLE"
    assert result.summary == "This blend is mostly cotton with polyester reinforcement."
    assert "water" in result.impact_category_explanations
    assert len(result.practical_recommendations) == 1


def test_gemini_adapter_prompt_grounding_contents(sample_analysis_data):
    """TASK 6: Verify prompt formulation enforces grounding rules and describes system accurately."""
    adapter = GeminiAdapter()
    prompt = adapter._build_grounding_prompt(sample_analysis_data)

    # Verify immutability instruction
    assert "Ground Truth Immutability" in prompt
    assert "MUST NOT modify, recalculate, contradict" in prompt
    assert "Prompt Injection Defense" in prompt
    assert "No Data Fabrication" in prompt
    assert "Distinguish Inputs vs Reference Benchmarks" in prompt
    # Verify no calibration claims in prompt
    assert "calibrated" not in prompt.lower()
    # Verify accurate description
    assert "rule-based comparative scoring engine" in prompt
    # Verify Phase 5 facts are present in prompt
    assert "cotton" in prompt
    assert "70.0" in prompt
    assert "68.5" in prompt
    assert "Think Twice" in prompt


def test_gemini_service_explain_analysis():
    """Verify GeminiService correctly converts models and delegates to adapter."""
    mock_explanation = AiExplanation(
        status="AVAILABLE",
        summary="Service test explanation",
        environmental_highlights=["highlight"],
        practical_recommendations=["recommendation"],
        care_and_repair_tips=["care tip"],
        limitations_and_uncertainty=["limitation"],
        is_fallback=False,
    )
    mock_adapter = GeminiAdapter(custom_generator=lambda payload: mock_explanation)
    service = GeminiService(adapter=mock_adapter)

    product = ProductAnalysis(
        composition=Composition(
            materials=[Material(name="wool", percentage=100.0)],
            confidence=0.95,
            source=AnalysisSource.OCR_LABEL,
        ),
        visual_prediction=VisionPrediction(predicted_material="wool", confidence=0.88),
        composition_source=AnalysisSource.OCR_LABEL,
    )
    impact = ScoringResult(
        overall_impact_score=45.0,
        recommendation="Buy",
        key_indicators={},
    )

    import asyncio
    result = asyncio.run(service.explain_analysis(product, {}, impact))
    assert result.status == "AVAILABLE"
    assert result.summary == "Service test explanation"


def test_gemini_service_public_fallback_method(sample_analysis_data):
    """TASK 5: Verify GeminiService exposes public generate_fallback_explanation method."""
    service = GeminiService()
    fallback = service.generate_fallback_explanation(
        analysis_data=sample_analysis_data,
        reason="Testing public service method",
        status="UNAVAILABLE",
    )
    assert fallback.is_fallback is True
    assert fallback.status == "UNAVAILABLE"
    assert "70% Cotton" in fallback.summary
    assert "68.5" in fallback.summary


def test_gemini_service_resilience_preserves_pipeline_data(sample_analysis_data):
    """TASK 5 & 7: Verify unhandled Gemini adapter exception never destroys Phase 5 results."""
    mock_adapter = MagicMock()
    mock_adapter.generate_explanation.side_effect = RuntimeError("Fatal network crash in Gemini SDK")
    mock_adapter.generate_fallback = GeminiAdapter().generate_fallback

    service = GeminiService(adapter=mock_adapter)

    product = ProductAnalysis(
        composition=Composition(
            materials=[Material(name="linen", percentage=100.0)],
            confidence=0.90,
            source=AnalysisSource.OCR_LABEL,
        ),
        visual_prediction=VisionPrediction(predicted_material="linen", confidence=0.85),
        composition_source=AnalysisSource.OCR_LABEL,
    )
    impact = ScoringResult(
        overall_impact_score=35.0,
        recommendation="Buy",
        key_indicators={},
    )

    import asyncio
    result = asyncio.run(service.explain_analysis(product, {}, impact))

    # Result must not crash; returns safe grounded fallback preserving linen and 35.0 score
    assert result.is_fallback is True
    assert result.status == "UNAVAILABLE"
    assert "100% Linen" in result.summary
    assert "35.0" in result.summary
