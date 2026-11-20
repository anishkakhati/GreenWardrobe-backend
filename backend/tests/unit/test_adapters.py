"""Comprehensive unit tests for module integration adapters and normalization service.

Covers:
- ScoringAdapter with real Maneesh engine execution.
- OCRAdapter with Harshit's regex parsing and dependency isolation.
- VisionAdapter with Anshika's ResNet18 asset bounds and test double injection.
- KnowledgeBaseAdapter contract and benchmark retrieval.
- NormalizationService alias mapping, percentage validation, and perception conflict detection.
- Real OCR live inference on clothing-label image fixtures.
- End-to-end cross-module data exchange pipeline.
"""

from pathlib import Path
import pytest

from app.core.exceptions import ScoringError
from app.integrations.knowledge_adapter import KnowledgeBaseAdapter, knowledge_adapter
from app.integrations.ocr_adapter import OCRAdapter, ocr_adapter
from app.integrations.scoring_adapter import ScoringAdapter, scoring_adapter
from app.integrations.vision_adapter import VisionAdapter, vision_adapter
from app.schemas.common import AnalysisSource
from app.schemas.response import (
    AnalysisResponse,
    OCRCompositionItem,
    OCRResult,
    VisionPrediction,
)
from app.services.image_service import ImageService
from app.services.normalization_service import NormalizationService, normalization_service


# ==============================================================================
# 1. Scoring Adapter Tests (Real Maneesh Engine Integration)
# ==============================================================================

def test_scoring_adapter_status_info():
    """Verify ScoringAdapter status metadata indicates verified integration."""
    status = scoring_adapter.status_info()
    assert status["adapter"] == "Maneesh-ScoringEngine-Adapter"
    assert status["status"] == "REAL INTEGRATION VERIFIED"
    assert "carbon" in status["weights_configured"]
    assert status["available_materials_count"] >= 8


def test_scoring_adapter_real_blend_calculation():
    """Verify real calculation on 65% polyester / 35% cotton blend."""
    comp = {"polyester": 65.0, "cotton": 35.0}
    result = scoring_adapter.calculate_impact(comp, shopping=True)

    assert result.overall_impact_score > 0.0
    assert result.overall_impact_score <= 100.0
    assert result.recommendation in ["Buy", "Think Twice"]
    assert result.breakdown is not None
    assert result.breakdown.water is not None
    assert result.breakdown.carbon is not None
    assert result.breakdown.microfibres is not None
    # Polyester has high microfibre shedding in Maneesh's benchmark
    assert result.breakdown.microfibres.score > 60.0


def test_scoring_adapter_garment_condition_override():
    """Verify damaged garment recommendation overrides score to 'Repair'."""
    comp = {"cotton": 100.0}
    result = scoring_adapter.calculate_impact(comp, garment_condition="damaged")
    assert result.recommendation == "Repair"
    assert "repair" in result.key_indicators["recommendation_reason"].lower()


def test_scoring_adapter_unknown_material_error():
    """Verify unknown material raises ScoringError with clear message."""
    with pytest.raises(ScoringError) as exc:
        scoring_adapter.calculate_impact({"kryptonite_fiber": 100.0})
    assert exc.value.status_code == 422
    assert "unknown material" in exc.value.message.lower()


def test_scoring_insufficient_data_coverage_rule(tmp_path):
    """Verify minimum 60% category weight coverage rule rejects insufficient data."""
    import pandas as pd
    from app.integrations.teammate_modules.scoring import EnvironmentalScorer

    # Create dummy dataset where all columns except water are missing (water weight = 0.20 < 0.60)
    sparse_csv = tmp_path / "sparse.csv"
    df = pd.DataFrame([
        {"material": "sparse_mat", "water": 50.0, "carbon": None, "energy": None, "chemicals": None, "microfibres": None, "waste": None}
    ])
    df.to_csv(sparse_csv, index=False)

    scorer = EnvironmentalScorer(data_path=sparse_csv)
    with pytest.raises(ValueError) as exc:
        scorer.score({"sparse_mat": 100.0})
    assert "Insufficient environmental data coverage" in str(exc.value)
    assert "60%" in str(exc.value)


# ==============================================================================
# 2. OCR Adapter & Parser Tests (Harshit's Logic)
# ==============================================================================

def test_ocr_adapter_status_info():
    """Verify OCRAdapter status cleanly identifies engine state."""
    status = ocr_adapter.status_info()
    assert status["adapter"] == "Harshit-ClothingLabelOCR-Adapter"
    assert "COTTON" in status["supported_fibres"]
    assert "POLYESTER" in status["supported_fibres"]


def test_ocr_parser_regex_extraction():
    """Verify Harshit's regex parses multiple fibers in standard label formats."""
    raw_label = "FABRIC MADE IN PORTUGAL. 80% COTTON 20% POLYESTER. EXCLUSIVE OF TRIMMINGS."
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition
    parsed = parse_composition(raw_label)

    assert "COTTON" in parsed
    assert parsed["COTTON"] == [80.0]
    assert "POLYESTER" in parsed
    assert parsed["POLYESTER"] == [20.0]


def test_ocr_adapter_execution_without_easyocr(make_image_bytes):
    """Verify OCRAdapter handles missing EasyOCR gracefully with transparent warning."""
    # Adapter with forced missing reader
    empty_adapter = OCRAdapter(ocr_reader=None)
    empty_adapter._initialized = True
    empty_adapter._easyocr_available = False

    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "label.jpg")

    import asyncio
    result = asyncio.run(empty_adapter.extract_composition(processed))
    assert result.unreadable is True
    assert len(result.warnings) > 0
    assert "not available" in result.warnings[0].lower()


def test_ocr_adapter_injected_double(make_image_bytes):
    """Verify OCRAdapter works with mock reader injection."""
    class FakeReader:
        def readtext(self, img, detail=1, paragraph=False):
            return [
                ([[0, 0], [10, 0], [10, 10], [0, 10]], "100% COTTON", 0.95)
            ]

    mock_adapter = OCRAdapter(ocr_reader=FakeReader())
    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "label.jpg")

    import asyncio
    result = asyncio.run(mock_adapter.extract_composition(processed))
    assert result.unreadable is False
    assert len(result.composition) == 1
    assert result.composition[0].material == "Cotton"
    assert result.composition[0].percentage == 100.0
    assert result.confidence == 0.95


def test_ocr_live_inference_single_fiber():
    """Verify actual EasyOCR inference on real label fixture (100% Linen)."""
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0003.jpg"
    if not fixture_path.exists():
        pytest.skip("Label fixture not found.")

    with open(fixture_path, "rb") as f:
        img_bytes = f.read()

    proc = ImageService.process_image(img_bytes, filename="label_0003.jpg")

    import asyncio
    res = asyncio.run(ocr_adapter.extract_composition(proc))

    assert res.unreadable is False
    assert len(res.composition) == 1
    assert res.composition[0].material.lower() == "linen"
    assert res.composition[0].percentage == 100.0
    assert res.confidence is not None
    assert res.confidence > 0.5


def test_ocr_live_inference_multi_fiber_blend():
    """Verify actual EasyOCR inference on real blend fixture (68% Nylon, 32% Wool)."""
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0002.jpg"
    if not fixture_path.exists():
        pytest.skip("Label fixture not found.")

    with open(fixture_path, "rb") as f:
        img_bytes = f.read()

    proc = ImageService.process_image(img_bytes, filename="label_0002.jpg")

    import asyncio
    res = asyncio.run(ocr_adapter.extract_composition(proc))

    assert res.unreadable is False
    materials_found = {c.material.lower(): c.percentage for c in res.composition}
    assert "nylon" in materials_found
    assert materials_found["nylon"] == 68.0
    assert "wool" in materials_found
    assert materials_found["wool"] == 32.0


def test_ocr_live_inference_label_0001_preserves_each_adjacent_percentage():
    """Regression: EasyOCR omits acrylic's percent sign but reads its value."""
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0001.jpg"
    proc = ImageService.process_image(fixture_path.read_bytes(), fixture_path.name, "image/jpeg")

    import asyncio
    result = asyncio.run(ocr_adapter.extract_composition(proc))

    materials = {item.material.lower(): item.percentage for item in result.composition}
    assert materials == {"acrylic": 16.0, "wool": 84.0}
    assert sum(materials.values()) == 100.0
    assert "16 ACRYLIC" in result.raw_text.upper()
    assert "84% WOOL" in result.raw_text.upper()


def test_ocr_parser_does_not_assign_next_fiber_percentage_to_previous_fiber():
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition

    assert parse_composition("16 ACRYLIC 84% WOOL") == {
        "ACRYLIC": [16.0],
        "WOOL": [84.0],
    }


def test_ocr_parser_preserves_repeated_text_block_observations():
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition

    assert parse_composition("16 ACRYLIC 84% WOOL 16 ACRYLIC 84% WOOL") == {
        "ACRYLIC": [16.0, 16.0],
        "WOOL": [84.0, 84.0],
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("80% COTTON 20% POLYESTER", {"COTTON": [80.0], "POLYESTER": [20.0]}),
        ("COTTON 80% POLYESTER 20%", {"COTTON": [80.0], "POLYESTER": [20.0]}),
        ("COTTON: 70%, POLYESTER - 30%", {"COTTON": [70.0], "POLYESTER": [30.0]}),
        ("COTTON (60%) POLYESTER (40%)", {"COTTON": [60.0], "POLYESTER": [40.0]}),
    ],
)
def test_ocr_parser_supported_percentage_syntax(text, expected):
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition

    assert parse_composition(text) == expected


def test_ocr_parser_unreadable_text_returns_no_composition():
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition

    assert parse_composition("CARE INSTRUCTIONS WASH COLD DO NOT BLEACH") == {}


# ==============================================================================
# 3. Vision Adapter Tests (Anshika's TextileNet ResNet-18)
# ==============================================================================

def test_vision_adapter_status_info():
    """Verify VisionAdapter status info accurately reflects model and configuration."""
    status = vision_adapter.status_info()
    assert status["adapter"] == "Anshika-TextileNet-Vision-Adapter"
    assert status["classes_count"] == 33
    assert status["status_code"] in (
        "OPERATIONAL",
        "ASSET_AVAILABLE_NOT_LOADED",
        "MODEL_CHECKPOINT_MISSING",
        "CHECKPOINT_UNAVAILABLE",
    )
    if status["checkpoint_exists"]:
        assert status["asset_available"] is True
        if status["model_loaded"]:
            assert status["status_code"] == "OPERATIONAL"
        else:
            assert status["status_code"] == "ASSET_AVAILABLE_NOT_LOADED"
    else:
        assert "ASSET MISSING" in status["status"]


def test_vision_status_does_not_initialize_or_load_model(tmp_path):
    checkpoint = tmp_path / "available-but-not-loaded.pth"
    checkpoint.write_bytes(b"not loaded by status inspection")
    adapter = VisionAdapter(checkpoint_path=checkpoint)

    status = adapter.status_info()

    assert adapter._initialized is False
    assert adapter._model is None
    assert status["asset_available"] is True
    assert status["initialization_attempted"] is False
    assert status["model_loaded"] is False
    assert status["inference_ready"] is False
    assert status["status_code"] == "ASSET_AVAILABLE_NOT_LOADED"


def test_vision_adapter_missing_checkpoint_safety(make_image_bytes):
    """Verify VisionAdapter does not crash when weights are absent."""
    missing_adapter = VisionAdapter(checkpoint_path=Path("nonexistent/res18_ckpt.pth"))
    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "garment.jpg")

    import asyncio
    prediction = asyncio.run(missing_adapter.predict_material(processed))
    assert prediction.predicted_material is None
    assert prediction.confidence is None
    assert len(prediction.warnings) > 0
    assert "checkpoint not found" in prediction.warnings[0].lower()


def test_vision_missing_checkpoint_status_is_transparent(make_image_bytes):
    missing_adapter = VisionAdapter(checkpoint_path=Path("nonexistent/res18_ckpt.pth"))
    status = missing_adapter.status_info()
    assert status["checkpoint_exists"] is False
    assert status["model_loaded"] is False
    assert status["inference_ready"] is False
    assert status["status_code"] == "MODEL_CHECKPOINT_MISSING"


def test_vision_adapter_live_inference(make_image_bytes):
    """Verify VisionAdapter executes real forward pass with trained TextileNet ResNet-18 weights."""
    if not vision_adapter.is_available:
        pytest.skip("TextileNet checkpoint not available.")

    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "garment.jpg")

    import asyncio
    prediction = asyncio.run(vision_adapter.predict_material(processed))
    assert vision_adapter._initialized is True
    assert vision_adapter._model is not None
    assert prediction.predicted_material is not None
    assert prediction.confidence is not None
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.alternatives) > 0
    assert len(prediction.probabilities) > 0
    assert prediction.predicted_material in prediction.probabilities


def test_vision_adapter_injected_predictor(make_image_bytes):
    """Verify VisionAdapter accepts injected custom predictor double."""
    def fake_predictor(img):
        return VisionPrediction(
            predicted_material="cotton",
            confidence=0.91,
            alternatives=["linen", "hemp"],
            probabilities={"cotton": 0.91, "linen": 0.06, "hemp": 0.03},
            warnings=[],
        )

    test_adapter = VisionAdapter(custom_predictor=fake_predictor)
    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "shirt.jpg")

    import asyncio
    prediction = asyncio.run(test_adapter.predict_material(processed))
    assert prediction.predicted_material == "cotton"
    assert prediction.confidence == 0.91
    assert "linen" in prediction.alternatives


# ==============================================================================
# 4. Knowledge Base Adapter Tests (Ankit Contract)
# ==============================================================================

def test_knowledge_base_adapter_status():
    """Verify KnowledgeBaseAdapter reflects operational status and 33-class taxonomy."""
    status = knowledge_adapter.status_info()
    assert status["adapter"] == "Ankit-KnowledgeBase-Adapter"
    assert status["status"] in ["INTEGRATED & OPERATIONAL", "ADAPTER READY — MODULE PENDING"]
    assert "polyester" in status["available_materials"]
    assert "cotton" in status["available_materials"]
    assert "flax_linen" in status["available_materials"]
    assert status["taxonomy_classes_covered"] == 33


def test_knowledge_base_retrieval():
    """Verify knowledge base retrieves verified benchmark metrics."""
    import asyncio
    profile = asyncio.run(knowledge_adapter.get_material_data("cotton"))
    assert profile is not None
    assert profile.material == "cotton"
    assert profile.water_liters_per_kg is not None
    assert profile.water_liters_per_kg > 0
    assert profile.co2_kg_per_kg is not None
    assert profile.chemicals_intensity == "High"
    assert "verified" in profile.custom_indicators["provenance"].lower()


def test_knowledge_base_batch_retrieval():
    """Verify batch retrieval tracks unknown materials."""
    import asyncio
    batch = asyncio.run(knowledge_adapter.get_batch_data(["polyester", "wool", "unknown_fiber_xyz"]))
    assert "polyester" in batch.profiles
    assert "wool" in batch.profiles
    assert "unknown_fiber_xyz" in batch.unrecognized_materials


# ==============================================================================
# 5. Normalization Service Tests
# ==============================================================================

def test_canonicalize_material_names():
    """Verify alias mapping standardizes trade names and synonyms."""
    assert NormalizationService.canonicalize_material_name("Spandex") == "elastane"
    assert NormalizationService.canonicalize_material_name("Lycra") == "elastane"
    assert NormalizationService.canonicalize_material_name("POLY") == "polyester"
    assert NormalizationService.canonicalize_material_name("Rayon") == "viscose"
    assert NormalizationService.canonicalize_material_name("Flax") == "linen"
    assert NormalizationService.canonicalize_material_name("Organic Cotton") == "organic_cotton"


def test_normalize_ocr_composition():
    """Verify normalization of OCR result, merging duplicates and checking percentage sums."""
    ocr_result = OCRResult(
        composition=[
            OCRCompositionItem(material="COTTON", percentage=60.0),
            OCRCompositionItem(material="Ctn", percentage=10.0),
            OCRCompositionItem(material="POLYESTER", percentage=30.0),
        ],
        confidence=0.92,
        raw_text="60% COTTON 10% CTN 30% POLYESTER",
        unreadable=False,
    )

    comp, warnings = NormalizationService.normalize_ocr_composition(ocr_result)
    assert comp is not None
    assert len(comp.materials) == 2  # Cotton + Polyester
    cotton_mat = next(m for m in comp.materials if m.name == "cotton")
    assert cotton_mat.percentage == 70.0
    assert comp.is_complete is True


def test_synthesize_product_analysis_perception_conflict():
    """Verify that conflict between OCR and Vision is recorded as an explicit warning."""
    ocr_result = OCRResult(
        composition=[OCRCompositionItem(material="Wool", percentage=100.0)],
        confidence=0.95,
        raw_text="100% WOOL",
        unreadable=False,
    )
    vision_result = VisionPrediction(
        predicted_material="cotton",
        confidence=0.88,
        alternatives=["linen"],
    )

    product, warnings = NormalizationService.synthesize_product_analysis(
        ocr_result=ocr_result,
        vision_result=vision_result,
    )

    assert product.composition is not None
    assert product.composition_source == AnalysisSource.OCR_LABEL
    assert any("Perception Discrepancy" in w for w in warnings)


def test_concurrent_independent_streams_and_cleanup(make_image_bytes):
    """Verify OCR and Vision can process independent streams concurrently from ProcessedImage without corruption."""
    import asyncio

    class FakeReader:
        def readtext(self, img, detail=1, paragraph=False):
            return [([[0, 0]], "100% COTTON", 0.94)]

    mock_ocr = OCRAdapter(ocr_reader=FakeReader())

    def fake_vision(img):
        stream = img.get_stream()
        assert len(stream.read()) > 0
        return VisionPrediction(
            predicted_material="cotton",
            confidence=0.90,
            alternatives=["linen"],
        )

    mock_vision = VisionAdapter(custom_predictor=fake_vision)

    data = make_image_bytes(fmt="JPEG")
    processed = ImageService.process_image(data, "test.jpg")

    async def _run_parallel():
        return await asyncio.gather(
            mock_ocr.extract_composition(processed),
            mock_vision.predict_material(processed),
        )

    ocr_res, vision_res = asyncio.run(_run_parallel())
    assert ocr_res.composition[0].material == "Cotton"
    assert vision_res.predicted_material == "cotton"

    # Verify cleanup only after all operations finish
    assert len(processed.raw_bytes) > 0
    processed.cleanup()
    assert len(processed.raw_bytes) == 0


def test_partial_failure_ocr_fails_vision_succeeds():
    """Verify graceful synthesis when OCR fails but Vision succeeds (fallback composition)."""
    ocr_result = OCRResult(unreadable=True, composition=[], warnings=["Label unreadable"])
    vision_result = VisionPrediction(
        predicted_material="linen",
        confidence=0.85,
        alternatives=["hemp"],
    )

    product, warnings = NormalizationService.synthesize_product_analysis(
        ocr_result=ocr_result,
        vision_result=vision_result,
    )

    assert product.composition is not None
    assert product.composition.materials[0].name == "linen"
    assert product.composition_source == AnalysisSource.COMPUTER_VISION
    assert any("Fallback composition estimated from Computer Vision" in w for w in warnings)


def test_partial_failure_vision_fails_ocr_succeeds():
    """Verify graceful synthesis when Vision is unavailable but OCR succeeds."""
    ocr_result = OCRResult(
        composition=[OCRCompositionItem(material="Silk", percentage=100.0)],
        confidence=0.98,
        unreadable=False,
    )
    vision_result = VisionPrediction(
        predicted_material=None,
        confidence=None,
        warnings=["Checkpoint res18_ckpt.pth missing"],
    )

    product, warnings = NormalizationService.synthesize_product_analysis(
        ocr_result=ocr_result,
        vision_result=vision_result,
    )

    assert product.composition is not None
    assert product.composition.materials[0].name == "silk"
    assert product.composition_source == AnalysisSource.OCR_LABEL
    assert any("Checkpoint res18_ckpt.pth missing" in w for w in warnings)


# ==============================================================================
# 6. End-to-End Cross-Module Data Exchange Tests
# ==============================================================================

def test_cross_module_end_to_end_pipeline():
    """Verify full end-to-end data exchange across ImageService, OCR, Normalization, KB, and Scoring."""
    import asyncio

    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0002.jpg"
    if not fixture_path.exists():
        pytest.skip("Label fixture not found.")

    with open(fixture_path, "rb") as f:
        img_bytes = f.read()

    # Step 1: Process Image
    proc_img = ImageService.process_image(img_bytes, filename="label_0002.jpg")

    # Step 2: Live OCR
    ocr_res = asyncio.run(ocr_adapter.extract_composition(proc_img))
    assert ocr_res.unreadable is False

    # Step 3: Vision (safely warns asset missing)
    vis_res = asyncio.run(vision_adapter.predict_material(proc_img))

    # Step 4: Normalization & Synthesis
    product, warnings = normalization_service.synthesize_product_analysis(ocr_res, vis_res)
    assert product.composition is not None
    assert product.composition.is_complete is True

    # Step 5: Knowledge Base Retrieval
    mat_names = [m.name for m in product.composition.materials]
    kb_res = asyncio.run(knowledge_adapter.get_batch_data(mat_names))
    assert len(kb_res.profiles) == len(mat_names)

    # Step 6: Environmental Scoring
    blend_dict = {m.name: m.percentage for m in product.composition.materials}
    score_res = scoring_adapter.calculate_impact(blend_dict, shopping=True)
    assert score_res.overall_impact_score > 0
    assert score_res.recommendation in ["Buy", "Think Twice", "Repair", "Reuse", "Second-hand"]
    assert score_res.breakdown is not None


def test_reconciliation_both_succeed_and_agree():
    """Verify Scenario 1: Both OCR and Vision succeed and agree."""
    ocr_result = OCRResult(
        composition=[OCRCompositionItem(material="Cotton", percentage=100.0)],
        confidence=0.92,
        unreadable=False,
    )
    vision_result = VisionPrediction(
        predicted_material="cotton",
        confidence=0.88,
    )

    reconciled, source, status, warnings = NormalizationService.reconcile_findings(
        ocr_result=ocr_result,
        vision_result=vision_result,
    )

    assert status == "AGREED"
    assert source == AnalysisSource.HYBRID
    assert reconciled is not None
    assert reconciled.materials[0].name == "cotton"
    assert reconciled.confidence == 0.92


def test_reconciliation_both_fail():
    """Verify Scenario 4: Both OCR and Vision fail."""
    ocr_result = OCRResult(
        composition=[],
        unreadable=True,
        warnings=["Label unreadable."],
    )
    vision_result = VisionPrediction(
        predicted_material=None,
        confidence=None,
        warnings=["Model weights missing."],
    )

    reconciled, source, status, warnings = NormalizationService.reconcile_findings(
        ocr_result=ocr_result,
        vision_result=vision_result,
    )

    assert status == "BOTH_FAILED"
    assert source == AnalysisSource.UNKNOWN
    assert reconciled is None
    assert any("unable to determine garment fabric composition" in w for w in warnings)


def test_scoring_accepts_tolerance_boundary_without_renormalizing(monkeypatch):
    """The original proportions reach the scorer unchanged within ±0.01%."""
    imperfect_blend = {"cotton": 33.33, "polyester": 33.33, "wool": 33.33}
    received = []
    original_score = scoring_adapter._scorer.score

    def capture_and_score(composition):
        received.append(dict(composition))
        return original_score(composition)

    monkeypatch.setattr(scoring_adapter._scorer, "score", capture_and_score)
    result = scoring_adapter.calculate_impact(imperfect_blend)
    assert result.overall_impact_score > 0
    assert result.overall_impact_score <= 100
    assert received == [imperfect_blend]


@pytest.mark.parametrize(
    ("total", "accepted"),
    [
        (98.0, False),
        (99.0, False),
        (99.9, False),
        (100.0, True),
        (100.1, False),
        (101.0, False),
        (102.0, False),
        (99.99, True),
        (100.01, True),
    ],
)
def test_scoring_total_uses_absolute_point_zero_one_tolerance(total, accepted):
    composition = {"cotton": 50.0, "polyester": round(total - 50.0, 2)}
    if accepted:
        result = scoring_adapter.calculate_impact(composition)
        assert result.overall_impact_score is not None
    else:
        with pytest.raises(ScoringError) as exc:
            scoring_adapter.calculate_impact(composition)
        assert exc.value.status_code == 422


def test_ocr_parser_reversed_syntax_and_colons():
    """Verify parse_composition handles colons, hyphens, and reversed syntax."""
    from app.integrations.teammate_modules.ocr.text_parser import parse_composition

    text = "FABRIC: COTTON: 70%, POLYESTER - 30%, ELASTANE (5%)"
    parsed = parse_composition(text)
    assert "COTTON" in parsed
    assert parsed["COTTON"] == [70.0]
    assert "POLYESTER" in parsed
    assert parsed["POLYESTER"] == [30.0]
    assert "ELASTANE" in parsed
    assert parsed["ELASTANE"] == [5.0]


def test_vision_model_forward_architecture():
    """Verify ResNet-18 model architecture executes tensor forward pass with shape (1, 33)."""
    import torch
    from app.integrations.teammate_modules.vision.resnet import get_resnet18

    model = get_resnet18(num_classes=33)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 33)
    probs = torch.softmax(output, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-4


# ==============================================================================
# 7. Comprehensive Pipeline Scenarios (11 Core Integration Scenarios)
# ==============================================================================

def test_pipeline_scenario1_real_ocr_vision_kb_scoring(make_image_bytes):
    """Scenario 1: Real OCR + real Vision + Knowledge Base + Scoring end-to-end."""
    import asyncio
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0002.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture label_0002.jpg not found.")

    with open(fixture_path, "rb") as f:
        label_bytes = f.read()

    garment_bytes = make_image_bytes(fmt="JPEG")

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=label_bytes,
            label_filename="label_0002.jpg",
            garment_image_bytes=garment_bytes,
            garment_filename="garment.jpg",
            shopping=True,
        )
    )

    assert response.success is True
    assert response.analysis_id is not None
    assert response.product.composition is not None
    # label_0002.jpg is 68% Nylon, 32% Wool
    materials = {m.name: m.percentage for m in response.product.composition.materials}
    assert "nylon" in materials
    assert "wool" in materials
    # Vision was also executed
    assert response.product.visual_prediction is not None
    assert response.product.visual_prediction.predicted_material is not None
    # Knowledge Base retrieved data
    assert "nylon" in response.environmental_data
    assert "wool" in response.environmental_data
    # Scoring produced result
    assert response.impact_assessment.overall_impact_score > 0
    assert response.impact_assessment.recommendation in ["Buy", "Think Twice", "Repair", "Reuse", "Second-hand"]


def test_pipeline_scenario2_ocr_success_vision_failure(make_image_bytes):
    """Scenario 2: OCR succeeds, Vision fails (simulated missing checkpoint)."""
    import asyncio
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0003.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture label_0003.jpg not found.")

    with open(fixture_path, "rb") as f:
        label_bytes = f.read()

    # Mock Vision adapter with missing checkpoint
    failing_vision = VisionAdapter(checkpoint_path=Path("nonexistent/missing_ckpt.pth"))

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=label_bytes,
            label_filename="label_0003.jpg",
            garment_image_bytes=make_image_bytes(fmt="JPEG"),
            vision_adapter_instance=failing_vision,
        )
    )

    assert response.success is True
    assert response.product.composition is not None
    assert response.product.composition_source == AnalysisSource.OCR_LABEL
    assert any("linen" in m.name for m in response.product.composition.materials)
    # Checkpoint warning is preserved
    assert any("checkpoint not found" in w.lower() for w in response.warnings)


def test_pipeline_scenario3_vision_success_ocr_failure(make_image_bytes):
    """Scenario 3: Vision succeeds, OCR unreadable / corrupted label."""
    import asyncio
    # Provide corrupted label bytes to force OCR unreadable
    corrupted_label = b"CORRUPTED_LABEL_NOT_A_VALID_IMAGE"
    garment_bytes = make_image_bytes(fmt="JPEG")

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=corrupted_label,
            label_filename="corrupted_label.jpg",
            garment_image_bytes=garment_bytes,
            garment_filename="garment.jpg",
        )
    )

    # When OCR fails, pipeline falls back to Computer Vision
    assert response.product.composition is not None
    assert response.product.composition_source == AnalysisSource.COMPUTER_VISION
    assert len(response.product.composition.materials) == 1
    # Check that warning about fallback was issued
    assert any("fallback composition estimated from computer vision" in w.lower() for w in response.warnings)
    # Knowledge base and scoring still ran on the vision fallback
    assert response.impact_assessment.overall_impact_score > 0


def test_pipeline_scenario4_both_modules_agree(make_image_bytes):
    """Scenario 4: Both OCR and Vision agree (HYBRID source)."""
    import asyncio
    # Mock OCR and Vision returning cotton
    mock_ocr = OCRAdapter(
        custom_extractor=lambda img: OCRResult(
            composition=[OCRCompositionItem(material="Cotton", percentage=100.0)],
            confidence=0.95,
            unreadable=False,
        )
    )
    mock_vision = VisionAdapter(
        custom_predictor=lambda img: VisionPrediction(
            predicted_material="cotton",
            confidence=0.89,
            alternatives=["linen", "hemp"],
            probabilities={"cotton": 0.89, "linen": 0.08, "hemp": 0.03},
        )
    )

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            garment_image_bytes=make_image_bytes(fmt="JPEG"),
            ocr_adapter_instance=mock_ocr,
            vision_adapter_instance=mock_vision,
        )
    )

    assert response.success is True
    assert response.product.composition_source == AnalysisSource.HYBRID
    assert response.product.composition.materials[0].name == "cotton"
    assert response.confidence["ocr"] == 0.95
    assert response.confidence["vision"] == 0.89


def test_pipeline_scenario5_ocr_vision_discrepancy(make_image_bytes):
    """Scenario 5: High-confidence OCR and Vision discrepancy flagged."""
    import asyncio
    mock_ocr = OCRAdapter(
        custom_extractor=lambda img: OCRResult(
            composition=[OCRCompositionItem(material="Cotton", percentage=100.0)],
            confidence=0.95,
            unreadable=False,
        )
    )
    mock_vision = VisionAdapter(
        custom_predictor=lambda img: VisionPrediction(
            predicted_material="leather",
            confidence=0.88,
            alternatives=["suede", "fur"],
            probabilities={"leather": 0.88, "suede": 0.09, "fur": 0.03},
        )
    )

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            garment_image_bytes=make_image_bytes(fmt="JPEG"),
            ocr_adapter_instance=mock_ocr,
            vision_adapter_instance=mock_vision,
        )
    )

    assert response.success is True
    assert response.product.composition_source == AnalysisSource.OCR_LABEL
    assert any("perception discrepancy" in w.lower() for w in response.warnings)


def test_pipeline_scenario6_unknown_material(make_image_bytes):
    """Scenario 6: Unknown material does not silently receive zero impact."""
    import asyncio
    mock_ocr = OCRAdapter(
        custom_extractor=lambda img: OCRResult(
            composition=[OCRCompositionItem(material="KryptoniteFiber", percentage=100.0)],
            confidence=0.90,
            unreadable=False,
        )
    )

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            ocr_adapter_instance=mock_ocr,
        )
    )

    # KryptoniteFiber is unknown to both KB and Scoring
    assert any("kryptonitefiber" in w.lower() for w in response.warnings)
    # An invalid score remains unavailable instead of receiving a placeholder.
    assert response.impact_assessment.overall_impact_score is None
    assert response.impact_assessment.recommendation is None
    assert response.impact_assessment.key_indicators.get("status") == "INSUFFICIENT_DATA"


def test_pipeline_scenario7_insufficient_kb_coverage():
    """Scenario 7: Insufficient data coverage (<60%) or unknown material prevents misleading scoring."""
    from app.integrations.teammate_modules.scoring import EnvironmentalScorer
    scorer = EnvironmentalScorer()

    # Blend where material is completely missing from database
    blend_insufficient = {"polyester": 20.0, "unknown_space_fiber": 80.0}
    with pytest.raises(ValueError) as excinfo:
        scorer.score(blend_insufficient)
    err_msg = str(excinfo.value).lower()
    assert "unknown material" in err_msg or "data coverage" in err_msg or "missing" in err_msg


def test_pipeline_scenario8_invalid_image():
    """Scenario 8: Completely invalid/corrupted bytes handled gracefully without pipeline crash."""
    import asyncio
    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=b"INVALID_TRASH_BYTES_0000",
            garment_image_bytes=b"INVALID_TRASH_BYTES_1111",
        )
    )

    # Pipeline does not crash; produces structured response with warnings
    assert response.success is False
    assert len(response.warnings) > 0
    assert response.product.composition is None
    assert response.impact_assessment.key_indicators.get("status") == "NO_COMPOSITION_DATA"


def test_pipeline_scenario9_malformed_composition(make_image_bytes):
    """Scenario 9: Malformed composition (e.g. 50% total) triggers integrity warnings."""
    import asyncio
    mock_ocr = OCRAdapter(
        custom_extractor=lambda img: OCRResult(
            composition=[OCRCompositionItem(material="Cotton", percentage=50.0)],
            confidence=0.80,
            unreadable=False,
        )
    )

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            ocr_adapter_instance=mock_ocr,
        )
    )

    assert any("partial or incomplete" in w.lower() for w in response.warnings)


def test_pipeline_scenario10_module_exception_resilience(make_image_bytes):
    """Scenario 10: Unexpected module runtime exception handled gracefully."""
    import asyncio

    def crashing_extractor(img):
        raise RuntimeError("Simulated unexpected GPU failure")

    mock_crashing_ocr = OCRAdapter(custom_extractor=crashing_extractor)

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=make_image_bytes(fmt="JPEG"),
            ocr_adapter_instance=mock_crashing_ocr,
        )
    )

    # Exception is captured into structured warnings, pipeline does not blow up
    assert any("simulated unexpected gpu failure" in w.lower() for w in response.warnings)


def test_pipeline_scenario11_standardized_response_validation(make_image_bytes):
    """Scenario 11: Validate full AnalysisResponse schema conformity and metadata completeness."""
    import asyncio
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / "label_0001.jpg"
    if not fixture_path.exists():
        pytest.skip("Fixture label_0001.jpg not found.")

    with open(fixture_path, "rb") as f:
        label_bytes = f.read()

    response = asyncio.run(
        NormalizationService.execute_pipeline(
            label_image_bytes=label_bytes,
            label_filename="label_0001.jpg",
            garment_image_bytes=make_image_bytes(fmt="JPEG"),
            garment_filename="garment.jpg",
        )
    )

    # Validate all fields in Pydantic schema
    assert isinstance(response, AnalysisResponse)
    assert isinstance(response.analysis_id, str) and len(response.analysis_id) > 10
    assert isinstance(response.timestamp, str)
    assert response.metadata.processing_time_ms > 0
    assert response.metadata.pipeline_version == "1.0.0"
    assert "ocr" in response.metadata.model_versions
    assert "vision" in response.metadata.model_versions
    assert len(response.limitations) >= 2
    assert isinstance(response.confidence, dict)

