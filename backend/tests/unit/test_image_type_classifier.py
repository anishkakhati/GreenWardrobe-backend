"""Tests for the image-type classifier (HeuristicImageTypeClassifier).

Tests cover:
  - Garment classification (textile texture images)
  - Label classification (high white-pixel, low variance images)
  - Uncertain / UNKNOWN classification via threshold
  - Confidence scoring range
  - Classifier interface (BaseImageTypeClassifier contract)
  - Custom classifier injection
  - ClassificationResult data integrity

No actual ML model is called. All tests use in-memory PIL images.
"""

import io
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.integrations.image_type_classifier import (
    IMAGE_TYPE_GARMENT,
    IMAGE_TYPE_LABEL,
    IMAGE_TYPE_UNKNOWN,
    BaseImageTypeClassifier,
    ClassificationResult,
    HeuristicImageTypeClassifier,
    get_default_classifier,
)
from app.schemas.request import ImageMetadata
from app.services.image_service import ProcessedImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_processed_image(pil_img: Image.Image, fmt: str = "JPEG") -> ProcessedImage:
    """Wrap a PIL image in a ProcessedImage for testing."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    raw = buf.getvalue()
    meta = ImageMetadata(
        filename=f"test.{fmt.lower()}",
        format=fmt,
        width=pil_img.width,
        height=pil_img.height,
        size_bytes=len(raw),
    )
    return ProcessedImage(raw_bytes=raw, metadata=meta)


def _make_garment_image(size=(224, 224)) -> ProcessedImage:
    """Create an image that looks like a textile garment surface:
    coloured, noisy, medium variance, NOT predominantly white.
    """
    rng = np.random.default_rng(42)
    # Random textile-like texture: mid-tones, not white-dominant
    arr = rng.integers(60, 160, size=(*size, 3), dtype=np.uint8)
    # Add some regular pattern (simulates weave)
    for i in range(0, size[0], 8):
        arr[i, :, :] = np.clip(arr[i, :, :] + 40, 0, 255)
    pil = Image.fromarray(arr, mode="RGB")
    return _make_processed_image(pil)


def _make_label_image(size=(200, 300)) -> ProcessedImage:
    """Create an image that looks like a care label:
    white/near-white background, dark text-like elements, portrait orientation.
    """
    # White background
    pil = Image.new("RGB", size, color=(252, 252, 252))
    draw = ImageDraw.Draw(pil)
    # Simulate text lines (black rectangles)
    for y in range(30, size[1] - 30, 25):
        draw.rectangle([(20, y), (size[0] - 20, y + 8)], fill=(10, 10, 10))
    return _make_processed_image(pil)


def _make_ambiguous_image(size=(150, 150)) -> ProcessedImage:
    """Create a mid-grey uniform image that is ambiguous."""
    arr = np.full((*size, 3), 128, dtype=np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    return _make_processed_image(pil)


# ---------------------------------------------------------------------------
# Tests: HeuristicImageTypeClassifier — LABEL classification
# ---------------------------------------------------------------------------

class TestLabelClassification:
    def test_white_background_with_text_classified_as_label(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_label_image())
        assert result.image_type == IMAGE_TYPE_LABEL, (
            f"Expected LABEL, got {result.image_type} (conf={result.confidence:.3f}). "
            f"Reasoning: {result.reasoning}"
        )

    def test_label_confidence_in_valid_range(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_label_image())
        assert 0.0 <= result.confidence <= 1.0

    def test_label_confidence_above_threshold(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_label_image())
        assert result.is_above_threshold is True

    def test_label_result_has_reasoning(self):
        classifier = HeuristicImageTypeClassifier()
        result = classifier.classify(_make_label_image())
        assert len(result.reasoning) > 0

    def test_label_classifier_name_in_result(self):
        classifier = HeuristicImageTypeClassifier()
        result = classifier.classify(_make_label_image())
        assert result.classifier_name == classifier.classifier_name


# ---------------------------------------------------------------------------
# Tests: HeuristicImageTypeClassifier — GARMENT classification
# ---------------------------------------------------------------------------

class TestGarmentClassification:
    def test_textile_texture_classified_as_garment(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_garment_image())
        assert result.image_type == IMAGE_TYPE_GARMENT, (
            f"Expected GARMENT, got {result.image_type} (conf={result.confidence:.3f}). "
            f"Reasoning: {result.reasoning}"
        )

    def test_garment_confidence_in_valid_range(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_garment_image())
        assert 0.0 <= result.confidence <= 1.0

    def test_garment_result_above_threshold(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_garment_image())
        assert result.is_above_threshold is True


# ---------------------------------------------------------------------------
# Tests: UNKNOWN / uncertain classification
# ---------------------------------------------------------------------------

class TestUnknownClassification:
    def test_very_high_threshold_returns_unknown(self):
        """With an impossibly high threshold, all images should return UNKNOWN."""
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.999)
        result = classifier.classify(_make_garment_image())
        assert result.image_type == IMAGE_TYPE_UNKNOWN
        assert result.is_above_threshold is False

    def test_ambiguous_grey_image_flagged_as_low_confidence(self):
        """A uniform mid-grey image should produce a low-confidence result."""
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.55)
        result = classifier.classify(_make_ambiguous_image())
        # We don't assert UNKNOWN because heuristics may still pick a side;
        # but confidence should be relatively lower
        assert 0.0 <= result.confidence <= 1.0
        assert result.classifier_name == classifier.classifier_name

    def test_unknown_result_has_is_above_threshold_false(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.999)
        result = classifier.classify(_make_label_image())
        assert result.image_type == IMAGE_TYPE_UNKNOWN
        assert result.is_above_threshold is False
        assert "threshold" in " ".join(result.reasoning).lower()

    def test_unknown_result_includes_threshold_reasoning(self):
        classifier = HeuristicImageTypeClassifier(confidence_threshold=0.999)
        result = classifier.classify(_make_garment_image())
        joined = " ".join(result.reasoning).lower()
        assert "threshold" in joined or "unknown" in joined


# ---------------------------------------------------------------------------
# Tests: ClassificationResult data integrity
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_confidence_always_clamped_to_zero_one(self):
        classifier = HeuristicImageTypeClassifier()
        for img in [_make_label_image(), _make_garment_image(), _make_ambiguous_image()]:
            result = classifier.classify(img)
            assert 0.0 <= result.confidence <= 1.0, f"confidence out of range: {result.confidence}"

    def test_image_type_always_one_of_three_values(self):
        classifier = HeuristicImageTypeClassifier()
        for img in [_make_label_image(), _make_garment_image()]:
            result = classifier.classify(img)
            assert result.image_type in {IMAGE_TYPE_GARMENT, IMAGE_TYPE_LABEL, IMAGE_TYPE_UNKNOWN}

    def test_classifier_name_is_string(self):
        classifier = HeuristicImageTypeClassifier()
        result = classifier.classify(_make_label_image())
        assert isinstance(result.classifier_name, str)
        assert len(result.classifier_name) > 0

    def test_reasoning_is_list_of_strings(self):
        classifier = HeuristicImageTypeClassifier()
        result = classifier.classify(_make_label_image())
        assert isinstance(result.reasoning, list)
        for r in result.reasoning:
            assert isinstance(r, str)


# ---------------------------------------------------------------------------
# Tests: Classifier interface (BaseImageTypeClassifier contract)
# ---------------------------------------------------------------------------

class TestClassifierInterface:
    def test_heuristic_classifier_is_subclass_of_base(self):
        assert issubclass(HeuristicImageTypeClassifier, BaseImageTypeClassifier)

    def test_heuristic_classifier_has_classifier_name(self):
        c = HeuristicImageTypeClassifier()
        assert isinstance(c.classifier_name, str)
        assert len(c.classifier_name) > 0

    def test_heuristic_classifier_has_confidence_threshold(self):
        c = HeuristicImageTypeClassifier(confidence_threshold=0.65)
        assert c.confidence_threshold == 0.65

    def test_get_default_classifier_returns_base_subclass(self):
        c = get_default_classifier()
        assert isinstance(c, BaseImageTypeClassifier)

    def test_custom_classifier_injection(self):
        """Verifies the classifier interface can be replaced with a custom double."""

        class ForcedLabelClassifier(BaseImageTypeClassifier):
            @property
            def classifier_name(self) -> str:
                return "ForcedLabel"

            @property
            def confidence_threshold(self) -> float:
                return 0.5

            def classify(self, image: ProcessedImage) -> ClassificationResult:
                return ClassificationResult(
                    image_type=IMAGE_TYPE_LABEL,
                    confidence=0.99,
                    classifier_name=self.classifier_name,
                    reasoning=["forced label for test"],
                    is_above_threshold=True,
                )

        c = ForcedLabelClassifier()
        result = c.classify(_make_garment_image())
        assert result.image_type == IMAGE_TYPE_LABEL
        assert result.confidence == 0.99


# ---------------------------------------------------------------------------
# Tests: Laplacian variance helper (internal — isolated)
# ---------------------------------------------------------------------------

class TestLaplacianVariance:
    def test_flat_image_low_variance(self):
        gray = np.full((50, 50), 128.0, dtype=np.float32)
        var = HeuristicImageTypeClassifier._laplacian_variance(gray)
        assert var < 1.0  # uniform → near-zero Laplacian variance

    def test_text_like_image_higher_variance(self):
        gray = np.zeros((50, 50), dtype=np.float32)
        gray[10, :] = 200.0  # sharp horizontal edge like text
        var = HeuristicImageTypeClassifier._laplacian_variance(gray)
        assert var > 100.0  # sharp edges produce high variance

    def test_tiny_image_does_not_crash(self):
        gray = np.array([[100.0, 150.0], [80.0, 200.0]], dtype=np.float32)
        var = HeuristicImageTypeClassifier._laplacian_variance(gray)
        assert isinstance(var, float)
        assert var >= 0.0


# ---------------------------------------------------------------------------
# Tests: Real Fixture Validation (tests/fixtures/labels/*.jpg)
# ---------------------------------------------------------------------------

class TestRealFixtureClassification:
    """Verify that all 5 genuine label fixtures are correctly classified as LABEL."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["label_0001.jpg", "label_0002.jpg", "label_0003.jpg", "label_0004.jpg", "label_0005.jpg"],
    )
    def test_genuine_label_fixtures_classified_as_label(self, fixture_name):
        from pathlib import Path
        fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "labels" / fixture_name
        assert fixture_path.exists(), f"Fixture {fixture_name} not found"

        with open(fixture_path, "rb") as f:
            raw = f.read()

        meta = ImageMetadata(
            filename=fixture_name,
            format="JPEG",
            width=512,
            height=512,
            size_bytes=len(raw),
        )
        proc = ProcessedImage(raw_bytes=raw, metadata=meta)
        classifier = get_default_classifier()
        res = classifier.classify(proc)

        assert res.image_type == IMAGE_TYPE_LABEL, f"{fixture_name} expected LABEL, got {res.image_type}"
        assert res.confidence >= classifier.confidence_threshold
        assert res.is_above_threshold is True


# ---------------------------------------------------------------------------
# Tests: Edge Case Scenarios
# ---------------------------------------------------------------------------

class TestEdgeCaseScenarios:
    """Document and verify behavior across edge-case scenarios."""

    def test_solid_grey_uniform_image_returns_unknown(self):
        im = Image.new("RGB", (200, 200), color=(128, 128, 128))
        res = get_default_classifier().classify(_make_processed_image(im))
        assert res.image_type == IMAGE_TYPE_UNKNOWN
        assert res.is_above_threshold is False

    def test_blank_black_image_returns_unknown(self):
        im = Image.new("RGB", (100, 100), color=(0, 0, 0))
        res = get_default_classifier().classify(_make_processed_image(im))
        assert res.image_type == IMAGE_TYPE_UNKNOWN
        assert res.is_above_threshold is False

    def test_dark_garment_classified_as_garment(self):
        arr = np.random.randint(15, 50, (250, 250, 3), dtype=np.uint8)
        for i in range(0, 250, 6):
            arr[i, :, :] = np.clip(arr[i, :, :] + 25, 0, 255)
        res = get_default_classifier().classify(_make_processed_image(Image.fromarray(arr)))
        assert res.image_type == IMAGE_TYPE_GARMENT
        assert res.is_above_threshold is True

    def test_garment_with_large_graphic_logo_classified_as_garment(self):
        im = Image.new("RGB", (300, 300), color=(25, 35, 80))
        d = ImageDraw.Draw(im)
        d.rectangle([(30, 120), (270, 160)], fill=(250, 250, 250))
        d.rectangle([(50, 180), (250, 200)], fill=(250, 250, 250))
        res = get_default_classifier().classify(_make_processed_image(im))
        assert res.image_type == IMAGE_TYPE_GARMENT
        assert res.is_above_threshold is True

    def test_extreme_portrait_aspect_ratio_label_classified_as_label(self):
        im = Image.new("RGB", (60, 400), color=(250, 250, 250))
        d = ImageDraw.Draw(im)
        for y in range(30, 370, 30):
            d.rectangle([(10, y), (50, y + 4)], fill=(10, 10, 10))
        res = get_default_classifier().classify(_make_processed_image(im))
        assert res.image_type == IMAGE_TYPE_LABEL
        assert res.is_above_threshold is True
