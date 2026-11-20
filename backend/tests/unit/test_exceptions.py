"""Unit tests for domain exceptions."""

from app.core.exceptions import (
    GreenWardrobeException,
    ImageValidationError,
    OCRProcessingError,
    VisionProcessingError,
    KnowledgeBaseError,
    ScoringError,
    GeminiServiceError,
)


def test_domain_exceptions_status_codes():
    """Verify that domain exceptions carry the expected default HTTP status codes."""
    exc_image = ImageValidationError("Image too large", status_code=413)
    assert exc_image.status_code == 413
    assert exc_image.message == "Image too large"

    exc_ocr = OCRProcessingError("OCR failed")
    assert exc_ocr.status_code == 422

    exc_vision = VisionProcessingError("Vision failed")
    assert exc_vision.status_code == 422

    exc_kb = KnowledgeBaseError("Material not found")
    assert exc_kb.status_code == 404

    exc_scoring = ScoringError("Scoring error")
    assert exc_scoring.status_code == 422

    exc_gemini = GeminiServiceError("Gemini rate limited")
    assert exc_gemini.status_code == 502
