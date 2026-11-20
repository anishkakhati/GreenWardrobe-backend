"""Textile label OCR text parser package from Harshit's OCR module."""

from app.integrations.teammate_modules.ocr.text_parser import (
    SUPPORTED_FIBRES,
    normalize_text,
    parse_composition,
    preprocess_image_array,
)

__all__ = [
    "SUPPORTED_FIBRES",
    "normalize_text",
    "parse_composition",
    "preprocess_image_array",
]
