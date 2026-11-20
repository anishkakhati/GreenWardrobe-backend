"""Textile label OCR text extraction and composition parsing logic.

Imported from Harshit's Clothing Label OCR module (origin/feature/-clothing-label-ocr).
Enhanced with robust boundary matching for colons, hyphens, and reversed material orders.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np

# Fibres officially recognized and evaluated in Harshit's OCR module
# + standard apparel extensions
SUPPORTED_FIBRES: List[str] = [
    "COTTON", "POLYESTER", "VISCOSE", "ELASTANE",
    "NYLON", "WOOL", "ACRYLIC", "LINEN", "MODAL", "SILK",
    "HEMP", "CASHMERE", "LYOCELL", "CUPRO", "RAMIE", "SPANDEX", "RAYON", "FLAX"
]


def normalize_text(text: str) -> str:
    """Normalize extracted OCR text to uppercase and collapse extra whitespace."""
    text = str(text).upper()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_composition(text: str) -> Dict[str, List[float]]:
    """Parse material names and their numerical percentages from text.

    Supports multiple formats:
    - Standard: '80% COTTON' or '80 % COTTON' or '80% OF COTTON'
    - Reversed: 'COTTON 80%' or 'COTTON: 80%' or 'COTTON - 80%'
    - Parenthetical: 'COTTON (80%)'
    """
    normalized = normalize_text(text)
    composition: Dict[str, List[float]] = {}

    # Determine the label's local ordering from its first recognized fibre.
    # Without this, a forward-only regex such as `FIBRE 84%` can steal the
    # following fibre's percentage when OCR drops the preceding '%' glyph.
    fibre_token = re.compile(
        r"\b(?:" + "|".join(re.escape(name) for name in sorted(SUPPORTED_FIBRES, key=len, reverse=True)) + r")\b"
    )
    first_fibre = fibre_token.search(normalized)
    percentage_first = True
    if first_fibre:
        before = normalized[:first_fibre.start()]
        after = normalized[first_fibre.end():]
        preceding_value = re.search(r"(\d+(?:\.\d+)?)\s*%?\s*(?:OF\s+)?$", before)
        following_value = re.match(r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%", after)
        parenthetical_value = re.match(r"\s*\(\s*\d+(?:\.\d+)?\s*%\s*\)", after)
        if following_value or parenthetical_value:
            percentage_first = bool(preceding_value)

    for fibre in SUPPORTED_FIBRES:
        preceding_pattern = (
            # OCR often drops the percent glyph while retaining a number that
            # is immediately before its material (e.g. "16 ACRYLIC"). Prefer
            # this preceding-number form so it cannot consume the next fibre's
            # percentage through the reversed syntax below.
            rf"(\d+(?:\.\d+)?)\s*%?\s*(?:OF\s+)?{fibre}\b"
        )
        following_pattern = rf"\b{fibre}\s*[:\-\s]?\s*(\d+(?:\.\d+)?)\s*%"
        parenthetical_pattern = rf"\b{fibre}\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)"
        patterns = (
            [preceding_pattern, following_pattern, parenthetical_pattern]
            if percentage_first
            else [following_pattern, preceding_pattern, parenthetical_pattern]
        )

        for pattern in patterns:
            matches = re.findall(pattern, normalized)
            if matches:
                composition[fibre] = [float(value) for value in matches]
                break

    return composition


def preprocess_image_array(image_bgr: np.ndarray) -> np.ndarray:
    """Apply Harshit's exact OpenCV enhancement pipeline to an in-memory BGR array.

    Steps:
    1. Cubic interpolation scaling (1.5x)
    2. Grayscale conversion
    3. CLAHE contrast enhancement
    4. 3x3 Laplacian-style sharpening kernel
    """
    try:
        import cv2
    except ImportError:
        # Fallback if cv2 is not installed in environment
        return image_bgr

    # Resize
    resized = cv2.resize(image_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened
