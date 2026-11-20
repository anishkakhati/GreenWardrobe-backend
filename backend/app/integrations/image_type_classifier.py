"""Image-type classification interface and heuristic MVP classifier.

Determines whether an uploaded image is a:
  - GARMENT  — a photograph of fabric/clothing surface
  - LABEL    — a photograph of a clothing care/composition label
  - UNKNOWN  — cannot be classified with sufficient confidence

Design notes
------------
No dedicated trained classifier exists in the GreenWardrobe repository for
distinguishing garment photographs from label photographs. The ResNet-18
checkpoint (res18_ckpt.pth) is a 33-class TEXTILE FIBRE classifier trained on
fabric surface textures — it cannot reliably distinguish a garment surface from
a label photograph, which may show any fabric type in the background.

This module implements a **clearly separated, replaceable classifier interface**
(BaseImageTypeClassifier) and an **honest MVP heuristic classifier**
(HeuristicImageTypeClassifier) that uses pixel-level image statistics to make a
probabilistic determination:

Heuristic signals used
----------------------
1. Grayscale variance (std dev of luminance)
   - Labels contain high-contrast text on a plain background → low-to-moderate
     local variance but high global structure.
   - Garments show continuous textile texture patterns → moderate-to-high variance.

2. White/near-white pixel ratio
   - Labels are typically printed on white or light-coloured backgrounds → high
     proportion of bright pixels.
   - Garments are rarely uniformly bright.

3. High-frequency edge density (Laplacian variance)
   - Sharp printed text produces high-frequency edges concentrated in local
     regions on an otherwise smooth background.
   - Textile texture produces diffuse mid-frequency edges across the whole image.

4. Aspect ratio
   - Most care labels are portrait or near-square (height ≥ width).
   - Garment photographs are more often landscape or square.

These signals are combined into a weighted score. The classifier returns:
  - image_type: "GARMENT" | "LABEL" | "UNKNOWN"
  - confidence: float in [0.0, 1.0]
  - reasoning: list of human-readable signal descriptions

Replacement contract
--------------------
To swap this heuristic for a trained ML classifier, implement
BaseImageTypeClassifier and pass the new instance to the route handler or the
analysis service. No other code needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.core.logging import logger
from app.services.image_service import ProcessedImage


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

IMAGE_TYPE_GARMENT = "GARMENT"
IMAGE_TYPE_LABEL = "LABEL"
IMAGE_TYPE_UNKNOWN = "UNKNOWN"

# Default confidence threshold: below this, return UNKNOWN
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result returned by an image-type classifier."""

    image_type: str
    """One of IMAGE_TYPE_GARMENT, IMAGE_TYPE_LABEL, IMAGE_TYPE_UNKNOWN."""

    confidence: float
    """Confidence score in [0.0, 1.0]. Values below threshold → UNKNOWN."""

    classifier_name: str
    """Human-readable name of the classifier that produced this result."""

    reasoning: List[str] = field(default_factory=list)
    """Human-readable list of signals and their contributions."""

    is_above_threshold: bool = True
    """False when confidence < configured threshold and type is set to UNKNOWN."""


# ---------------------------------------------------------------------------
# Abstract interface (replacement contract)
# ---------------------------------------------------------------------------

class BaseImageTypeClassifier(ABC):
    """Abstract classifier for distinguishing garment from label images.

    Implement this to replace the heuristic MVP with a trained ML model.
    """

    @property
    @abstractmethod
    def classifier_name(self) -> str:
        """Human-readable name of the classifier."""

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence below which the result is returned as UNKNOWN."""

    @abstractmethod
    def classify(self, image: ProcessedImage) -> ClassificationResult:
        """Classify the image synchronously.

        Parameters
        ----------
        image:
            A validated ``ProcessedImage`` object.

        Returns
        -------
        ClassificationResult
            Contains image_type, confidence, reasoning, and threshold flag.
        """


# ---------------------------------------------------------------------------
# Heuristic MVP classifier
# ---------------------------------------------------------------------------

class HeuristicImageTypeClassifier(BaseImageTypeClassifier):
    """Pixel-statistics heuristic classifier for garment vs label images.

    This is NOT a trained ML classifier. It uses interpretable image statistics
    to produce a probabilistic determination. It is designed to be replaced by
    a proper trained model once labelled training data is available.

    See module docstring for the full signal description.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._threshold = confidence_threshold

    @property
    def classifier_name(self) -> str:
        return "HeuristicImageTypeClassifier-v1"

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    def classify(self, image: ProcessedImage) -> ClassificationResult:
        """Run pixel-statistics heuristic classification."""
        try:
            return self._run_heuristic(image)
        except Exception as exc:
            logger.warning("HeuristicImageTypeClassifier error: %s", exc)
            return ClassificationResult(
                image_type=IMAGE_TYPE_UNKNOWN,
                confidence=0.0,
                classifier_name=self.classifier_name,
                reasoning=[f"Classification error: {str(exc)}"],
                is_above_threshold=False,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_heuristic(self, image: ProcessedImage) -> ClassificationResult:
        pil = image.to_rgb_pil()
        arr = np.asarray(pil, dtype=np.float32)  # H x W x 3

        h, w = arr.shape[:2]
        reasoning: List[str] = []

        # Signal 1: grayscale (luminance) global standard deviation
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gray_std = float(np.std(gray))

        # Signal 2: white/near-white pixel ratio
        # A pixel is "light" if all channels > 200 (out of 255)
        light_mask = np.all(arr > 200, axis=2)
        light_ratio = float(light_mask.sum()) / (h * w)

        # Signal 3: Laplacian (edge) variance — proxy for text sharpness
        lap_var = self._laplacian_variance(gray)

        # Signal 4: aspect ratio — labels tend to be portrait or square
        aspect = w / h if h > 0 else 1.0  # > 1 → landscape; < 1 → portrait

        reasoning.append(
            f"Gray std={gray_std:.1f} (labels: low-moderate texture variation)"
        )
        reasoning.append(
            f"Light pixel ratio={light_ratio:.2f} (labels: high, garments: low)"
        )
        reasoning.append(
            f"Laplacian variance={lap_var:.1f} (text edges are sharp and concentrated)"
        )
        reasoning.append(
            f"Aspect ratio={aspect:.2f} (labels: ≤1.2, garments: varies)"
        )

        # Check for uniform / featureless image (e.g. solid color or flat gray)
        if gray_std < 10.0 and lap_var < 50.0:
            reasoning.append(
                f"Image is uniform/flat (gray_std={gray_std:.1f}, lap_var={lap_var:.1f}) → ambiguous"
            )
            return ClassificationResult(
                image_type=IMAGE_TYPE_UNKNOWN,
                confidence=0.30,
                classifier_name=self.classifier_name,
                reasoning=reasoning,
                is_above_threshold=False,
            )

        # -------------------------------------------------------------------
        # Score towards LABEL (1.0) or GARMENT (0.0)
        # -------------------------------------------------------------------
        label_score = 0.0
        total_weight = 0.0

        # Signal 1: Light pixel ratio (strongest indicator of white/light label ribbon)
        w1 = 4.0
        if light_ratio > 0.50:
            label_score += w1 * 1.0
        elif light_ratio > 0.25:
            label_score += w1 * ((light_ratio - 0.25) / 0.25)
        total_weight += w1

        # Signal 2: Edge sharpness (text characters on labels create sharp edges)
        w2 = 2.0
        if lap_var > 150.0:
            label_score += w2 * 0.8
        elif lap_var > 50.0:
            label_score += w2 * 0.4
        total_weight += w2

        # Signal 3: Aspect ratio (care labels tend to be portrait or square, ≤ 1.1)
        w3 = 1.0
        if aspect <= 1.1:
            label_score += w3 * 0.8
        elif aspect <= 1.4:
            label_score += w3 * 0.4
        total_weight += w3

        # Signal 4: Text contrast on light background
        w4 = 1.5
        if light_ratio > 0.30 and 25.0 <= gray_std <= 125.0:
            label_score += w4 * 1.0
        elif light_ratio <= 0.20:
            label_score += 0.0
        total_weight += w4

        # Normalize to [0, 1]
        raw_label_prob = label_score / total_weight if total_weight > 0 else 0.5

        # Map to confidence and type:
        # raw_label_prob > 0.5 → LABEL; < 0.5 → GARMENT
        if raw_label_prob >= 0.5:
            image_type = IMAGE_TYPE_LABEL
            confidence = 0.5 + (raw_label_prob - 0.5)
        else:
            image_type = IMAGE_TYPE_GARMENT
            confidence = 0.5 + (0.5 - raw_label_prob)

        # Clamp
        confidence = float(min(max(confidence, 0.0), 1.0))

        above = confidence >= self._threshold
        final_type = image_type if above else IMAGE_TYPE_UNKNOWN

        reasoning.append(
            f"Raw label probability={raw_label_prob:.3f} → type={image_type}, "
            f"confidence={confidence:.3f}, threshold={self._threshold}"
        )
        if not above:
            reasoning.append(
                "Confidence below threshold — returning UNKNOWN to avoid incorrect routing."
            )

        return ClassificationResult(
            image_type=final_type,
            confidence=round(confidence, 3),
            classifier_name=self.classifier_name,
            reasoning=reasoning,
            is_above_threshold=above,
        )

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Compute Laplacian variance of a 2D float array without cv2 dependency."""
        # 3×3 discrete Laplacian kernel
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # Manual 2D convolution on interior pixels only
        h, w = gray.shape
        if h < 3 or w < 3:
            return float(np.var(gray))
        # Pad with edge reflection
        padded = np.pad(gray, 1, mode="edge")
        lap = (
            padded[0:h, 1:w+1]
            + padded[2:h+2, 1:w+1]
            + padded[1:h+1, 0:w]
            + padded[1:h+1, 2:w+2]
            - 4 * padded[1:h+1, 1:w+1]
        )
        return float(np.var(lap))


# ---------------------------------------------------------------------------
# Global singleton — replace with a trained ML classifier here when ready
# ---------------------------------------------------------------------------

_default_classifier = HeuristicImageTypeClassifier(
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
)


def get_default_classifier() -> BaseImageTypeClassifier:
    """Return the global default image-type classifier instance."""
    return _default_classifier
