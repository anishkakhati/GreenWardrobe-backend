"""OCR Adapter integrating Harshit Upadhyay's Clothing Label OCR module.

Connects the actual composition parsing logic, OpenCV image preprocessing,
and EasyOCR engine from origin/feature/-clothing-label-ocr.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from app.core.exceptions import OCRProcessingError
from app.core.logging import logger
from app.integrations.base import BaseOCRAdapter
from app.integrations.teammate_modules.ocr.text_parser import (
    SUPPORTED_FIBRES,
    normalize_text,
    parse_composition,
    preprocess_image_array,
)
from app.schemas.response import OCRCompositionItem, OCRResult
from app.services.image_service import ProcessedImage


class OCRAdapter(BaseOCRAdapter):
    """Adapter wrapping Harshit's Clothing Label OCR model and parser."""

    def __init__(
        self,
        ocr_reader: Optional[Any] = None,
        custom_parser: Optional[Callable[[str], Dict[str, List[float]]]] = None,
        custom_extractor: Optional[Callable[[ProcessedImage], OCRResult]] = None,
    ):
        self._reader = ocr_reader
        self._parser = custom_parser or parse_composition
        self._custom_extractor = custom_extractor
        self._easyocr_available = False
        self._initialized = False

    def _ensure_reader(self) -> None:
        """Lazy-load the EasyOCR reader on first execution to avoid slow app startup."""
        if self._initialized:
            return

        if self._reader is not None:
            self._easyocr_available = True
            self._initialized = True
            return

        try:
            import easyocr
            logger.info("Initializing EasyOCR reader for English care labels (CPU/auto)...")
            self._reader = easyocr.Reader(["en"], gpu=False)
            self._easyocr_available = True
        except ImportError:
            logger.warning("EasyOCR package is not installed. OCR will run in parser-only mode.")
            self._easyocr_available = False
        except Exception as e:
            logger.warning("EasyOCR model initialization failed: %s", str(e))
            self._easyocr_available = False
        finally:
            self._initialized = True

    @property
    def adapter_name(self) -> str:
        return "Harshit-ClothingLabelOCR-Adapter"

    @property
    def is_available(self) -> bool:
        if self._custom_extractor is not None:
            return True
        self._ensure_reader()
        return self._easyocr_available or self._reader is not None

    def status_info(self) -> Dict[str, Any]:
        self._ensure_reader()
        status_label = (
            "REAL INTEGRATION VERIFIED"
            if self._easyocr_available
            else "CODE CONNECTED BUT NOT INFERENCE VERIFIED"
        )
        return {
            "adapter": self.adapter_name,
            "status": status_label,
            "engine": "EasyOCR + Regex Composition Parser",
            "supported_fibres": SUPPORTED_FIBRES,
            "easyocr_installed": self._easyocr_available,
        }

    def _sync_extract(self, image: ProcessedImage) -> OCRResult:
        """Synchronous CPU worker executing OpenCV preprocessing and OCR extraction."""
        self._ensure_reader()

        if self._custom_extractor is not None:
            return self._custom_extractor(image)

        # If EasyOCR is unavailable, return an unreadable result with transparent diagnostic warning
        if not self.is_available or self._reader is None:
            return OCRResult(
                composition=[],
                confidence=None,
                raw_text="",
                unreadable=True,
                warnings=[
                    "OCR engine (EasyOCR) is not available in the runtime environment. Label extraction skipped."
                ],
            )

        try:
            # 1. Convert ProcessedImage to numpy array for OpenCV preprocessing
            pil_img = image.to_pil()
            img_array = np.array(pil_img)

            # Convert RGB to BGR if needed by OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                try:
                    import cv2
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                except ImportError:
                    img_bgr = img_array
            else:
                img_bgr = img_array

            # 2. Apply Harshit's exact OpenCV enhancement pipeline
            enhanced_image = preprocess_image_array(img_bgr)

            # 3. Execute EasyOCR text detection
            results = self._reader.readtext(enhanced_image, detail=1, paragraph=False)

            # 4. Sort regions top-to-bottom, left-to-right (per Harshit's logic)
            results.sort(
                key=lambda item: (
                    min(point[1] for point in item[0]),
                    min(point[0] for point in item[0]),
                )
            )

            raw_text = " ".join(item[1] for item in results)
            confidences = [float(item[2]) for item in results if len(item) > 2]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            # 5. Parse textile composition using Harshit's regex
            parsed = self._parser(raw_text)

            composition_items: List[OCRCompositionItem] = []
            for fibre_name, percentages in parsed.items():
                # Use first detected percentage if multiple matches
                pct = percentages[0] if percentages else 0.0
                composition_items.append(
                    OCRCompositionItem(
                        material=fibre_name.title(),
                        percentage=float(pct),
                    )
                )

            unreadable = len(composition_items) == 0

            warnings = []
            if unreadable:
                warnings.append("No recognized textile composition percentages found on label.")

            return OCRResult(
                composition=composition_items,
                confidence=round(avg_confidence, 2) if avg_confidence > 0 else None,
                raw_text=raw_text,
                unreadable=unreadable,
                warnings=warnings,
            )

        except Exception as e:
            logger.exception("Error during OCR execution: %s", str(e))
            raise OCRProcessingError(
                message=f"OCR execution failed: {str(e)}",
                status_code=422,
            )

    async def extract_composition(
        self, image: ProcessedImage, **kwargs: Any
    ) -> OCRResult:
        """Asynchronously extract textile composition by offloading CPU inference to worker thread."""
        return await asyncio.to_thread(self._sync_extract, image)


# Global default OCR adapter instance
ocr_adapter = OCRAdapter()
