"""Main Analysis Orchestration Service for GreenWardrobe.

Phase 7 refactor: accepts ONE image, classifies it as GARMENT or LABEL,
routes it to the correct pipeline branch, then converges into the shared
analysis pipeline.

Pipeline flow
-------------
Single UploadFile
    ↓
ImageService.validate_and_process_upload()     ← validation, security
    ↓
ImageTypeClassifier.classify()                 ← GARMENT | LABEL | UNKNOWN
    ↓
  LABEL → OCR branch   │   GARMENT → Vision branch   │   UNKNOWN → reject
    ↓                   │       ↓                      │
normalization_service.execute_pipeline()       ← converges both paths
    ↓
GeminiService.explain_analysis()               ← optional explanation
    ↓
AnalysisResponse
"""

from typing import Any, Optional

from fastapi import UploadFile

from app.core.exceptions import ImageClassificationError, ImageValidationError
from app.core.logging import logger
from app.integrations.image_type_classifier import (
    IMAGE_TYPE_GARMENT,
    IMAGE_TYPE_LABEL,
    IMAGE_TYPE_UNKNOWN,
    BaseImageTypeClassifier,
    ClassificationResult,
    get_default_classifier,
)
from app.schemas.response import AnalysisResponse, ImageClassificationMetadata
from app.services.gemini_service import GeminiService, gemini_service
from app.services.image_service import ProcessedImage, image_service
from app.services.normalization_service import normalization_service


class AnalysisService:
    """Orchestrates single-image garment environmental analysis and AI explanation."""

    def __init__(
        self,
        gemini_svc: Optional[GeminiService] = None,
        classifier: Optional[BaseImageTypeClassifier] = None,
    ):
        self._gemini_svc = gemini_svc or gemini_service
        self._classifier = classifier or get_default_classifier()

    # ------------------------------------------------------------------
    # Public entry point (new single-image contract)
    # ------------------------------------------------------------------

    async def analyze_image(
        self,
        image_file: UploadFile,
        *,
        shopping: bool = True,
        second_hand_available: bool = False,
        condition: str = "new",
        # Optional adapter overrides for testing
        ocr_adapter_instance: Optional[Any] = None,
        vision_adapter_instance: Optional[Any] = None,
        knowledge_adapter_instance: Optional[Any] = None,
        scoring_adapter_instance: Optional[Any] = None,
        gemini_service_instance: Optional[Any] = None,
        classifier_instance: Optional[BaseImageTypeClassifier] = None,
    ) -> AnalysisResponse:
        """Validate, classify, route, and analyse a single uploaded image.

        Parameters
        ----------
        image_file:
            The single uploaded image (JPEG / PNG / WEBP, ≤ 10 MB).
        shopping:
            True if the user is evaluating the item before purchase.
        second_hand_available:
            True if second-hand alternatives are known to be available.
        condition:
            Garment condition: ``"new"``, ``"used"``, or ``"damaged"``.
        """
        if not image_file:
            raise ImageValidationError(
                "Missing required image field.", status_code=400
            )

        proc_image: Optional[ProcessedImage] = None

        try:
            # ── Step 1: Validate & process upload ──────────────────────────
            proc_image = await image_service.validate_and_process_upload(image_file)

            # ── Step 2: Classify image type ────────────────────────────────
            classifier = classifier_instance or self._classifier
            classification: ClassificationResult = classifier.classify(proc_image)

            logger.info(
                "Image '%s' classified as %s (confidence=%.3f, classifier=%s)",
                proc_image.metadata.filename,
                classification.image_type,
                classification.confidence,
                classification.classifier_name,
            )

            # ── Step 3: Reject UNKNOWN images ─────────────────────────────
            if classification.image_type == IMAGE_TYPE_UNKNOWN:
                raise ImageClassificationError(
                    message=(
                        "The uploaded image could not be confidently classified as a garment photograph "
                        "or a clothing label photograph (confidence="
                        f"{classification.confidence:.2f}, threshold="
                        f"{classifier.confidence_threshold:.2f}). "
                        "Please upload a clear, well-lit photograph of either the garment fabric surface "
                        "or the inner care/composition label."
                    ),
                    status_code=422,
                    details={
                        "image_type": IMAGE_TYPE_UNKNOWN,
                        "confidence": classification.confidence,
                        "threshold": classifier.confidence_threshold,
                        "classifier": classification.classifier_name,
                        "reasoning": classification.reasoning,
                    },
                )

            # ── Step 4: Route to the correct pipeline branch ───────────────
            image_bytes = proc_image.raw_bytes
            filename = proc_image.metadata.filename

            label_bytes: Optional[bytes] = None
            garment_bytes: Optional[bytes] = None
            label_filename: str = "label.jpg"
            garment_filename: str = "garment.jpg"

            if classification.image_type == IMAGE_TYPE_LABEL:
                label_bytes = image_bytes
                label_filename = filename
                analysis_source = "OCR"
            else:  # IMAGE_TYPE_GARMENT
                garment_bytes = image_bytes
                garment_filename = filename
                analysis_source = "VISION"

            # ── Step 5: Run shared analysis pipeline ───────────────────────
            pipeline_response: AnalysisResponse = (
                await normalization_service.execute_pipeline(
                    label_image_bytes=label_bytes,
                    label_filename=label_filename,
                    garment_image_bytes=garment_bytes,
                    garment_filename=garment_filename,
                    shopping=shopping,
                    second_hand_available=second_hand_available,
                    condition=condition,
                    ocr_adapter_instance=ocr_adapter_instance,
                    vision_adapter_instance=vision_adapter_instance,
                    knowledge_adapter_instance=knowledge_adapter_instance,
                    scoring_adapter_instance=scoring_adapter_instance,
                )
            )

        finally:
            # Guaranteed cleanup
            if proc_image is not None:
                try:
                    proc_image.cleanup()
                except Exception as e:
                    logger.debug("Error during image cleanup: %s", e)

        # ── Step 6: Attach image-classification metadata to response ───────
        pipeline_response.image_classification = ImageClassificationMetadata(
            image_type=classification.image_type,
            classification_confidence=classification.confidence,
            classifier_name=classification.classifier_name,
            analysis_source=analysis_source,
            classification_reasoning=classification.reasoning,
        )

        # ── Step 7: Gemini explanation layer ───────────────────────────────
        gem_svc = gemini_service_instance or self._gemini_svc
        try:
            explanation = await gem_svc.explain_analysis(
                product=pipeline_response.product,
                environmental_data=pipeline_response.environmental_data,
                impact_assessment=pipeline_response.impact_assessment,
                confidence=pipeline_response.confidence,
                warnings=pipeline_response.warnings,
                limitations=pipeline_response.limitations,
            )
        except Exception as e:
            logger.warning("Gemini explanation layer failed: %s", e)
            explanation = gem_svc.generate_fallback_explanation(
                analysis_data={
                    "product": pipeline_response.product.model_dump(mode="json"),
                    "environmental_data": {
                        k: v.model_dump(mode="json")
                        for k, v in pipeline_response.environmental_data.items()
                    },
                    "impact_assessment": pipeline_response.impact_assessment.model_dump(
                        mode="json"
                    ),
                    "warnings": pipeline_response.warnings,
                    "limitations": pipeline_response.limitations,
                },
                reason=f"Explanation layer error: {str(e)}",
                status="UNAVAILABLE",
            )

        # ── Step 8: Attach explanation & update metadata ───────────────────
        pipeline_response.ai_explanation = explanation
        pipeline_response.metadata.gemini_model = gem_svc.adapter.model_name
        pipeline_response.metadata.explanation_status = explanation.status

        return pipeline_response

    # ------------------------------------------------------------------
    # Legacy two-image entry point (preserved for backward compatibility
    # and existing test coverage — delegates to the shared pipeline directly)
    # ------------------------------------------------------------------

    async def analyze_garment(
        self,
        label_file: UploadFile,
        garment_file: UploadFile,
        *,
        shopping: bool = True,
        second_hand_available: bool = False,
        condition: str = "new",
        ocr_adapter_instance: Optional[Any] = None,
        vision_adapter_instance: Optional[Any] = None,
        knowledge_adapter_instance: Optional[Any] = None,
        scoring_adapter_instance: Optional[Any] = None,
        gemini_service_instance: Optional[Any] = None,
    ) -> AnalysisResponse:
        """Legacy two-image entry point preserved for existing test coverage.

        .. deprecated::
            Use :meth:`analyze_image` for the new single-image workflow.
            This method is maintained so that all 122 existing tests continue
            to pass without modification.
        """
        if not label_file:
            raise ImageValidationError(
                "Missing required label_image file.", status_code=400
            )
        if not garment_file:
            raise ImageValidationError(
                "Missing required garment_image file.", status_code=400
            )

        proc_label: Optional[ProcessedImage] = None
        proc_garment: Optional[ProcessedImage] = None

        try:
            proc_label = await image_service.validate_and_process_upload(label_file)
            proc_garment = await image_service.validate_and_process_upload(garment_file)

            label_bytes = proc_label.raw_bytes
            garment_bytes = proc_garment.raw_bytes
            label_filename = proc_label.metadata.filename
            garment_filename = proc_garment.metadata.filename

            pipeline_response: AnalysisResponse = (
                await normalization_service.execute_pipeline(
                    label_image_bytes=label_bytes,
                    label_filename=label_filename,
                    garment_image_bytes=garment_bytes,
                    garment_filename=garment_filename,
                    shopping=shopping,
                    second_hand_available=second_hand_available,
                    condition=condition,
                    ocr_adapter_instance=ocr_adapter_instance,
                    vision_adapter_instance=vision_adapter_instance,
                    knowledge_adapter_instance=knowledge_adapter_instance,
                    scoring_adapter_instance=scoring_adapter_instance,
                )
            )
        finally:
            if proc_label is not None:
                try:
                    proc_label.cleanup()
                except Exception as e:
                    logger.debug("Error during proc_label cleanup: %s", e)
            if proc_garment is not None:
                try:
                    proc_garment.cleanup()
                except Exception as e:
                    logger.debug("Error during proc_garment cleanup: %s", e)

        gem_svc = gemini_service_instance or self._gemini_svc
        try:
            explanation = await gem_svc.explain_analysis(
                product=pipeline_response.product,
                environmental_data=pipeline_response.environmental_data,
                impact_assessment=pipeline_response.impact_assessment,
                confidence=pipeline_response.confidence,
                warnings=pipeline_response.warnings,
                limitations=pipeline_response.limitations,
            )
        except Exception as e:
            logger.warning("Gemini explanation layer failed: %s", e)
            explanation = gem_svc.generate_fallback_explanation(
                analysis_data={
                    "product": pipeline_response.product.model_dump(mode="json"),
                    "environmental_data": {
                        k: v.model_dump(mode="json")
                        for k, v in pipeline_response.environmental_data.items()
                    },
                    "impact_assessment": pipeline_response.impact_assessment.model_dump(
                        mode="json"
                    ),
                    "warnings": pipeline_response.warnings,
                    "limitations": pipeline_response.limitations,
                },
                reason=f"Explanation layer error: {str(e)}",
                status="UNAVAILABLE",
            )

        pipeline_response.ai_explanation = explanation
        pipeline_response.metadata.gemini_model = gem_svc.adapter.model_name
        pipeline_response.metadata.explanation_status = explanation.status

        return pipeline_response


# Global default analysis service instance
analysis_service = AnalysisService()
