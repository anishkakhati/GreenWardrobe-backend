"""Textile composition normalization and cross-module synthesis service.

Normalizes material nomenclature across OCR, Computer Vision, and Knowledge Base.
Validates blend percentage sums, checks confidence thresholds, and implements
an explicit 5-scenario reconciliation strategy between label OCR and surface
computer vision predictions without silently mutating uncertain findings.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.composition import (
    COMPOSITION_TOTAL_TOLERANCE_PERCENT,
    composition_total_is_valid,
)
from app.core.logging import logger
from app.schemas.common import AnalysisSource
from app.schemas.response import (
    AnalysisMetadata,
    AnalysisResponse,
    Composition,
    Material,
    OCRResult,
    ProductAnalysis,
    ScoringResult,
    VisionPrediction,
)

# Standardized material alias dictionary
# Maps known alternative spellings, acronyms, trade names, and plurals to canonical keys
_TEXTILE_ALIASES: Dict[str, str] = {
    # Cotton variants
    "cotton": "cotton",
    "organic cotton": "organic_cotton",
    "organic_cotton": "organic_cotton",
    "ctn": "cotton",
    "coton": "cotton",
    "denim": "cotton",
    # Synthetics
    "polyester": "polyester",
    "poly": "polyester",
    "pet": "polyester",
    "dacron": "polyester",
    "nylon": "nylon",
    "polyamide": "nylon",
    "pa": "nylon",
    "acrylic": "acrylic",
    "polyacrylic": "acrylic",
    "elastane": "elastane",
    "spandex": "elastane",
    "lycra": "elastane",
    "elastane_spandex": "elastane",
    "polyurethane": "elastane",
    "polyolefin": "polyolefin",
    "polypropylene": "polyolefin",
    "olefin": "polyolefin",
    "aramid": "aramid",
    "kevlar": "aramid",
    "nomex": "aramid",
    # Cellulosics & Bast Fibers
    "viscose": "viscose",
    "rayon": "viscose",
    "viscose_rayon": "viscose",
    "modal": "modal",
    "lyocell": "lyocell",
    "tencel": "lyocell",
    "cupro": "cupro",
    "bemberg": "cupro",
    "triacetate": "triacetate_acetate",
    "acetate": "triacetate_acetate",
    "triacetate_acetate": "triacetate_acetate",
    "linen": "linen",
    "flax": "linen",
    "flax_linen": "linen",
    "hemp": "hemp",
    "jute": "jute",
    "ramie": "ramie",
    "abaca": "abaca",
    "sisal": "sisal",
    # Animal Proteins & Pelts
    "silk": "silk",
    "mulberry silk": "silk",
    "wool": "wool",
    "merino": "wool",
    "merino wool": "wool",
    "cashmere": "cashmere",
    "alpaca": "alpaca",
    "angora": "angora",
    "mohair": "mohair",
    "camel": "camel",
    "camel hair": "camel",
    "llama": "llama",
    "yak": "yak",
    "horse_hair": "horse_hair",
    "horsehair": "horse_hair",
    "fur": "fur",
    "leather": "leather",
    "suede": "suede",
    # Bio-Engineered Fibers
    "soybean_fiber": "soybean_fiber",
    "soy fiber": "soybean_fiber",
    "milk_fiber": "milk_fiber",
    "milk casein": "milk_fiber",
}


class NormalizationService:
    """Service providing canonicalization, percentage validation, and perception cross-checks."""

    @staticmethod
    def canonicalize_material_name(raw_name: str) -> str:
        """Standardize a textile fiber name to its canonical lower_case key."""
        if not raw_name:
            return "unknown"

        cleaned = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
        # Direct check
        if cleaned in _TEXTILE_ALIASES:
            return _TEXTILE_ALIASES[cleaned]

        # Space-separated check
        spaced = cleaned.replace("_", " ")
        if spaced in _TEXTILE_ALIASES:
            return _TEXTILE_ALIASES[spaced]

        return cleaned

    @classmethod
    def normalize_ocr_composition(
        cls, ocr_result: OCRResult
    ) -> Tuple[Optional[Composition], List[str]]:
        """Convert raw OCR extraction into a normalized Composition object with integrity warnings."""
        warnings: List[str] = list(ocr_result.warnings)

        if ocr_result.unreadable or not ocr_result.composition:
            return None, warnings

        materials_map: Dict[str, float] = {}
        for item in ocr_result.composition:
            canon = cls.canonicalize_material_name(item.material)
            # Combine duplicate fiber entries if OCR matched twice
            materials_map[canon] = materials_map.get(canon, 0.0) + item.percentage

        material_list = [
            Material(name=name, percentage=round(pct, 2))
            for name, pct in materials_map.items()
        ]

        comp = Composition(
            materials=material_list,
            confidence=ocr_result.confidence,
            source=AnalysisSource.OCR_LABEL,
        )

        total_pct = comp.total_percentage
        # Composition and scorer share an absolute ±0.01 percentage-point tolerance.
        if not composition_total_is_valid(m.percentage for m in comp.materials):
            warnings.append(
                "Label composition is partial or incomplete: percentages sum to "
                f"{total_pct:.2f}% (scoring requires 100% within ±"
                f"{COMPOSITION_TOTAL_TOLERANCE_PERCENT}%)."
            )

        return comp, warnings

    @classmethod
    def reconcile_findings(
        cls,
        ocr_result: Optional[OCRResult] = None,
        vision_result: Optional[VisionPrediction] = None,
    ) -> Tuple[Optional[Composition], AnalysisSource, str, List[str]]:
        """Implement explicit 5-scenario reconciliation strategy.

        Returns:
            Tuple of:
            - reconciled_composition (Optional[Composition])
            - primary_source (AnalysisSource)
            - reconciliation_status (str)
            - warnings (List[str])
        """
        warnings: List[str] = []
        ocr_comp: Optional[Composition] = None

        if ocr_result is not None:
            ocr_comp, ocr_warns = cls.normalize_ocr_composition(ocr_result)
            warnings.extend(ocr_warns)

        has_ocr = ocr_comp is not None and len(ocr_comp.materials) > 0
        has_vision = (
            vision_result is not None
            and vision_result.predicted_material is not None
            and (vision_result.confidence or 0.0) > 0.0
        )

        if vision_result is not None:
            warnings.extend(vision_result.warnings)

        # Scenario 1: Both modules succeed and agree
        if has_ocr and has_vision:
            ocr_materials = {m.name for m in ocr_comp.materials}
            cv_mat = cls.canonicalize_material_name(vision_result.predicted_material)

            if cv_mat in ocr_materials:
                # Agreement confirmed
                reconciled = Composition(
                    materials=ocr_comp.materials,
                    confidence=max(ocr_comp.confidence or 0.0, vision_result.confidence or 0.0),
                    source=AnalysisSource.HYBRID,
                )
                return reconciled, AnalysisSource.HYBRID, "AGREED", warnings

            # Scenario 5: Both succeed but disagree significantly
            if (vision_result.confidence or 0.0) >= 0.70:
                warnings.append(
                    f"Perception Discrepancy: Surface computer vision predicted '{cv_mat}' ({vision_result.confidence:.0%} confidence), "
                    f"but label OCR detected {sorted(list(ocr_materials))}. Relying on label composition with discrepancy warning."
                )
                return ocr_comp, AnalysisSource.OCR_LABEL, "DISCREPANCY_FLAGGED", warnings

            # Vision had low confidence, prefer OCR
            return ocr_comp, AnalysisSource.OCR_LABEL, "OCR_PRIMARY", warnings

        # Scenario 2: OCR succeeds, Vision unavailable / missing checkpoint
        if has_ocr and not has_vision:
            return ocr_comp, AnalysisSource.OCR_LABEL, "OCR_ONLY", warnings

        # Scenario 3: Vision succeeds, OCR unreadable / missing
        if not has_ocr and has_vision:
            cv_mat = cls.canonicalize_material_name(vision_result.predicted_material)
            warnings.append(
                f"Label unreadable or missing. Fallback composition estimated from Computer Vision ({cv_mat})."
            )
            cv_comp = Composition(
                materials=[Material(name=cv_mat, percentage=100.0)],
                confidence=vision_result.confidence,
                source=AnalysisSource.COMPUTER_VISION,
            )
            return cv_comp, AnalysisSource.COMPUTER_VISION, "VISION_FALLBACK", warnings

        # Scenario 4: Both fail / unreadable
        warnings.append(
            "Both label OCR and Computer Vision were unable to determine garment fabric composition."
        )
        return None, AnalysisSource.UNKNOWN, "BOTH_FAILED", warnings

    @classmethod
    def synthesize_product_analysis(
        cls,
        ocr_result: Optional[OCRResult] = None,
        vision_result: Optional[VisionPrediction] = None,
    ) -> Tuple[ProductAnalysis, List[str]]:
        """Synthesize findings from OCR and Computer Vision using the reconciliation strategy."""
        composition, source, status, warnings = cls.reconcile_findings(
            ocr_result=ocr_result,
            vision_result=vision_result,
        )

        product = ProductAnalysis(
            composition=composition,
            visual_prediction=vision_result,
            composition_source=source,
        )

        return product, warnings

    @classmethod
    async def execute_pipeline(
        cls,
        *,
        garment_image_bytes: Optional[bytes] = None,
        garment_filename: str = "garment.jpg",
        label_image_bytes: Optional[bytes] = None,
        label_filename: str = "label.jpg",
        shopping: bool = True,
        second_hand_available: bool = False,
        condition: str = "new",
        ocr_adapter_instance: Optional[Any] = None,
        vision_adapter_instance: Optional[Any] = None,
        knowledge_adapter_instance: Optional[Any] = None,
        scoring_adapter_instance: Optional[Any] = None,
    ) -> AnalysisResponse:
        """Execute full cross-module analysis pipeline in proper dependency order:
        ImageService -> OCRAdapter / VisionAdapter -> NormalizationService -> KnowledgeBaseAdapter -> ScoringAdapter -> AnalysisResponse.
        """
        start_time = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        warnings: List[str] = []
        limitations: List[str] = [
            "Material impact scores are derived from rule-based calculations using comparative benchmark data; they do not represent empirical garment-specific LCA measurements.",
            "Computer vision predictions reflect surface visual texture classification and may confuse visually similar fibers.",
        ]

        # Lazy import adapters to avoid circular dependencies
        from app.services.image_service import ImageService
        from app.integrations.ocr_adapter import ocr_adapter
        from app.integrations.vision_adapter import vision_adapter
        from app.integrations.knowledge_adapter import knowledge_adapter
        from app.integrations.scoring_adapter import scoring_adapter
        from app.core.exceptions import ScoringError

        ocr = ocr_adapter_instance or ocr_adapter
        vision = vision_adapter_instance or vision_adapter
        kb = knowledge_adapter_instance or knowledge_adapter
        scorer = scoring_adapter_instance or scoring_adapter

        ocr_res: Optional[OCRResult] = None
        vis_res: Optional[VisionPrediction] = None

        # 1. Process Label Image & Run OCR
        if label_image_bytes is not None:
            proc_label = None
            try:
                proc_label = ImageService.process_image(label_image_bytes, filename=label_filename)
                ocr_res = await ocr.extract_composition(proc_label)
            except Exception as e:
                logger.warning("OCR processing error in pipeline: %s", str(e))
                ocr_res = OCRResult(unreadable=True, warnings=[f"OCR extraction failed: {str(e)}"])
            finally:
                if proc_label is not None:
                    proc_label.cleanup()

        # 2. Process Garment Image & Run Computer Vision
        if garment_image_bytes is not None:
            proc_garment = None
            try:
                proc_garment = ImageService.process_image(garment_image_bytes, filename=garment_filename)
                vis_res = await vision.predict_material(proc_garment)
            except Exception as e:
                logger.warning("Vision processing error in pipeline: %s", str(e))
                vis_res = VisionPrediction(
                    predicted_material=None,
                    warnings=[f"Vision prediction failed: {str(e)}"],
                )
            finally:
                if proc_garment is not None:
                    proc_garment.cleanup()

        # 3. If neither image was provided
        if label_image_bytes is None and garment_image_bytes is None:
            warnings.append("No garment or label image was provided for pipeline analysis.")

        # 4. Synthesize & Reconcile findings
        product, synth_warnings = cls.synthesize_product_analysis(
            ocr_result=ocr_res,
            vision_result=vis_res,
        )
        warnings.extend(synth_warnings)

        if product.composition_source == AnalysisSource.COMPUTER_VISION:
            limitations.append(
                "Fabric blend percentages could not be verified from a care label tag; "
                "composition is estimated as 100% of the surface-identified fiber for comparative benchmarking."
            )

        # 5. Knowledge Base Retrieval
        environmental_data: Dict[str, Any] = {}
        if product.composition and product.composition.materials:
            mat_names = [m.name for m in product.composition.materials]
            try:
                kb_res = await kb.get_batch_data(mat_names)
                environmental_data = kb_res.profiles
                if kb_res.unrecognized_materials:
                    warnings.append(
                        f"Materials not found in knowledge base: {', '.join(kb_res.unrecognized_materials)}"
                    )
            except Exception as e:
                logger.warning("Knowledge Base query error in pipeline: %s", str(e))
                warnings.append(f"Knowledge Base query failed: {str(e)}")

        # 6. Scoring Engine Execution
        impact_assessment: ScoringResult
        if product.composition and product.composition.materials:
            blend_dict = {m.name: m.percentage for m in product.composition.materials}
            try:
                impact_assessment = scorer.calculate_impact(
                    blend_dict,
                    garment_condition=condition,
                    shopping=shopping,
                    second_hand_available=second_hand_available,
                )
            except ScoringError as e:
                if e.status_code >= 500:
                    logger.error(
                        "Scoring engine failed (status_code=%s).", e.status_code
                    )
                    status = "ENGINE_ERROR"
                    reason = "The scoring service is unavailable or failed."
                    calculation_basis = "unscored_due_to_engine_error"
                    warning = (
                        "The scoring service is unavailable or failed. "
                        "Composition and material reference information are retained, "
                        "but no score or recommendation was produced."
                    )
                else:
                    logger.info(
                        "Scoring could not use the supplied composition (status_code=%s).",
                        e.status_code,
                    )
                    status = "INSUFFICIENT_DATA"
                    reason = (
                        "Composition data are incomplete, invalid, or unsupported "
                        "by the available scoring data."
                    )
                    calculation_basis = "unscored_due_to_insufficient_data"
                    warning = (
                        "Composition data are incomplete, invalid, or unsupported "
                        "by the scoring data. No score or recommendation was produced."
                    )
                warnings.append(warning)
                impact_assessment = ScoringResult(
                    overall_impact_score=None,
                    recommendation=None,
                    key_indicators={"status": status, "reason": reason},
                    calculation_basis=calculation_basis,
                )
            except Exception as e:
                logger.exception("Unexpected scoring engine failure.")
                warnings.append(
                    "The scoring service is unavailable or failed. Composition and "
                    "material reference information are retained, but no score or "
                    "recommendation was produced."
                )
                impact_assessment = ScoringResult(
                    overall_impact_score=None,
                    recommendation=None,
                    key_indicators={
                        "status": "ENGINE_ERROR",
                        "reason": "The scoring service is unavailable or failed.",
                    },
                    calculation_basis="unscored_due_to_engine_error",
                )
        else:
            impact_assessment = ScoringResult(
                overall_impact_score=None,
                recommendation=None,
                key_indicators={"status": "NO_COMPOSITION_DATA"},
                calculation_basis="unscored_due_to_missing_composition",
            )
            warnings.append("Garment fabric composition could not be established; impact assessment omitted.")

        # 7. Operational Metadata
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        model_versions: Dict[str, str] = {}
        for mod_name, mod_inst in [("ocr", ocr), ("vision", vision), ("scoring", scorer), ("knowledge_base", kb)]:
            try:
                model_versions[mod_name] = mod_inst.status_info().get("adapter", mod_name)
            except Exception:
                model_versions[mod_name] = mod_name

        metadata = AnalysisMetadata(
            processing_time_ms=elapsed_ms,
            model_versions=model_versions,
            pipeline_version="1.0.0",
        )

        confidence_dict: Dict[str, Optional[float]] = {
            "ocr": ocr_res.confidence if ocr_res else None,
            "vision": vis_res.confidence if vis_res else None,
        }

        # Deduplicate warnings while preserving order
        deduped_warnings = list(dict.fromkeys(warnings))

        is_success = bool(product.composition and product.composition.materials)

        return AnalysisResponse(
            success=is_success,
            analysis_id=analysis_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            product=product,
            environmental_data=environmental_data,
            impact_assessment=impact_assessment,
            ai_explanation=None,
            confidence=confidence_dict,
            limitations=limitations,
            warnings=deduped_warnings,
            metadata=metadata,
        )


# Global default normalization service instance
normalization_service = NormalizationService()
