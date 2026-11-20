"""Gemini Explanation Service.

Orchestrates the conversion of structured Phase 5 analytical outputs into
personalized, consumer-friendly sustainability explanations using GeminiAdapter.
"""

from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.integrations.gemini_adapter import GeminiAdapter, gemini_adapter
from app.schemas.response import (
    AiExplanation,
    MaterialEnvironmentalProfile,
    ProductAnalysis,
    ScoringResult,
)


class GeminiService:
    """Service providing AI-assisted sustainability explanations strictly grounded in Phase 5 data."""

    def __init__(self, adapter: Optional[GeminiAdapter] = None):
        self._adapter = adapter or gemini_adapter

    @property
    def adapter(self) -> GeminiAdapter:
        return self._adapter

    def generate_fallback_explanation(
        self,
        analysis_data: Dict[str, Any],
        reason: str = "Standard deterministic explanation generated.",
        status: str = "UNAVAILABLE",
    ) -> AiExplanation:
        """Public method to generate a grounded, deterministic fallback explanation.

        Safe against unhandled exceptions during fallback generation to guarantee
        that Phase 5 analytical results are never lost.
        """
        try:
            return self._adapter.generate_fallback(
                analysis_data=analysis_data,
                reason=reason,
                status=status,
            )
        except Exception as e:
            logger.warning("Error generating fallback explanation from adapter: %s", str(e))
            # Minimal bulletproof fallback guaranteed never to fail
            return AiExplanation(
                status="UNAVAILABLE",
                summary="Garment analysis completed with comparative benchmark indicators.",
                environmental_highlights=["Comparative assessment based on textile reference benchmarks."],
                practical_recommendations=["Refer to garment care label for maintenance instructions."],
                care_and_repair_tips=[
                    "General textile-care guidance (not garment-specific): Wash cold and line dry to preserve fabric."
                ],
                limitations_and_uncertainty=[
                    f"Fallback generation notice: {str(e)}",
                    "Comparative benchmarks; not empirical declarations.",
                ],
                data_quality_disclaimer="Environmental impact indicators and scores are comparative benchmarks based on published literature and demo data; they do not represent empirical certified product declarations.",
                is_fallback=True,
            )

    async def explain_analysis(
        self,
        product: ProductAnalysis,
        environmental_data: Dict[str, MaterialEnvironmentalProfile],
        impact_assessment: ScoringResult,
        confidence: Optional[Dict[str, Optional[float]]] = None,
        warnings: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
    ) -> AiExplanation:
        """Generate consumer-friendly explanation grounded strictly in Phase 5 analytical facts."""
        payload: Dict[str, Any] = {
            "product": product.model_dump(mode="json") if hasattr(product, "model_dump") else product,
            "environmental_data": {
                k: v.model_dump(mode="json") if hasattr(v, "model_dump") else v
                for k, v in environmental_data.items()
            },
            "impact_assessment": impact_assessment.model_dump(mode="json") if hasattr(impact_assessment, "model_dump") else impact_assessment,
            "confidence": confidence or {},
            "warnings": warnings or [],
            "limitations": limitations or [],
        }

        try:
            explanation = await self._adapter.generate_explanation(payload)
            return explanation
        except Exception as e:
            logger.warning("Gemini explanation generation encountered unexpected error: %s", str(e))
            return self.generate_fallback_explanation(
                analysis_data=payload,
                reason=f"Gemini service exception: {str(e)}",
                status="UNAVAILABLE",
            )


# Global default Gemini service instance
gemini_service = GeminiService()
