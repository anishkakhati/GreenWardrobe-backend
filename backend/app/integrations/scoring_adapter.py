"""Scoring Adapter integrating Maneesh Kumar's Environmental Scoring & Recommendation Engine.

Connects the actual rule-based scoring engine from origin/Scoring-&-Decision-Engine.
Maps normalized compositions to category-weighted environmental impact indices
and generates explainable decision recommendations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from app.core.composition import (
    COMPOSITION_TOTAL_TOLERANCE_PERCENT,
    composition_total_is_valid,
)
from app.core.exceptions import ScoringError
from app.core.logging import logger
from app.integrations.base import BaseScoringAdapter
from app.integrations.teammate_modules.scoring import (
    EnvironmentalScorer,
    RecommendationEngine,
)
from app.schemas.response import (
    ImpactScoreBreakdown,
    IndicatorScore,
    ScoringResult,
)


class ScoringAdapter(BaseScoringAdapter):
    """Adapter for Maneesh's EnvironmentalScorer and RecommendationEngine."""

    def __init__(
        self,
        scorer: Optional[EnvironmentalScorer] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        data_path: Optional[Path] = None,
    ):
        try:
            self._scorer = scorer or EnvironmentalScorer(data_path=data_path)
            self._recommendation_engine = recommendation_engine or RecommendationEngine()
            self._available = True
        except Exception as e:
            logger.error("Failed to initialize EnvironmentalScorer: %s", e)
            self._scorer = None
            self._recommendation_engine = None
            self._available = False

    @property
    def adapter_name(self) -> str:
        return "Maneesh-ScoringEngine-Adapter"

    @property
    def is_available(self) -> bool:
        return self._available and self._scorer is not None

    def status_info(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "status": "REAL INTEGRATION VERIFIED" if self.is_available else "FAILED_INITIALIZATION",
            "owner": "Maneesh Kumar",
            "weights_configured": list(self._scorer.weights.keys()) if self._scorer else [],
            "available_materials_count": len(self._scorer.data.index) if self._scorer else 0,
            "provenance": "Rule-based engine with DEMO PLACEHOLDER benchmarks; not certified laboratory LCA measurements.",
            "coverage_rule": "Enforces 60% minimum data coverage threshold.",
            "categories": ["water", "carbon", "energy", "chemicals", "microfibres", "waste"],
        }

    def calculate_impact(
        self,
        composition: Dict[str, float],
        *,
        garment_condition: str = "new",
        shopping: bool = False,
        second_hand_available: bool = False,
        **kwargs: Any,
    ) -> ScoringResult:
        """Calculate weighted impact score and recommendation using Maneesh's engine."""
        if not self.is_available:
            raise ScoringError(
                message="Environmental scoring engine is not initialized.",
                status_code=500,
            )

        if not composition:
            raise ScoringError(
                message="Cannot calculate impact for empty fabric composition.",
                status_code=400,
            )

        if not composition_total_is_valid(composition.values()):
            try:
                total_pct = sum(float(value) for value in composition.values())
            except (TypeError, ValueError):
                total_pct = float("nan")
            raise ScoringError(
                message=(
                    "Composition percentages must total 100% within an absolute "
                    f"tolerance of {COMPOSITION_TOTAL_TOLERANCE_PERCENT} percentage points; "
                    f"received {total_pct:.2f}%."
                ),
                status_code=422,
                details={"composition": composition},
            )

        # Pass through the extracted proportions unchanged; the scorer
        # validates the same strict total and never repairs the blend.
        try:
            score_data = self._scorer.score(dict(composition))
        except ValueError as e:
            # EnvironmentalScorer uses ValueError for expected input/data
            # validation failures such as an unknown material or low coverage.
            logger.info("Scoring input/data validation rejected: %s", str(e))
            raise ScoringError(
                message=str(e),
                status_code=422,
                details={"composition": composition},
            )
        except Exception as e:
            logger.exception("Unexpected error in environmental scoring engine.")
            raise ScoringError(
                message=f"Scoring engine failure: {str(e)}",
                status_code=500,
            )

        # Recommendation errors are engine failures even if the implementation
        # happens to raise ValueError; the composition has already passed input
        # validation and the scorer has already returned a result.
        try:
            rec_data = self._recommendation_engine.recommend(
                score_data["overall_score"],
                garment_condition=garment_condition,
                shopping=shopping,
                second_hand_available=second_hand_available,
            )
        except Exception as e:
            logger.exception("Unexpected error in recommendation engine.")
            raise ScoringError(
                message=f"Recommendation engine failure: {str(e)}",
                status_code=500,
            )

        try:
            # Map category scores into structured Pydantic breakdown.
            category_scores = score_data.get("category_scores", {})
            breakdown = ImpactScoreBreakdown(
                water=IndicatorScore(
                    score=category_scores.get("water", 0.0) or 0.0,
                    description="Water consumption and depletion impact",
                ) if category_scores.get("water") is not None else None,
                carbon=IndicatorScore(
                    score=category_scores.get("carbon", 0.0) or 0.0,
                    description="Greenhouse gas emissions impact",
                ) if category_scores.get("carbon") is not None else None,
                energy=IndicatorScore(
                    score=category_scores.get("energy", 0.0) or 0.0,
                    description="Embodied manufacturing and dyeing energy",
                ) if category_scores.get("energy") is not None else None,
                chemicals=IndicatorScore(
                    score=category_scores.get("chemicals", 0.0) or 0.0,
                    description="Agricultural pesticides and synthetic processing chemicals",
                ) if category_scores.get("chemicals") is not None else None,
                microfibres=IndicatorScore(
                    score=category_scores.get("microfibres", 0.0) or 0.0,
                    description="Microplastic shedding risk per wash cycle",
                ) if category_scores.get("microfibres") is not None else None,
                waste=IndicatorScore(
                    score=category_scores.get("waste", 0.0) or 0.0,
                    description="End-of-life disposal and biodegradability burden",
                ) if category_scores.get("waste") is not None else None,
            )

            # 4. Formulate key environmental indicators
            key_indicators = {
                "impact_classification": score_data.get("classification", "Unknown"),
                "data_coverage": f"{score_data.get('data_coverage', 1.0) * 100:.0f}%",
                "recommendation_reason": rec_data.get("reason", ""),
            }

            return ScoringResult(
                overall_impact_score=float(score_data["overall_score"]),
                recommendation=rec_data["recommendation"],
                breakdown=breakdown,
                key_indicators=key_indicators,
                calculation_basis="rule_based_lca_weighted_average",
            )

        except Exception as e:
            logger.exception("Could not convert scoring-engine output to response.")
            raise ScoringError(
                message=f"Scoring engine response failure: {str(e)}",
                status_code=500,
            )


# Global default scoring adapter instance
scoring_adapter = ScoringAdapter()
