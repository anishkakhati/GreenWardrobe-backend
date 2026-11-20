from pathlib import Path
import numpy as np
import pandas as pd

from app.core.composition import composition_total_is_valid
from .config import CATEGORY_WEIGHTS, SCORE_BANDS, MIN_DATA_COVERAGE


class EnvironmentalScorer:
    """Rule-based material/blend impact scorer.

    Input category values must already be normalized to 0..100, where higher
    always means worse environmental impact. The included CSV contains DEMO
    placeholders, not verified lifecycle-assessment measurements.
    """

    def __init__(self, data_path=None, weights=None):
        if data_path is None:
            candidate1 = Path(__file__).resolve().parent / "data" / "material_impact_data.csv"
            candidate2 = Path(__file__).resolve().parent.parent / "data" / "material_impact_data.csv"
            data_path = candidate1 if candidate1.exists() else candidate2
        self.data = pd.read_csv(data_path).set_index("material")
        self.weights = weights or CATEGORY_WEIGHTS.copy()

        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError("Category weights must add up to 1.0.")
        missing_cols = set(self.weights) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")

        for category in self.weights:
            vals = pd.to_numeric(self.data[category], errors="coerce").dropna()
            if ((vals < 0) | (vals > 100)).any():
                raise ValueError(f"{category} values must be between 0 and 100.")

    @staticmethod
    def _validate_composition(composition):
        if not isinstance(composition, dict) or not composition:
            raise ValueError("composition must be a non-empty dict, e.g. {'polyester': 65, 'cotton': 35}.")
        cleaned = {}
        for material, percentage in composition.items():
            material = str(material).strip().lower().replace(" ", "_")
            try:
                percentage = float(percentage)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid percentage for {material}.")
            if percentage < 0:
                raise ValueError("Material percentages cannot be negative.")
            if percentage > 0:
                cleaned[material] = percentage
        total = sum(cleaned.values())
        if not composition_total_is_valid(cleaned.values()):
            raise ValueError(f"Composition must total 100%; received {total:.2f}%.")
        return cleaned

    def score(self, composition):
        """Return overall score, category scores, coverage, band and explanation."""
        composition = self._validate_composition(composition)

        unknown = set(composition) - set(self.data.index)
        if unknown:
            raise ValueError(
                f"Unknown material(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(self.data.index)}"
            )

        # Blend each environmental category by the material percentages.
        category_scores = {}
        category_coverage = {}
        for category, weight in self.weights.items():
            weighted_total = 0.0
            known_share = 0.0
            for material, percentage in composition.items():
                value = self.data.loc[material, category]
                if pd.notna(value):
                    share = percentage / 100.0
                    weighted_total += float(value) * share
                    known_share += share

            if known_share > 0:
                category_scores[category] = weighted_total / known_share
                category_coverage[category] = known_share
            else:
                category_scores[category] = None
                category_coverage[category] = 0.0

        # Missing categories are omitted and remaining category weights
        # are proportionally renormalized. Coverage makes this visible.
        available_weight = sum(
            self.weights[c] for c, value in category_scores.items() if value is not None
        )
        if available_weight < MIN_DATA_COVERAGE:
            raise ValueError(
                f"Insufficient environmental data coverage ({available_weight:.0%}). "
                f"At least {MIN_DATA_COVERAGE:.0%} is required to publish a score."
            )

        overall = sum(
            category_scores[c] * self.weights[c]
            for c in self.weights if category_scores[c] is not None
        ) / available_weight

        band = next((label for upper, label in SCORE_BANDS if overall <= upper), "Very High Impact")
        return {
            "overall_score": round(float(overall), 2),
            "score_scale": "0-100; higher means higher environmental impact",
            "classification": band,
            "category_scores": {
                k: None if v is None else round(float(v), 2)
                for k, v in category_scores.items()
            },
            "data_coverage": round(float(available_weight), 3),
            "category_weights_used": {
                k: self.weights[k] for k, v in category_scores.items() if v is not None
            },
            "composition": composition,
            "assumptions": [
                "Material category inputs are normalized impact indices from 0 to 100.",
                "Blend impacts are calculated as a percentage-weighted average.",
                "Missing category data is excluded and remaining weights are renormalized.",
                "Scores are indicative and depend on dataset scope, geography and lifecycle boundaries."
            ],
        }
