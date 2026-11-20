class RecommendationEngine:
    """Simple explainable rules; condition/context can override score-only advice."""

    def recommend(
        self,
        environmental_score,
        *,
        garment_condition="new",
        shopping=False,
        second_hand_available=False,
    ):
        if not 0 <= float(environmental_score) <= 100:
            raise ValueError("environmental_score must be between 0 and 100.")

        condition = garment_condition.strip().lower()

        # These actions depend on garment condition, not just the material score.
        if condition in {"damaged", "minor_damage", "repairable"}:
            return {
                "recommendation": "Repair",
                "reason": "The garment is damaged or repairable; extending its life may avoid replacement.",
            }

        if condition in {"usable", "good", "existing"} and not shopping:
            return {
                "recommendation": "Reuse",
                "reason": "The garment is still usable; continue wearing it rather than replacing it.",
            }

        if shopping and second_hand_available:
            return {
                "recommendation": "Second-hand",
                "reason": "A suitable pre-owned option is available and may avoid demand for a newly produced garment.",
            }

        if float(environmental_score) <= 40:
            return {
                "recommendation": "Buy",
                "reason": "The estimated impact is in the low-to-moderate range of this provisional model; buy only if needed.",
            }

        return {
            "recommendation": "Think Twice",
            "reason": "The estimated environmental impact is above the model's provisional Buy threshold; consider alternatives.",
        }
