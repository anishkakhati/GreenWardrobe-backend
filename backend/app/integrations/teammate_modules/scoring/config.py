# All category weights must add up to 1.0.
# These are provisional project weights, not a universally accepted scientific standard.
CATEGORY_WEIGHTS = {
    "water": 0.20,
    "carbon": 0.25,
    "energy": 0.15,
    "chemicals": 0.15,
    "microfibres": 0.10,
    "waste": 0.15,
}

# Impact scores are on a 0-100 scale: 0 = lower impact, 100 = higher impact.
# Thresholds are provisional and should be approved by the project team.
SCORE_BANDS = [
    (20, "Very Low Impact"),
    (40, "Low Impact"),
    (60, "Moderate Impact"),
    (80, "High Impact"),
    (100, "Very High Impact"),
]

# Minimum share of total category weight that must have data to publish a score.
MIN_DATA_COVERAGE = 0.60
