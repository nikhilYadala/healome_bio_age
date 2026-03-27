"""
Healome Aging Clock — blood-based biological age estimation from standard clinical biomarkers.

Primary model: GradientBoosting trained on ~50K NHANES records (1999-2020).
Two variants:
  - standard: 21 features (16 lab biomarkers + 5 questionnaire items)
  - extended: 35 features (expanded lab panel + questionnaire)

Usage:
    from healome_clock import predict_age
    result = predict_age({"glycohemoglobin_percent": 5.4, "glucose_mg_dl": 95, ...}, chronological_age=45)
    print(result.summary())
"""

__version__ = "0.1.0"

from healome_clock.inference import predict_age, HealomeClock, AgeResult
from healome_clock.feature_aliases import (
    NHANES_TO_CANONICAL_KEY,
    normalize_blood_panel_to_nhanes,
    list_friendly_features_for_variant,
)
from healome_clock.models.tree import (
    TreeModel,
    STANDARD_21_FEATURES,
    EXTENDED_35_FEATURES,
    FEATURE_NAMES,
)

__all__ = [
    "predict_age",
    "HealomeClock",
    "AgeResult",
    "TreeModel",
    "STANDARD_21_FEATURES",
    "EXTENDED_35_FEATURES",
    "FEATURE_NAMES",
    "NHANES_TO_CANONICAL_KEY",
    "normalize_blood_panel_to_nhanes",
    "list_friendly_features_for_variant",
]
