"""
Inference pipeline for the Healome Aging Clock.

Primary model: tree-based GradientBoosting (standard 21-feature or extended 35-feature).
Experimental: autoencoder+CNN (see models.experimental).
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from pathlib import Path
from dataclasses import dataclass, field

from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes
from healome_clock.models.tree import (
    TreeModel,
    STANDARD_21_FEATURES,
    EXTENDED_35_FEATURES,
    FEATURE_NAMES,
    MODEL_CONFIGS,
)


@dataclass
class AgeResult:
    """Container for biological age prediction results."""

    biological_age: float
    chronological_age_delta: Optional[float] = None
    age_acceleration: Optional[str] = None
    biomarker_values: Dict[str, float] = field(default_factory=dict)
    model_variant: str = "standard"
    model_version: str = "0.1.0"

    def __repr__(self):
        parts = [f"AgeResult(biological_age={self.biological_age:.1f}"]
        if self.chronological_age_delta is not None:
            parts.append(f", delta={self.chronological_age_delta:+.1f}")
        parts.append(f", variant='{self.model_variant}')")
        return "".join(parts)

    def summary(self) -> str:
        lines = [f"Healome Biological Age Estimate: {self.biological_age:.1f} years"]
        if self.chronological_age_delta is not None:
            direction = "older" if self.chronological_age_delta > 0 else "younger"
            lines.append(
                f"  Delta: {abs(self.chronological_age_delta):.1f} years {direction} "
                f"than chronological age"
            )
            if self.age_acceleration:
                lines.append(f"  Classification: {self.age_acceleration}")
        lines.append(f"  Model: {self.model_variant} ({MODEL_CONFIGS[self.model_variant]['n_features']} features)")
        return "\n".join(lines)


class HealomeClock:
    """
    Main interface for the Healome Aging Clock.

    Args:
        variant: "standard" (21 features) or "extended" (35 features).
        model_path: Override path to .joblib weights file.

    Example:
        >>> clock = HealomeClock(variant="standard")
        >>> result = clock.predict({"glycohemoglobin_percent": 5.4, "glucose_mg_dl": 95, ...}, chronological_age=45)
        >>> print(result.biological_age)
    """

    def __init__(
        self,
        variant: str = "standard",
        model_path: Optional[Union[str, Path]] = None,
    ):
        self.variant = variant
        self._tree = TreeModel(variant=variant, model_path=model_path)

    @property
    def features(self):
        return self._tree.features

    @property
    def config(self):
        return self._tree.config

    def predict(
        self,
        blood_panel: Union[str, Path, pd.DataFrame, Dict],
        chronological_age: Optional[float] = None,
    ) -> Union[AgeResult, list]:
        """
        Predict biological age from blood panel data.

        Args:
            blood_panel: CSV/JSON path, DataFrame, or dict of biomarker values.
                Use canonical snake_case names (see README) or original NHANES codes.
            chronological_age: If provided, computes delta and acceleration classification.

        Returns:
            AgeResult for single sample, list of AgeResult for multiple.
        """
        if isinstance(blood_panel, dict):
            normalized = normalize_blood_panel_to_nhanes(blood_panel, self.variant)
            df = pd.DataFrame([normalized])
        elif isinstance(blood_panel, pd.DataFrame):
            df = normalize_blood_panel_to_nhanes(blood_panel.copy(), self.variant)
        else:
            path = Path(blood_panel)
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            df = normalize_blood_panel_to_nhanes(df, self.variant)

        preds = self._tree.predict(df)

        results = []
        for i, bio_age in enumerate(preds):
            delta = None
            accel = None
            if chronological_age is not None:
                delta = float(bio_age - chronological_age)
                if delta >= 5:
                    accel = "accelerated aging"
                elif delta <= -5:
                    accel = "decelerated aging"
                else:
                    accel = "normal aging"

            biomarkers = {}
            for feat in self._tree.features:
                if feat in df.columns and not pd.isna(df.iloc[i].get(feat)):
                    name = FEATURE_NAMES.get(feat, feat)
                    biomarkers[name] = float(df.iloc[i][feat])

            results.append(AgeResult(
                biological_age=float(bio_age),
                chronological_age_delta=delta,
                age_acceleration=accel,
                biomarker_values=biomarkers,
                model_variant=self.variant,
            ))

        return results[0] if len(results) == 1 else results

    def feature_importances(self) -> pd.DataFrame:
        """Return feature importances from the trained model."""
        return self._tree.feature_importances()

    def __repr__(self):
        return f"HealomeClock(variant='{self.variant}', features={self._tree.n_features})"


_default_clock = None


def predict_age(
    blood_panel: Union[str, Path, pd.DataFrame, Dict],
    chronological_age: Optional[float] = None,
    variant: str = "standard",
    model_path: Optional[Union[str, Path]] = None,
) -> Union[AgeResult, list]:
    """
    Convenience function for biological age prediction.

    Args:
        blood_panel: Blood panel data (CSV path, DataFrame, or dict).
            Keys may be canonical friendly names or NHANES codes (see README).
        chronological_age: Optional chronological age for delta computation.
        variant: "standard" (21 features) or "extended" (35 features).
        model_path: Override path to model weights.

    Returns:
        AgeResult with biological age estimate.

    Example:
        >>> from healome_clock import predict_age
        >>> result = predict_age({"glycohemoglobin_percent": 5.4, "glucose_mg_dl": 95, ...})
        >>> print(result.biological_age)
    """
    global _default_clock
    if (_default_clock is None
        or _default_clock.variant != variant
        or (model_path and Path(model_path) != _default_clock._tree._model_path)):
        _default_clock = HealomeClock(variant=variant, model_path=model_path)
    return _default_clock.predict(blood_panel, chronological_age=chronological_age)
