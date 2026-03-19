"""
Tests for the Healome Aging Clock inference and evaluation pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestFeatureAliases:
    def test_friendly_dict_maps_to_nhanes_standard(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes
        from healome_clock.models.tree import STANDARD_21_FEATURES

        out = normalize_blood_panel_to_nhanes(
            {"glycohemoglobin_percent": 5.4, "glucose_mg_dl": 95},
            "standard",
        )
        assert out["LBXGH"] == 5.4
        assert out["LBXSGL"] == 95
        assert set(out.keys()) <= set(STANDARD_21_FEATURES)

    def test_unknown_dict_key_warns(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        with pytest.warns(UserWarning, match="Ignoring unknown"):
            out = normalize_blood_panel_to_nhanes(
                {"glycohemoglobin_percent": 5.4, "not_a_real_marker": 1},
                "standard",
            )
        assert "LBXGH" in out
        assert "not_a_real_marker" not in out

    def test_nhanes_codes_passthrough(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        out = normalize_blood_panel_to_nhanes({"LBXGH": 5.4, "LBXSGL": 95}, "standard")
        assert out == {"LBXGH": 5.4, "LBXSGL": 95}

    def test_synonym_hba1c(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        out = normalize_blood_panel_to_nhanes({"hba1c_percent": 5.4}, "standard")
        assert out["LBXGH"] == 5.4

    def test_dataframe_rename(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        df = pd.DataFrame([{"glycohemoglobin_percent": 5.4, "glucose_mg_dl": 95}])
        out = normalize_blood_panel_to_nhanes(df, "standard")
        assert "LBXGH" in out.columns
        assert "LBXSGL" in out.columns
        assert float(out["LBXGH"].iloc[0]) == 5.4

    def test_conflicting_duplicate_raises(self):
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        with pytest.raises(ValueError, match="Conflicting"):
            normalize_blood_panel_to_nhanes(
                {"LBXGH": 5.0, "glycohemoglobin_percent": 6.0},
                "standard",
            )

    def test_list_friendly_features_order_matches_model(self):
        from healome_clock.feature_aliases import list_friendly_features_for_variant
        from healome_clock.models.tree import STANDARD_21_FEATURES, EXTENDED_35_FEATURES

        std = list_friendly_features_for_variant("standard")
        assert [row[1] for row in std] == STANDARD_21_FEATURES
        ext = list_friendly_features_for_variant("extended")
        assert [row[1] for row in ext] == EXTENDED_35_FEATURES


class TestFeatureConstants:
    def test_standard_feature_count(self):
        from healome_clock.models.tree import STANDARD_21_FEATURES
        assert len(STANDARD_21_FEATURES) == 21

    def test_extended_feature_count(self):
        from healome_clock.models.tree import EXTENDED_35_FEATURES
        assert len(EXTENDED_35_FEATURES) == 35

    def test_standard_subset_of_extended(self):
        from healome_clock.models.tree import STANDARD_21_FEATURES, EXTENDED_35_FEATURES
        standard_set = set(STANDARD_21_FEATURES)
        extended_set = set(EXTENDED_35_FEATURES)
        missing = standard_set - extended_set
        assert len(missing) == 0, f"Standard features not in extended: {missing}"

    def test_feature_names_coverage(self):
        from healome_clock.models.tree import (
            STANDARD_21_FEATURES, EXTENDED_35_FEATURES, FEATURE_NAMES
        )
        all_feats = set(STANDARD_21_FEATURES) | set(EXTENDED_35_FEATURES)
        for feat in all_feats:
            assert feat in FEATURE_NAMES, f"{feat} missing from FEATURE_NAMES"

    def test_model_configs_exist(self):
        from healome_clock.models.tree import MODEL_CONFIGS
        assert "standard" in MODEL_CONFIGS
        assert "extended" in MODEL_CONFIGS
        assert MODEL_CONFIGS["standard"]["n_features"] == 21
        assert MODEL_CONFIGS["extended"]["n_features"] == 35

    def test_importance_order_counts(self):
        from healome_clock.models.tree import (
            STANDARD_21_IMPORTANCE_ORDER, EXTENDED_35_IMPORTANCE_ORDER
        )
        assert len(STANDARD_21_IMPORTANCE_ORDER) == 21
        assert len(EXTENDED_35_IMPORTANCE_ORDER) == 35


class TestTreeModel:
    def test_instantiation(self):
        from healome_clock.models.tree import TreeModel
        model = TreeModel(variant="standard")
        assert model.variant == "standard"
        assert model.n_features == 21

    def test_invalid_variant_raises(self):
        from healome_clock.models.tree import TreeModel
        with pytest.raises(ValueError, match="Unknown variant"):
            TreeModel(variant="nonexistent")

    def test_prepare_features_order(self):
        from healome_clock.models.tree import TreeModel, STANDARD_21_FEATURES
        model = TreeModel(variant="standard")
        data = {feat: float(i) for i, feat in enumerate(STANDARD_21_FEATURES)}
        data["EXTRA_COL"] = 999.0
        df = pd.DataFrame([data])
        x = model.prepare_features(df)
        assert x.shape == (1, 21)
        for i, feat in enumerate(STANDARD_21_FEATURES):
            assert x[0, i] == float(i)

    def test_prepare_features_missing_cols(self):
        from healome_clock.models.tree import TreeModel
        model = TreeModel(variant="standard")
        df = pd.DataFrame([{"LBXGH": 5.4, "LBXSGL": 95}])
        x = model.prepare_features(df)
        assert x.shape == (1, 21)
        assert not np.any(np.isnan(x))


class TestAgeResult:
    def test_result_creation(self):
        from healome_clock.inference import AgeResult
        result = AgeResult(
            biological_age=42.3,
            chronological_age_delta=-2.7,
            age_acceleration="decelerated aging",
            model_variant="standard",
        )
        assert result.biological_age == 42.3
        assert "42.3" in repr(result)

    def test_result_summary(self):
        from healome_clock.inference import AgeResult
        result = AgeResult(
            biological_age=42.3,
            chronological_age_delta=-2.7,
            age_acceleration="decelerated aging",
        )
        summary = result.summary()
        assert "42.3" in summary
        assert "younger" in summary
        assert "decelerated" in summary


class TestHealomeClock:
    def test_instantiation(self):
        from healome_clock.inference import HealomeClock
        clock = HealomeClock(variant="standard")
        assert clock.variant == "standard"
        assert len(clock.features) == 21

    def test_repr(self):
        from healome_clock.inference import HealomeClock
        clock = HealomeClock(variant="extended")
        r = repr(clock)
        assert "extended" in r
        assert "35" in r


class TestMetrics:
    def test_compute_age_metrics(self):
        from healome_clock.evaluation.metrics import compute_age_metrics
        y_true = np.array([30, 40, 50, 60, 70])
        y_pred = np.array([32, 38, 52, 58, 72])
        metrics = compute_age_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "pearson_r" in metrics
        assert metrics["mae"] == 2.0
        assert metrics["n_samples"] == 5

    def test_compute_subgroup_metrics(self):
        from healome_clock.evaluation.metrics import compute_subgroup_metrics
        y_true = np.array([25, 35, 55, 65])
        y_pred = np.array([27, 33, 57, 63])
        groups = np.array(["young", "young", "old", "old"])
        df = compute_subgroup_metrics(y_true, y_pred, groups)
        assert len(df) == 2
        assert "mae" in df.columns

    def test_compute_age_bucket_metrics(self):
        from healome_clock.evaluation.metrics import compute_age_bucket_metrics
        y_true = np.array([25, 35, 45, 55, 65, 75])
        y_pred = y_true + 2
        df = compute_age_bucket_metrics(y_true, y_pred)
        assert len(df) >= 2


class TestSurvival:
    def test_classify_aging_rate(self):
        from healome_clock.evaluation.survival import classify_aging_rate
        bio = np.array([50, 40, 45])
        chrono = np.array([40, 50, 45])
        labels = classify_aging_rate(bio, chrono, threshold=5)
        assert labels[0] == "accelerated"
        assert labels[1] == "decelerated"
        assert labels[2] == "normal"


class TestLeaderboard:
    def test_create_submission(self):
        from healome_clock.evaluation.leaderboard import create_submission
        y_true = np.array([30, 40, 50, 60, 70])
        y_pred = np.array([32, 38, 52, 58, 72])
        sub = create_submission("test_model", y_true, y_pred, authors="Test")
        assert sub["model_name"] == "test_model"
        assert "track1_age_prediction" in sub
        assert sub["track1_age_prediction"]["mae"] == 2.0


class TestDataRegistry:
    def test_register_and_list(self):
        from healome_clock.data.registry import (
            register_data_source, list_data_sources, get_data_source
        )
        register_data_source(
            "test_source",
            loader=lambda: pd.DataFrame(),
            description="Test data source",
        )
        sources = list_data_sources()
        assert len(sources) >= 1
        assert "test_source" in sources["name"].values

        source = get_data_source("test_source")
        assert source["description"] == "Test data source"

    def test_missing_source_raises(self):
        from healome_clock.data.registry import get_data_source
        with pytest.raises(KeyError):
            get_data_source("nonexistent_source_xyz")
