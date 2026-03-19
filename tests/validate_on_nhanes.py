"""
Validate Healome Clock models against real NHANES data.

Reproduces the exact data loading and train/test split from the
training notebook to verify that the .joblib models produce the
expected MAE, R², and Pearson correlation on the held-out test set.

Expected results (from training notebook):
  Standard 21-feat: MAE ~5.11, R² ~0.906 (90% test split)
                    MAE ~6.37, R² ~0.863 (30% test split)
  Extended 35-feat: MAE ~6.07, R² ~0.873 (30% test split)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from healome_clock.models.tree import (
    TreeModel,
    STANDARD_21_FEATURES,
    EXTENDED_35_FEATURES,
)
from healome_clock.evaluation.metrics import compute_age_metrics, print_metrics


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
NHANES_BASE = os.path.abspath(os.path.join(_REPO_ROOT, "nhanes_data_dump"))
NHANES_PRIMARY = os.path.join(NHANES_BASE, "2017-2020")
NHANES_EXTENDED_DIRS = [
    os.path.join(NHANES_BASE, "extended_data", "2015-16"),
    os.path.join(NHANES_BASE, "extended_data", "2013-14"),
    os.path.join(NHANES_BASE, "extended_data", "2011-12"),
    os.path.join(NHANES_BASE, "extended_data", "2009-10"),
    os.path.join(NHANES_BASE, "extended_data", "2007-08"),
    os.path.join(NHANES_BASE, "extended_data", "2005-06"),
    os.path.join(NHANES_BASE, "extended_data", "2003-04"),
]


def load_primary_nhanes():
    """Load and merge primary NHANES 2017-2020 data (matching notebook exactly)."""
    print("Loading primary NHANES 2017-2020...")
    dd_age = pd.read_sas(os.path.join(NHANES_PRIMARY, "P_BIOPRO.XPT"), format="XPORT")
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_MCQ.XPT"), format="XPORT"),
        how="outer", on=["SEQN"],
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_TRIGLY.XPT"), format="XPORT"),
        how="outer", on=["SEQN"],
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_HSCRP.XPT"), format="XPORT"),
        on=["SEQN"], how="outer",
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_HDL.XPT"), format="XPORT"),
        on=["SEQN"], how="outer",
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_CBC.XPT"), format="XPORT"),
        on=["SEQN"], how="outer",
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_GHB.XPT"), format="XPORT"),
        on=["SEQN"], how="outer",
    )
    dd_age = dd_age.merge(
        pd.read_sas(os.path.join(NHANES_PRIMARY, "P_DEMO.XPT"), format="XPORT"),
        on=["SEQN"], how="outer",
    )
    print(f"  Primary: {dd_age.shape[0]} records, {dd_age.shape[1]} columns")
    return dd_age


def load_extended_nhanes(dd_age):
    """Load and append extended NHANES cycles (matching notebook exactly)."""
    print("Loading extended NHANES cycles...")
    for pfx in NHANES_EXTENDED_DIRS:
        if not os.path.exists(pfx):
            print(f"  Skipping {pfx} (not found)")
            continue
        new_dd_age = None
        for file in listdir(pfx):
            if not file.upper().endswith(".XPT"):
                continue
            try:
                df = pd.read_sas(os.path.join(pfx, file), format="xport")
                if new_dd_age is None:
                    new_dd_age = df
                else:
                    new_dd_age = new_dd_age.merge(df, how="outer")
            except Exception:
                pass
        if new_dd_age is not None:
            dd_age = pd.concat([dd_age, new_dd_age], ignore_index=True)
            print(f"  Added {os.path.basename(pfx)}: total now {dd_age.shape[0]}")
    return dd_age


def validate_model(dd_age, features, model_variant, test_size=0.3):
    """Run validation for a model variant."""
    print(f"\n{'='*60}")
    print(f"  Validating: {model_variant} ({len(features)} features)")
    print(f"  Test split: {test_size}, random_state=3454")
    print(f"{'='*60}")

    x = np.array(dd_age.loc[:, features])
    y = np.array(dd_age.loc[:, "RIDAGEYR"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=3454,
    )

    model = TreeModel(variant=model_variant)
    _ = model.model  # trigger load

    y_pred_train = model.model.predict(x_train)
    y_pred_test = model.model.predict(x_test)

    train_metrics = compute_age_metrics(y_train, y_pred_train)
    test_metrics = compute_age_metrics(y_test, y_pred_test)

    print_metrics(train_metrics, f"{model_variant} — TRAIN")
    print_metrics(test_metrics, f"{model_variant} — TEST")

    return train_metrics, test_metrics


def main():
    print("=" * 60)
    print("  NHANES Validation of Healome Clock Models")
    print("=" * 60)

    dd_age = load_primary_nhanes()
    dd_age = load_extended_nhanes(dd_age)

    # Filter out children (age < 18) before imputation — MCQ questionnaire
    # fields are 100% null for children, and mean-imputing them produces
    # meaningless predictions.
    n_before = len(dd_age)
    dd_age = dd_age[dd_age["RIDAGEYR"] >= 18].copy()
    print(f"\nFiltered to adults (age >= 18): {n_before} -> {len(dd_age)} records")

    dd_age = dd_age.fillna(dd_age.mean())

    print(f"Final dataset: {dd_age.shape[0]} records, {dd_age.shape[1]} columns")
    print(f"Age range: {dd_age['RIDAGEYR'].min():.0f} - {dd_age['RIDAGEYR'].max():.0f}")
    print(f"Mean age: {dd_age['RIDAGEYR'].mean():.1f}")

    # Validate standard model (70/30 split to match notebook)
    std_train, std_test = validate_model(
        dd_age, STANDARD_21_FEATURES, "standard", test_size=0.3,
    )

    # Validate extended model (70/30 split to match notebook)
    ext_train, ext_test = validate_model(
        dd_age, EXTENDED_35_FEATURES, "extended", test_size=0.3,
    )

    # Also validate standard with 90% test (the "loaded model" test from notebook)
    std_train_90, std_test_90 = validate_model(
        dd_age, STANDARD_21_FEATURES, "standard", test_size=0.9,
    )

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Standard 21-feat (30% test):  MAE={std_test['mae']:.2f}  R²={std_test['r2']:.4f}  Pearson={std_test['pearson_r']:.4f}")
    print(f"  Standard 21-feat (90% test):  MAE={std_test_90['mae']:.2f}  R²={std_test_90['r2']:.4f}  Pearson={std_test_90['pearson_r']:.4f}")
    print(f"  Extended 35-feat (30% test):  MAE={ext_test['mae']:.2f}  R²={ext_test['r2']:.4f}  Pearson={ext_test['pearson_r']:.4f}")

    print("\n  Expected from notebook:")
    print(f"  Standard 21-feat (30% test):  MAE=6.37  R²=0.8630  Pearson=0.9290")
    print(f"  Standard 21-feat (90% test):  MAE=5.11  R²=0.9058  Pearson=0.9518")
    print(f"  Extended 35-feat (30% test):  MAE=6.07  R²=0.8730  Pearson=0.9343")
    print("=" * 60)


if __name__ == "__main__":
    main()
