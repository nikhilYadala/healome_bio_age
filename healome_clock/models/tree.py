"""
Tree-based biological age model (GradientBoostingRegressor).

Two variants:
  - standard (21 features): 16 lab biomarkers + 5 MCQ questionnaire items
  - extended (35 features): adds lipids, liver, hematologic markers

Feature order is critical — .joblib models expect features in the exact
order they were trained on.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Union, Optional, Dict, List
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
HF_WEIGHTS_REPO = "Healome/healome-clock-weights"

# --- Exact feature orders as used during training (order matters for .joblib) ---

STANDARD_21_FEATURES: List[str] = [
    "LBXMCVSI",  # Mean cell volume (fL)
    "LBXGH",     # Glycohemoglobin (%)
    "LBXSATSI",  # Alanine Aminotransferase / ALT (IU/L)
    "LBXRBCSI",  # Red blood cell count (million cells/uL)
    "MCQ220",    # Ever told you had cancer or malignancy
    "LBXPLTSI",  # Platelet count (1000 cells/uL)
    "LBXSLDSI",  # Lactate Dehydrogenase / LDH (IU/L)
    "MCQ160D",   # Ever told you had angina/angina pectoris
    "LBXLYPCT",  # Lymphocyte percent (%)
    "LBDLYMNO",  # Lymphocyte number (1000 cells/uL)
    "LBXSCK",    # Creatine Phosphokinase / CPK (IU/L)
    "LBXSCR",    # Creatinine (mg/dL)
    "MCQ160A",   # Doctor ever said you had arthritis
    "LBXSAPSI",  # Alkaline Phosphatase / ALP (IU/L)
    "MCQ500",    # Ever told you had any liver condition
    "LBXSKSI",   # Potassium (mmol/L)
    "LBXRDW",    # Red cell distribution width (%)
    "LBXMOPCT",  # Monocyte percent (%)
    "LBXSBU",    # Blood Urea Nitrogen (mg/dL)
    "MCQ550",    # Has DR ever said you have gallstones
    "LBXSGL",    # Glucose (mg/dL)
]

EXTENDED_35_FEATURES: List[str] = [
    "LBXMCVSI",  # Mean cell volume (fL)
    "LBXHGB",    # Hemoglobin (g/dL)
    "LBXHCT",    # Hematocrit (%)
    "LBXSATSI",  # ALT (IU/L)
    "LBXRBCSI",  # Red blood cell count (million cells/uL)
    "LBXWBCSI",  # White blood cell count (1000 cells/uL)
    "LBXPLTSI",  # Platelet count (1000 cells/uL)
    "LBXSLDSI",  # LDH (IU/L)
    "MCQ160D",   # Ever told you had angina
    "LBXLYPCT",  # Lymphocyte percent (%)
    "LBXSASSI",  # AST (IU/L)
    "LBDLYMNO",  # Lymphocyte number (1000 cells/uL)
    "LBXSCK",    # CPK (IU/L)
    "LBXSTB",    # Total Bilirubin (mg/dL)
    "MCQ160A",   # Doctor ever said you had arthritis
    "LBXSAPSI",  # ALP (IU/L)
    "LBXSCA",    # Total Calcium (mg/dL)
    "LBXSBU",    # Blood Urea Nitrogen (mg/dL)
    "MCQ550",    # Has DR ever said you have gallstones
    "LBXSGL",    # Glucose (mg/dL)
    "LBXSCLSI",  # Chloride (mmol/L)
    "LBDLDLM",   # LDL-Cholesterol, Martin-Hopkins (mg/dL)
    "LBXGH",     # Glycohemoglobin (%)
    "LBDHDD",    # Direct HDL-Cholesterol (mg/dL)
    "MCQ220",    # Ever told you had cancer or malignancy
    "LBXTR",     # Triglyceride (mg/dL)
    "LBXSTP",    # Total Protein (g/dL)
    "LBXSCR",    # Creatinine (mg/dL)
    "MCQ500",    # Ever told you had any liver condition
    "LBXSKSI",   # Potassium (mmol/L)
    "LBXSC3SI",  # Bicarbonate (mmol/L)
    "LBXSOSSI",  # Osmolality (mmol/Kg)
    "LBXRDW",    # Red cell distribution width (%)
    "LBXMOPCT",  # Monocyte percent (%)
    "LBXSNASI",  # Sodium (mmol/L)
]

# Feature importance ranking (least → most important), from notebook outputs
STANDARD_21_IMPORTANCE_ORDER: List[str] = [
    "MCQ500", "MCQ550", "LBXSKSI", "LBXSAPSI", "LBXLYPCT", "LBXRDW",
    "LBXMOPCT", "LBXPLTSI", "LBXSCK", "LBXSLDSI", "LBXSCR", "LBXSGL",
    "LBXSATSI", "LBXRBCSI", "LBDLYMNO", "MCQ220", "LBXMCVSI", "LBXSBU",
    "MCQ160A", "LBXGH", "MCQ160D",
]

EXTENDED_35_IMPORTANCE_ORDER: List[str] = [
    "MCQ550", "MCQ500", "LBDLDLM", "LBXSNASI", "LBXSCLSI", "LBXHGB",
    "LBXSCA", "LBXHCT", "LBXSKSI", "LBXSTB", "LBXSTP", "LBXSOSSI",
    "LBXWBCSI", "LBXSASSI", "LBXTR", "LBXSC3SI", "LBXLYPCT", "LBXMOPCT",
    "LBXSAPSI", "LBXSCK", "LBXRDW", "LBXPLTSI", "LBXSLDSI", "LBXSCR",
    "LBXRBCSI", "LBDHDD", "LBDLYMNO", "LBXSGL", "LBXSATSI", "MCQ220",
    "LBXMCVSI", "LBXSBU", "MCQ160A", "LBXGH", "MCQ160D",
]

FEATURE_NAMES: Dict[str, str] = {
    "LBXMCVSI": "Mean Cell Volume", "LBXGH": "Glycohemoglobin",
    "LBXSATSI": "ALT", "LBXRBCSI": "Red Blood Cell Count",
    "MCQ220": "Cancer History", "LBXPLTSI": "Platelet Count",
    "LBXSLDSI": "LDH", "MCQ160D": "Angina History",
    "LBXLYPCT": "Lymphocyte %", "LBDLYMNO": "Lymphocyte Count",
    "LBXSCK": "CPK", "LBXSCR": "Creatinine",
    "MCQ160A": "Arthritis History", "LBXSAPSI": "ALP",
    "MCQ500": "Liver Condition History", "LBXSKSI": "Potassium",
    "LBXRDW": "RDW", "LBXMOPCT": "Monocyte %",
    "LBXSBU": "BUN", "MCQ550": "Gallstones History",
    "LBXSGL": "Glucose", "LBXHGB": "Hemoglobin",
    "LBXHCT": "Hematocrit", "LBXWBCSI": "WBC Count",
    "LBXSASSI": "AST", "LBXSTB": "Total Bilirubin",
    "LBXSCA": "Total Calcium", "LBXSCLSI": "Chloride",
    "LBDLDLM": "LDL Cholesterol", "LBDHDD": "HDL Cholesterol",
    "LBXTR": "Triglycerides", "LBXSTP": "Total Protein",
    "LBXSC3SI": "Bicarbonate", "LBXSOSSI": "Osmolality",
    "LBXSNASI": "Sodium", "MCQ160C": "CHD History",
    "MCQ092": "Blood Transfusion", "OSQ230": "Metal Objects",
}

MODEL_CONFIGS = {
    "standard": {
        "features": STANDARD_21_FEATURES,
        "importance_order": STANDARD_21_IMPORTANCE_ORDER,
        "weights_file": "standard_21feat.joblib",
        "n_features": 21,
        "description": "21-feature model (16 lab biomarkers + 5 questionnaire)",
        "training_params": {
            "model_type": "GradientBoostingRegressor",
            "n_estimators": 4000, "max_depth": 8,
            "min_samples_split": 30, "learning_rate": 0.01, "loss": "ls",
        },
        "metrics": {
            "test_r2": 0.906, "test_mae": 5.11, "test_pearson": 0.952,
            "train_r2": 0.928, "train_mae": 4.47,
        },
    },
    "extended": {
        "features": EXTENDED_35_FEATURES,
        "importance_order": EXTENDED_35_IMPORTANCE_ORDER,
        "weights_file": "extended_35feat.joblib",
        "n_features": 35,
        "description": "35-feature model (extended lab panel + questionnaire)",
        "training_params": {
            "model_type": "GradientBoostingRegressor",
            "n_estimators": 6000, "max_depth": 10,
            "min_samples_split": 30, "learning_rate": 0.01, "loss": "ls",
        },
        "metrics": {
            "test_r2": 0.873, "test_mae": 6.07, "test_pearson": 0.934,
            "train_r2": 0.966, "train_mae": 2.32,
        },
    },
}


class TreeModel:
    """
    Wrapper for trained GradientBoosting biological age models.

    Loads a .joblib model and ensures features are provided in the
    exact order the model was trained on.
    """

    def __init__(self, variant: str = "standard", model_path: Optional[Union[str, Path]] = None):
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(MODEL_CONFIGS)}")

        self.variant = variant
        self.config = MODEL_CONFIGS[variant]
        self.features = self.config["features"]
        self.n_features = self.config["n_features"]

        if model_path:
            self._model_path = Path(model_path)
        else:
            self._model_path = WEIGHTS_DIR / self.config["weights_file"]

        self._model = None

    def _load(self):
        if not self._model_path.exists():
            self._download_weights()
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self._model_path}. "
                f"Expected file: {self.config['weights_file']}. "
                f"Download from https://huggingface.co/{HF_WEIGHTS_REPO} or install with: pip install huggingface_hub"
            )
        self._model = joblib.load(self._model_path)

    def _download_weights(self) -> None:
        """Try to download missing weights from the Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return
        filename = self.config["weights_file"]
        try:
            hf_hub_download(
                repo_id=HF_WEIGHTS_REPO,
                filename=filename,
                local_dir=WEIGHTS_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception:
            pass

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and order features from a DataFrame to match training order.

        Missing features are filled with the column mean (matching the
        training-time imputation strategy of dd_age.fillna(dd_age.mean())).
        """
        for col in self.features:
            if col not in df.columns:
                df = df.copy()
                df[col] = np.nan

        ordered = df[self.features].copy()
        ordered = ordered.fillna(ordered.mean())

        remaining_nans = ordered.isna().any(axis=1)
        if remaining_nans.any():
            ordered = ordered.fillna(0)

        return ordered.values

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict biological age from a DataFrame with biomarker columns."""
        x = self.prepare_features(df)
        return self.model.predict(x)

    def predict_single(self, biomarkers: Dict[str, float]) -> float:
        """Predict biological age from a dict of biomarker values (NHANES codes or friendly names)."""
        from healome_clock.feature_aliases import normalize_blood_panel_to_nhanes

        normalized = normalize_blood_panel_to_nhanes(biomarkers, self.variant)
        df = pd.DataFrame([normalized])
        return float(self.predict(df)[0])

    def feature_importances(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.features,
            "name": [FEATURE_NAMES.get(f, f) for f in self.features],
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def __repr__(self):
        return (
            f"TreeModel(variant='{self.variant}', "
            f"n_features={self.n_features}, "
            f"test_mae={self.config['metrics']['test_mae']})"
        )
