"""
Map canonical friendly biomarker keys (and synonyms) to NHANES variable codes.

Models were trained on NHANES column names; this layer lets callers use
snake_case clinical names instead. NHANES codes are still accepted.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from healome_clock.models.tree import MODEL_CONFIGS

# Primary friendly key per NHANES code (snake_case). Used in docs and optional APIs.
NHANES_TO_CANONICAL_KEY: Dict[str, str] = {
    "LBXMCVSI": "mean_cell_volume_fl",
    "LBXGH": "glycohemoglobin_percent",
    "LBXSATSI": "alt_iu_l",
    "LBXRBCSI": "rbc_count_million_per_ul",
    "MCQ220": "ever_cancer_or_malignancy",
    "LBXPLTSI": "platelet_count_thousand_per_ul",
    "LBXSLDSI": "ldh_iu_l",
    "MCQ160D": "ever_angina",
    "LBXLYPCT": "lymphocyte_percent",
    "LBDLYMNO": "lymphocyte_count_thousand_per_ul",
    "LBXSCK": "cpk_iu_l",
    "LBXSCR": "creatinine_mg_dl",
    "MCQ160A": "ever_arthritis",
    "LBXSAPSI": "alp_iu_l",
    "MCQ500": "ever_liver_condition",
    "LBXSKSI": "potassium_mmol_l",
    "LBXRDW": "rdw_percent",
    "LBXMOPCT": "monocyte_percent",
    "LBXSBU": "bun_mg_dl",
    "MCQ550": "ever_gallstones",
    "LBXSGL": "glucose_mg_dl",
    "LBXHGB": "hemoglobin_g_dl",
    "LBXHCT": "hematocrit_percent",
    "LBXWBCSI": "wbc_count_thousand_per_ul",
    "LBXSASSI": "ast_iu_l",
    "LBXSTB": "total_bilirubin_mg_dl",
    "LBXSCA": "calcium_mg_dl",
    "LBXSCLSI": "chloride_mmol_l",
    "LBDLDLM": "ldl_cholesterol_mg_dl",
    "LBDHDD": "hdl_cholesterol_mg_dl",
    "LBXTR": "triglycerides_mg_dl",
    "LBXSTP": "total_protein_g_dl",
    "LBXSC3SI": "bicarbonate_mmol_l",
    "LBXSOSSI": "osmolality_mmol_kg",
    "LBXSNASI": "sodium_mmol_l",
}

# Optional synonyms (lowercase) -> NHANES code
EXTRA_ALIASES_LOWER: Dict[str, str] = {
    "hba1c_percent": "LBXGH",
    "hba1c": "LBXGH",
    "glycohemoglobin": "LBXGH",
    "alt": "LBXSATSI",
    "alp": "LBXSAPSI",
    "ast": "LBXSASSI",
    "cpk": "LBXSCK",
    "ck": "LBXSCK",
    "ldl": "LBDLDLM",
    "hdl": "LBDHDD",
    "mcv": "LBXMCVSI",
    "rbc": "LBXRBCSI",
    "wbc": "LBXWBCSI",
    "plt": "LBXPLTSI",
    "platelets": "LBXPLTSI",
    "bun": "LBXSBU",
}


# Human-readable unit / coding notes for documentation (aligned with NHANES training)
NHANES_UNITS_DOC: Dict[str, str] = {
    "LBXMCVSI": "fL",
    "LBXGH": "% (glycated hemoglobin)",
    "LBXSATSI": "IU/L",
    "LBXRBCSI": "million cells/µL",
    "MCQ220": "1=Yes, 2=No (ever told had cancer/malignancy)",
    "LBXPLTSI": "1000 cells/µL",
    "LBXSLDSI": "IU/L",
    "MCQ160D": "1=Yes, 2=No (ever told had angina)",
    "LBXLYPCT": "% of WBCs",
    "LBDLYMNO": "1000 cells/µL",
    "LBXSCK": "IU/L",
    "LBXSCR": "mg/dL",
    "MCQ160A": "1=Yes, 2=No (ever told had arthritis)",
    "LBXSAPSI": "IU/L",
    "MCQ500": "1=Yes, 2=No (ever told had liver condition)",
    "LBXSKSI": "mmol/L",
    "LBXRDW": "%",
    "LBXMOPCT": "% of WBCs",
    "LBXSBU": "mg/dL",
    "MCQ550": "1=Yes, 2=No (ever told had gallstones)",
    "LBXSGL": "mg/dL",
    "LBXHGB": "g/dL",
    "LBXHCT": "%",
    "LBXWBCSI": "1000 cells/µL",
    "LBXSASSI": "IU/L",
    "LBXSTB": "mg/dL",
    "LBXSCA": "mg/dL",
    "LBXSCLSI": "mmol/L",
    "LBDLDLM": "mg/dL (LDL, Martin–Hopkins)",
    "LBDHDD": "mg/dL (direct HDL)",
    "LBXTR": "mg/dL",
    "LBXSTP": "g/dL",
    "LBXSC3SI": "mmol/L",
    "LBXSOSSI": "mmol/kg",
    "LBXSNASI": "mmol/L",
}


def _normalize_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_")


def _build_alias_lookup_table() -> Dict[str, str]:
    """Lowercase key -> NHANES code (all known aliases)."""
    table: Dict[str, str] = {}
    for nhanes, canonical in NHANES_TO_CANONICAL_KEY.items():
        table[_normalize_key(canonical)] = nhanes
        table[_normalize_key(nhanes)] = nhanes
    for alias, nhanes in EXTRA_ALIASES_LOWER.items():
        table[_normalize_key(alias)] = nhanes
    return table


_ALIAS_LOOKUP = _build_alias_lookup_table()


def resolve_key_to_nhanes(key: str, allowed_nhanes: Set[str]) -> Optional[str]:
    """
    Map a single column/dict key to an NHANES code if it belongs to this variant.

    Returns None if the key is unknown or not used by this variant.
    """
    raw = str(key).strip()
    if raw in allowed_nhanes:
        return raw
    nk = _normalize_key(raw)
    if nk in _ALIAS_LOOKUP:
        nhanes = _ALIAS_LOOKUP[nk]
        if nhanes in allowed_nhanes:
            return nhanes
        return None
    return None


def normalize_blood_panel_to_nhanes(
    blood_panel: Union[pd.DataFrame, Dict],
    variant: str,
    *,
    warn_on_unknown: bool = True,
) -> Union[pd.DataFrame, Dict]:
    """
    Rename friendly keys / columns to NHANES codes expected by the model.

    - NHANES codes pass through unchanged.
    - Unknown keys in a dict are skipped (optional warning).
    - DataFrame: unknown columns are left unchanged (may be ignored by the model).
    """
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(MODEL_CONFIGS)}")
    allowed: Set[str] = set(MODEL_CONFIGS[variant]["features"])

    if isinstance(blood_panel, dict):
        out: Dict[str, float] = {}
        seen: Dict[str, float] = {}
        for k, v in blood_panel.items():
            nhanes = resolve_key_to_nhanes(k, allowed)
            if nhanes is None:
                if warn_on_unknown:
                    warnings.warn(
                        f"Ignoring unknown or variant-inapplicable key: {k!r}",
                        UserWarning,
                        stacklevel=2,
                    )
                continue
            if nhanes in seen and seen[nhanes] != v:
                raise ValueError(
                    f"Conflicting values for {nhanes}: {seen[nhanes]!r} vs {v!r} "
                    f"(from keys {k!r} and earlier duplicate target)"
                )
            seen[nhanes] = v
            out[nhanes] = v
        return out

    df = blood_panel.copy()
    pending: List[tuple] = []
    for col in list(df.columns):
        if col in allowed:
            continue
        nhanes = resolve_key_to_nhanes(col, allowed)
        if nhanes is None:
            continue
        pending.append((col, nhanes))

    for old, nhanes in pending:
        if old == nhanes:
            continue
        if nhanes not in df.columns:
            df = df.rename(columns={old: nhanes})
        else:
            # Friendly alias duplicates existing NHANES column: must agree where both defined
            both = df[old].notna() & df[nhanes].notna()
            if both.any() and not (df.loc[both, old] == df.loc[both, nhanes]).all():
                raise ValueError(
                    f"Column {old!r} maps to {nhanes} but values disagree with the existing "
                    f"{nhanes!r} column"
                )
            df = df.drop(columns=[old])

    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def list_friendly_features_for_variant(variant: str) -> List[tuple]:
    """(canonical_key, nhanes_code, units_doc) for each model feature in order."""
    feats = MODEL_CONFIGS[variant]["features"]
    rows = []
    for n in feats:
        canon = NHANES_TO_CANONICAL_KEY.get(n, n.lower())
        unit = NHANES_UNITS_DOC.get(n, "")
        rows.append((canon, n, unit))
    return rows
