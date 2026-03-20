# The Healome Aging Clock

**Open-source blood-based biological age estimation from standard clinical biomarkers.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilYadala/healome_bio_age/blob/main/notebooks/demo.ipynb)

---

## Core thesis

This project is built around one idea: there is a **small, fixed set of blood-derived measurements**—a **minimal clique**—that, together with the right low-bias model, captures almost all the signal that matters for **biological age** and **mortality prediction**. Hyperparameter tuning and feature ablation were aimed at finding that clique, not at maximizing complexity. Features outside this set either add little or actively **hurt** some metrics once the core signal is already represented.

## Why This Exists

Many biological age models are difficult to interpret in clinical contexts, hard to benchmark reproducibly, and show high variability on repeated measurements. I built this release to provide a transparent, reproducible baseline using routine blood biomarkers and a public dataset. The methodology, model, and evaluation are fully open so the community can inspect, compare, and improve on this work.

## Model and feature selection

We evaluated a wide range of predictors, from simple to highly flexible: **linear regression**, **elastic net**, **gradient-boosted trees**, **bagging / tree ensembles**, broader **ensembles**, **neural networks**, **CNNs**, and **variational autoencoders**. We **settled on the architectures that performed best** on our validation objectives and then focused on **feature pruning**.

Starting from roughly **120 blood-related biomarkers**, we progressively dropped inputs that did not improve (or did not stably improve) held-out accuracy. That process yielded the **35-feature extended** model. We took one further reduction step to the **21-feature standard** model: a smaller panel **without a meaningful accuracy penalty**, so routine labs stay sufficient where the extended panel is not available. On the standard model, performance remains in a strong range on at least one of the headline metrics we track.

**Beyond the extended model**, adding more markers did **not** improve metrics; in several cases it caused **clear degradation**. That pattern supports the thesis above: once the minimal clique is covered, extra features are mostly noise or redundancy for this task.

## Design Principles

- **Interpretable** — Built on standard clinical biomarkers (CBC + CMP + medical history) that map to physiological systems physicians already reason about.
- **Reproducible** — Trained entirely on open, public survey data. Anyone can retrain, validate, and audit.
- **Validated against mortality** — Survival analysis confirms the model's biological age estimates predict mortality (univariate Cox PH HR = 1.098 per year, concordance = 0.83).
- **Extensible** — Structured for community contributions: new data sources, models, and benchmarks.

## Quick Start

```bash
git clone https://github.com/nikhilYadala/healome_bio_age.git
cd healome_bio_age
pip install -e .
```

Model weights and validation data are on the [Hugging Face Hub](https://huggingface.co/Healome): [model weights](https://huggingface.co/Healome/healome-clock-weights), [validation dataset](https://huggingface.co/datasets/Healome/nhanes-validation-data). Download once (see below) or let the library fetch weights automatically when you first use a model.

### Requirements

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| numpy | >=1.21, <2 |
| pandas | >=1.3 |
| scikit-learn | >=1.0, <1.1 (models trained with 0.24.1) |
| joblib | >=1.0 |
| matplotlib | >=3.4 |

**Optional extras:**
- `pip install -e ".[survival]"` — adds lifelines (for Kaplan-Meier, Cox PH)
- `pip install -e ".[neural]"` — adds PyTorch (experimental model)

### Biomarker names (friendly keys)

`predict_age` and `HealomeClock.predict` accept **canonical snake_case names** (recommended) or the **original survey variable codes** used in training (e.g. `LBXGH`). Common synonyms work too (e.g. `hba1c_percent` → glycohemoglobin). See `healome_clock.feature_aliases` and `NHANES_TO_CANONICAL_KEY` in code for the full map.

**Questionnaire (history) fields** use `1 = Yes`, `2 = No` for the six items below, matching how the model was trained.

### Example — standard model (21 features)

```python
from healome_clock import predict_age

result = predict_age({
    "mean_cell_volume_fl": 93,                      # MCV, fL
    "glycohemoglobin_percent": 5.4,                 # HbA1c, % (glycated hemoglobin)
    "alt_iu_l": 22,                                 # ALT, IU/L
    "rbc_count_million_per_ul": 4.52,               # RBC, million cells/µL
    "ever_cancer_or_malignancy": 2,                 # 1=Yes, 2=No
    "platelet_count_thousand_per_ul": 245,          # platelets, 1000 cells/µL
    "ldh_iu_l": 138,                                # LDH, IU/L
    "ever_angina": 2,                               # 1=Yes, 2=No
    "lymphocyte_percent": 28.5,                     # % of WBCs
    "lymphocyte_count_thousand_per_ul": 2.1,        # 1000 cells/µL
    "cpk_iu_l": 132,                                # CPK, IU/L
    "creatinine_mg_dl": 0.80,                       # mg/dL
    "ever_arthritis": 2,                            # 1=Yes, 2=No
    "alp_iu_l": 68,                                 # alkaline phosphatase, IU/L
    "ever_liver_condition": 2,                      # 1=Yes, 2=No
    "potassium_mmol_l": 4.0,                        # mmol/L
    "rdw_percent": 12.8,                            # RDW, %
    "monocyte_percent": 6.2,                        # % of WBCs
    "bun_mg_dl": 15,                                # BUN, mg/dL
    "ever_gallstones": 2,                          # 1=Yes, 2=No
    "glucose_mg_dl": 95,                            # mg/dL
}, chronological_age=45, variant="standard")

print(result.summary())
```

### Example — extended model (35 features)

```python
result = predict_age({
    "mean_cell_volume_fl": 93,                      # MCV, fL
    "hemoglobin_g_dl": 14.2,                        # g/dL
    "hematocrit_percent": 42.1,                     # %
    "alt_iu_l": 22,                                 # IU/L
    "rbc_count_million_per_ul": 4.52,               # million cells/µL
    "wbc_count_thousand_per_ul": 6.1,               # 1000 cells/µL
    "platelet_count_thousand_per_ul": 245,          # 1000 cells/µL
    "ldh_iu_l": 138,                                # IU/L
    "ever_angina": 2,                               # 1=Yes, 2=No
    "lymphocyte_percent": 28.5,                     # % of WBCs
    "ast_iu_l": 24,                                 # IU/L
    "lymphocyte_count_thousand_per_ul": 2.1,        # 1000 cells/µL
    "cpk_iu_l": 132,                                # IU/L
    "total_bilirubin_mg_dl": 0.8,                   # mg/dL
    "ever_arthritis": 2,                            # 1=Yes, 2=No
    "alp_iu_l": 68,                                 # IU/L
    "calcium_mg_dl": 9.5,                           # mg/dL
    "bun_mg_dl": 15,                                # mg/dL
    "ever_gallstones": 2,                           # 1=Yes, 2=No
    "glucose_mg_dl": 95,                            # mg/dL
    "chloride_mmol_l": 102,                         # mmol/L
    "ldl_cholesterol_mg_dl": 110,                   # mg/dL (Martin–Hopkins)
    "glycohemoglobin_percent": 5.4,                 # %
    "hdl_cholesterol_mg_dl": 55,                    # mg/dL (direct HDL)
    "ever_cancer_or_malignancy": 2,                 # 1=Yes, 2=No
    "triglycerides_mg_dl": 90,                      # mg/dL
    "total_protein_g_dl": 7.0,                      # g/dL
    "creatinine_mg_dl": 0.80,                       # mg/dL
    "ever_liver_condition": 2,                      # 1=Yes, 2=No
    "potassium_mmol_l": 4.0,                        # mmol/L
    "bicarbonate_mmol_l": 25,                       # mmol/L
    "osmolality_mmol_kg": 280,                      # mmol/kg
    "rdw_percent": 12.8,                            # %
    "monocyte_percent": 6.2,                        # % of WBCs
    "sodium_mmol_l": 140,                           # mmol/L
}, chronological_age=45, variant="extended")

print(result.summary())
```

CSV/JSON column names can use these same friendly keys (or the original survey codes).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilYadala/healome_bio_age/blob/main/notebooks/demo.ipynb)

## Two Model Variants

| Variant | Features | Test MAE | Test R² | Pearson r |
|---------|----------|----------|---------|-----------|
| **Standard** | 21 (15 lab + 6 questionnaire) | 5.11 years | 0.906 | 0.952 |
| **Extended** | 35 (expanded lab panel) | 6.07 years | 0.873 | 0.934 |

Both models: GradientBoosting trained on ~50K public survey records (2003–2020), validated with Cox PH survival analysis (univariate HR = 1.098 per year for extended model, concordance = 0.83).

Use **standard** for routine panels (21 inputs) or **extended** when you have the larger lipid/liver/electrolyte set (35 inputs). Pass biomarkers as **canonical friendly names** (see examples above) or the original survey codes.

Models load from `healome_clock/models/weights/` (standard_21feat.joblib, extended_35feat.joblib). If the files are missing, the library will try to download them from the Hub; otherwise see [Downloading model weights and validation data](#downloading-model-weights-and-validation-data) below.

```python
from healome_clock import HealomeClock

# Standard model (15 blood markers + 6 medical history questions)
clock = HealomeClock(variant="standard")

# Extended model (35 features for comprehensive panels)
clock = HealomeClock(variant="extended")
```

## Downloading model weights and validation data

Assets live on the **Hugging Face Hub** under [Healome](https://huggingface.co/Healome). If a link below does not open, use the org page → **Models** / **Datasets** and select the matching repo.

| Resource | Hugging Face | Local path (after download) |
|----------|--------------|-----------------------------|
| **Model weights** | [`Healome/healome-clock-weights`](https://huggingface.co/Healome/healome-clock-weights) | `healome_clock/models/weights/` |
| **Validation data** (XPT bundle for `tests/validate_on_nhanes.py`) | [`Healome/nhanes-validation-data`](https://huggingface.co/datasets/Healome/nhanes-validation-data) · [files](https://huggingface.co/datasets/Healome/nhanes-validation-data/tree/main) | `nhanes_data_dump/` |

### Model weights (standard_21feat.joblib, extended_35feat.joblib)

**Option A — automatic:** If you have `huggingface_hub` installed, the library will download missing weights from the Hub the first time you use `HealomeClock` or `predict_age`.

**Option B — Python:**
```python
from huggingface_hub import hf_hub_download
from pathlib import Path

weights_dir = Path("healome_clock/models/weights")
weights_dir.mkdir(parents=True, exist_ok=True)
for name in ["standard_21feat.joblib", "extended_35feat.joblib"]:
    hf_hub_download(repo_id="Healome/healome-clock-weights", filename=name, local_dir=weights_dir)
```

**Option C — CLI:**
```bash
huggingface-cli download Healome/healome-clock-weights standard_21feat.joblib --local-dir healome_clock/models/weights
huggingface-cli download Healome/healome-clock-weights extended_35feat.joblib --local-dir healome_clock/models/weights
```

### Validation data (`nhanes_data_dump`)

To run `tests/validate_on_nhanes.py`, download the dataset snapshot into the repo root:

```bash
huggingface-cli download Healome/nhanes-validation-data --local-dir nhanes_data_dump --repo-type dataset
```

Or in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Healome/nhanes-validation-data", repo_type="dataset", local_dir="nhanes_data_dump")
```

Place the contents so that `nhanes_data_dump/2017-2020/` and (optionally) `nhanes_data_dump/extended_data/` match the structure described in `nhanes_data_dump/README.md`.

**Maintainers:** To upload or update Hub assets, use `python scripts/upload_to_huggingface.py` (weights) or add `--dataset` for the validation bundle. Requires `huggingface_hub` and `huggingface-cli login`.

## Training Data

Trained on approximately 50,000 records from public survey cycles 2003–2020. See [MODEL_CARD.md](MODEL_CARD.md) and [DATASET_FACTS.md](DATASET_FACTS.md) for details and primary sources.

## Internal Validation

Validated against a proprietary longitudinal clinical dataset (~1.5M blood-test records). Summary statistics and validation results are in [DATASET_FACTS.md](DATASET_FACTS.md). Raw data cannot be shared for patient privacy reasons.

## Survival Analysis

The model's biological age predictions are validated against mortality outcomes using linked public mortality records:

- **Cox PH**: Univariate HR = 1.098 per year of biological age (95% CI: 1.095–1.100), concordance = 0.83 — biological age is a strong predictor of mortality
- **Kaplan-Meier**: Clear separation between accelerated aging (bio_age - chrono_age >= 5 years) and decelerated aging groups

See [BENCHMARKS.md](BENCHMARKS.md) for full results.

## Benchmarking & Leaderboard

This repo maintains a dual-track leaderboard for community benchmarking:

1. **Track 1**: Age prediction accuracy (MAE, R², RMSE)
2. **Track 2**: Mortality prediction (Cox PH concordance, Kaplan-Meier)

See [benchmarks/README.md](benchmarks/README.md) for how to submit your model.

## Repo Structure

```
├── healome_clock/
│   ├── models/              # Tree-based (primary) + experimental neural net
│   ├── data/                # Public survey + mortality data loaders
│   ├── evaluation/          # Metrics, survival analysis, leaderboard
│   ├── inference.py         # Main prediction API
│   └── visualization.py     # Plotting utilities
├── notebooks/
│   ├── demo.ipynb           # Quick start (2 min)
│   ├── training.ipynb       # Full training pipeline
│   └── evaluation.ipynb     # Benchmarking walkthrough
├── benchmarks/              # Leaderboard submissions
├── data/                    # Sample input data
└── figures/                 # Generated plots
```

## Limitations

Read [LIMITATIONS.md](LIMITATIONS.md) before using. Key points:
- Trained on a US-representative survey; other populations not validated
- Estimates overall biological age; no organ-specific resolution
- Not a medical device; not for clinical decision-making without oversight

## Future Directions

- Extended model using proprietary longitudinal clinical dataset
- Organ-specific biological age estimation
- Methodology paper

Star this repo or follow [@Healome](https://twitter.com/nikhilyadala) for updates.

## About Healome

Healome is a longevity-focused health technology company building blood-based aging models and tools for longitudinal health tracking. Learn more at [healome.ai](https://healome.ai).

## Citation

```bibtex
@article{yadala2026healome,
  title={Healome Clock: An Open-Source, Interpretable, and Actionable Blood-Based Biological Aging Clock with Disease-Specific Mortality Validation},
  author={Yadala, Nikhil},
  journal={medRxiv},
  year={2026},
  doi={10.1101/2026.XX.XX.XXXXXXX},
  url={https://doi.org/10.1101/2026.XX.XX.XXXXXXX}
}
```

## Contributing

I believe the field benefits from better benchmarking and open scrutiny. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## License

Apache License 2.0. See [LICENSE](LICENSE).
