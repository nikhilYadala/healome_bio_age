# Model Card: Healome Aging Clock

## Model Overview

| Field | Value |
|-------|-------|
| **Model type** | Gradient Boosted Trees (sklearn GradientBoostingRegressor) |
| **Variants** | Standard (21 features) and Extended (35 features) |
| **Input** | Lab biomarkers (CBC + CMP) and medical history questionnaire items |
| **Output** | Estimated biological age (continuous, in years) |
| **Training data** | NHANES (CDC), ~50,000 records across 2003-2020 cycles |
| **Intended use** | Research, personal health exploration, benchmarking |
| **Not intended for** | Clinical diagnosis, medical decision-making without professional oversight |
| **Experimental** | Autoencoder + 1D CNN (PyTorch) available in `models/experimental/` |

## Architecture

### Primary: Gradient Boosted Trees

The production model uses sklearn's `GradientBoostingRegressor`, trained to predict chronological age from blood biomarkers and medical history.

**Standard model (21 features):**
- `n_estimators=4000, max_depth=8, min_samples_split=30, learning_rate=0.01`
- 16 lab biomarkers + 5 medical history questionnaire items
- Test MAE: 5.11 years, R²: 0.906

**Extended model (35 features):**
- `n_estimators=6000, max_depth=10, min_samples_split=30, learning_rate=0.01`
- 30 lab biomarkers + 5 medical history items
- Test MAE: 6.07 years, R²: 0.873

### Experimental: Autoencoder + CNN

An experimental neural network architecture is also provided in `healome_clock/models/experimental/`. It uses a two-branch design: an autoencoder compresses 42 biomarkers into a 30-dim latent space, and a 1D CNN predicts age from the latent representation. See `models/experimental/autoencoder_cnn.py` for details.

## Input Features

### Standard Model — 21 Features (16 lab + 5 questionnaire)

| NHANES Code | Name | Unit | Category |
|-------------|------|------|----------|
| LBXMCVSI | Mean Cell Volume | fL | CBC |
| LBXGH | Glycohemoglobin (HbA1c) | % | Metabolic |
| LBXSATSI | ALT | IU/L | Liver |
| LBXRBCSI | Red Blood Cell Count | million cells/uL | CBC |
| LBXPLTSI | Platelet Count | 1000 cells/uL | CBC |
| LBXSLDSI | LDH | IU/L | Metabolic |
| LBXLYPCT | Lymphocyte % | % | CBC |
| LBDLYMNO | Lymphocyte Count | 1000 cells/uL | CBC |
| LBXSCK | CPK | IU/L | Metabolic |
| LBXSCR | Creatinine | mg/dL | Kidney |
| LBXSAPSI | ALP | IU/L | Liver |
| LBXSKSI | Potassium | mmol/L | Electrolyte |
| LBXRDW | Red Cell Distribution Width | % | CBC |
| LBXMOPCT | Monocyte % | % | CBC |
| LBXSBU | Blood Urea Nitrogen | mg/dL | Kidney |
| LBXSGL | Glucose | mg/dL | Metabolic |
| MCQ220 | Cancer history | binary | Questionnaire |
| MCQ160D | Angina history | binary | Questionnaire |
| MCQ160A | Arthritis history | binary | Questionnaire |
| MCQ500 | Liver condition history | binary | Questionnaire |
| MCQ550 | Gallstones history | binary | Questionnaire |

### Extended Model — 35 Features (adds 14 markers)

Includes all 21 standard features plus:

| NHANES Code | Name | Unit |
|-------------|------|------|
| LBXHGB | Hemoglobin | g/dL |
| LBXHCT | Hematocrit | % |
| LBXWBCSI | White Blood Cell Count | 1000 cells/uL |
| LBXSASSI | AST | IU/L |
| LBXSTB | Total Bilirubin | mg/dL |
| LBXSCA | Total Calcium | mg/dL |
| LBXSCLSI | Chloride | mmol/L |
| LBDLDLM | LDL Cholesterol (Martin-Hopkins) | mg/dL |
| LBDHDD | Direct HDL Cholesterol | mg/dL |
| LBXTR | Triglycerides | mg/dL |
| LBXSTP | Total Protein | g/dL |
| LBXSC3SI | Bicarbonate | mmol/L |
| LBXSOSSI | Osmolality | mmol/Kg |
| LBXSNASI | Sodium | mmol/L |

## Training Data

| Field | Value |
|-------|-------|
| Dataset | NHANES (National Health and Nutrition Examination Survey) |
| Source | [CDC NHANES](https://www.cdc.gov/nchs/nhanes/index.htm) |
| Years included | 1999–2020 (multiple survey cycles) |
| Sample size after preprocessing | ~50,000 records |
| Target variable | Chronological age (RIDAGEYR) |
| Public availability | Yes |

### Data sources within NHANES

Laboratory data was merged from the Biochemistry Profile (BIOPRO) and Complete Blood Count (CBC) datasets, joined on SEQN (respondent sequence number) with demographic data providing the age target.

## Preprocessing

| Step | Method |
|------|--------|
| Missing value handling | Mean imputation across features (training set statistics) |
| Normalization | None (raw biomarker values used directly) |
| Outlier handling | None applied at preprocessing; model trained on full distribution |
| Feature engineering | None; raw NHANES biomarker values used as-is |
| Train/test split | 70/30 random split (random_state=3454) |

## Training Procedure

### Standard Model (21 features)

| Parameter | Value |
|-----------|-------|
| Model | `GradientBoostingRegressor` (sklearn) |
| n_estimators | 4,000 |
| max_depth | 8 |
| min_samples_split | 30 |
| learning_rate | 0.01 |
| loss | least squares ("ls") |
| Train/test split | 70/30, random_state=3454 |

### Extended Model (35 features)

| Parameter | Value |
|-----------|-------|
| Model | `GradientBoostingRegressor` (sklearn) |
| n_estimators | 6,000 |
| max_depth | 10 |
| min_samples_split | 30 |
| learning_rate | 0.01 |
| loss | least squares ("ls") |
| Train/test split | 70/30, random_state=3454 |

## Evaluation

### Standard Model — NHANES Test Set

| Metric | Train | Test |
|--------|-------|------|
| MAE | 4.47 years | **5.11 years** |
| R² | 0.928 | **0.906** |
| Pearson r | 0.963 | **0.952** |

### Extended Model — NHANES Test Set

| Metric | Train | Test |
|--------|-------|------|
| MAE | 2.32 years | **6.07 years** |
| R² | 0.966 | **0.873** |
| Pearson r | 0.983 | **0.934** |

### Survival Analysis

| Metric | Value |
|--------|-------|
| Univariate Cox PH HR | **1.098 per year** (95% CI: 1.095–1.100) |
| Concordance | **0.83** |

See [BENCHMARKS.md](BENCHMARKS.md) for full performance analysis including feature importance rankings and subgroup breakdowns.

## Ethical Considerations

- **Demographic bias**: NHANES oversamples certain demographic groups (Hispanic, Black, Asian, older adults, low-income) by design. Model performance may vary across subpopulations.
- **Geographic scope**: Trained on a US-representative survey. Applicability to non-US populations has not been validated.
- **Over-interpretation risk**: Users may interpret biological age as a definitive health indicator. The model estimates a statistical quantity, not a clinical diagnosis.
- **Equity**: Biomarker reference ranges vary by ancestry and sex. The model does not explicitly account for these differences beyond what is captured in the training data distribution.
