# Benchmarks

## Model Performance Summary

| Model | Features | Train MAE | Train R² | Test MAE | Test R² | Test Pearson |
|-------|----------|-----------|----------|----------|---------|-------------|
| **Standard (21-feat)** | 15 lab + 6 MCQ | 4.47 | 0.928 | **5.11** | **0.906** | 0.952 |
| **Extended (35-feat)** | 26 lab + 9 MCQ | 2.32 | 0.966 | **6.07** | **0.873** | 0.934 |

Training data: ~50,000 NHANES records (2003-2020). Test split: random, seed=3454.

## 21-Feature Standard Model

### Training (70/30 split)

| Metric | Train | Test |
|--------|-------|------|
| MSE | 44.95 | 85.72 |
| R² | 0.928 | 0.863 |
| MAE | 4.47 years | 6.37 years |
| Pearson r | 0.963 | 0.929 |

### Validation (loaded model, 90% test split)

| Metric | Value |
|--------|-------|
| MSE | 58.72 |
| R² | 0.906 |
| MAE | **5.11 years** |
| Pearson r | 0.952 |

### Feature Importance (least to most important)

1. MCQ500 — Liver condition history
2. MCQ550 — Gallstones history
3. LBXSKSI — Potassium
4. LBXSAPSI — Alkaline Phosphatase
5. LBXLYPCT — Lymphocyte %
6. LBXRDW — Red cell distribution width
7. LBXMOPCT — Monocyte %
8. LBXPLTSI — Platelet count
9. LBXSCK — CPK
10. LBXSLDSI — LDH
11. LBXSCR — Creatinine
12. LBXSGL — Glucose
13. LBXSATSI — ALT
14. LBXRBCSI — Red blood cell count
15. LBDLYMNO — Lymphocyte count
16. MCQ220 — Cancer history
17. LBXMCVSI — Mean cell volume
18. LBXSBU — BUN
19. MCQ160A — Arthritis history
20. **LBXGH — Glycohemoglobin** (2nd most important)
21. **MCQ160D — Angina history** (most important)

## 35-Feature Extended Model

### Training (70/30 split)

| Metric | Train | Test |
|--------|-------|------|
| MSE | 21.42 | 79.49 |
| R² | 0.966 | 0.873 |
| MAE | 2.32 years | 6.07 years |
| Pearson r | 0.983 | 0.934 |

### Top 5 Most Important Features

1. **MCQ160D** — Angina history
2. **LBXGH** — Glycohemoglobin
3. **MCQ160A** — Arthritis history
4. **LBXSBU** — BUN
5. **LBXMCVSI** — Mean cell volume

## Survival Analysis

Survival analysis uses CDC-linked mortality data merged with the NHANES cohort (41,823 observations, 5,805 mortality events). All Cox PH models below use `lifelines.CoxPHFitter`.

### Cox Proportional Hazards — All-Cause Mortality

**Univariate Cox PH** (single covariate, no derived columns):

| Clock | HR | 95% CI | p | Concordance |
|-------|-----|--------|---|-------------|
| Healome Standard (21-feat) | 1.093 | 1.090–1.095 | < 0.005 | 0.81 |
| **Healome Extended (35-feat)** | **1.098** | **1.095–1.100** | **< 0.005** | **0.83** |
| PhenoAge (Levine 2018) | 1.071 | 1.070–1.072 | < 0.005 | 0.86 |

The Healome Clock achieves a higher per-year hazard ratio than PhenoAge (1.098 vs. 1.071), meaning each year of Healome biological age carries 9.8% additional mortality risk compared to 7.1% for PhenoAge. PhenoAge achieves higher concordance (0.86 vs. 0.83) because it was trained directly on a mortality phenotype, optimizing for rank-ordering individuals by death risk. For a clinically actionable clock, per-year HR is the more relevant metric: it directly quantifies how much a one-year reduction in biological age lowers mortality risk.

**Bivariate Cox PH** (clock + chronological age):

| Clock | Clock HR | Clock 95% CI | Chrono Age HR | Concordance |
|-------|----------|--------------|---------------|-------------|
| Healome Standard (21-feat) | 1.020 | 1.017–1.023 | 1.080 | 0.85 |
| Healome Extended (35-feat) | 1.022 | 1.018–1.025 | 1.077 | 0.85 |
| PhenoAge (Levine 2018) | 1.048 | 1.046–1.051 | 1.044 | 0.87 |

Both clocks contribute significant mortality prediction beyond chronological age alone (all p < 0.005).

### Disease-Specific Mortality

Univariate Cox PH models for each major CDC cause of death (UCOD_LEADING):

| Cause of Death | Events | HR | 95% CI | p | Concordance |
|----------------|--------|----|--------|---|-------------|
| Pneumonia and influenza | 199 | 1.131 | 1.115–1.147 | < 0.005 | 0.88 |
| Nephritis / kidney disease | 124 | 1.125 | 1.105–1.145 | < 0.005 | 0.87 |
| Unintentional injuries | 132 | 1.121 | 1.102–1.140 | < 0.005 | 0.88 |
| Heart disease | 1,504 | 1.114 | 1.109–1.120 | < 0.005 | 0.86 |
| Chronic lower resp. disease | 307 | 1.112 | 1.101–1.124 | < 0.005 | 0.86 |
| Diabetes | 315 | 1.107 | 1.096–1.118 | < 0.005 | 0.84 |
| Alzheimer's disease | 199 | 1.095 | 1.082–1.108 | < 0.005 | 0.83 |
| All other causes | 1,553 | 1.091 | 1.087–1.096 | < 0.005 | 0.81 |
| Cancer | 1,299 | 1.088 | 1.083–1.093 | < 0.005 | 0.82 |
| Cerebrovascular disease (stroke) | 173 | 1.036 | 1.026–1.046 | < 0.005 | 0.64 |

### Kaplan-Meier Survival Curves

Individuals are classified by aging rate:
- **Accelerated aging**: biological_age - chronological_age >= 5 years
- **Decelerated aging**: biological_age - chronological_age <= -5 years

The Kaplan-Meier curves show clear separation between these groups, with the decelerated aging group showing significantly better survival.

<!-- ![Kaplan-Meier Survival Curves](figures/kaplan_meier_aging_rate.png) -->

## Comparison to Other Models

| Model | Type | Features | Test MAE | Test R² | HR (univariate) | Concordance |
|-------|------|----------|----------|---------|-----------------|-------------|
| **Healome Standard** | GradientBoosting | 21 | **5.11** | **0.906** | 1.093 | 0.81 |
| **Healome Extended** | GradientBoosting | 35 | 6.07 | 0.873 | **1.098** | 0.83 |
| PhenoAge (Levine 2018) | Formula-based | 10 | — | — | 1.071 | 0.86 |

PhenoAge is implemented in `healome_clock.evaluation.phenoage` for easy benchmarking. I encourage the community to add GrimAge, DunedinPACE, and other clocks. See [benchmarks/README.md](benchmarks/README.md).

## Training Convergence

### Standard model (21 features)

- Model type: `GradientBoostingRegressor`
- n_estimators: 4,000
- max_depth: 8
- min_samples_split: 30
- learning_rate: 0.01

### Extended model (35 features)

- Model type: `GradientBoostingRegressor`
- n_estimators: 6,000
- max_depth: 10
- min_samples_split: 30
- learning_rate: 0.01
