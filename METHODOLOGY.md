# Methodology

This document provides a distilled overview of the design decisions, tradeoffs, and rationale behind the Healome Aging Clock. A formal methodology paper is in preparation.

## Motivation

Biological age estimation from blood biomarkers has been pursued through various approaches — linear models (Levine's PhenoAge, Klemera-Doubal), elastic nets, gradient-boosted trees, and deep learning. Each makes different tradeoffs between interpretability, accuracy, stability, and data requirements.

The design was motivated by three observations:

1. **Clinical accessibility**: CBC and CMP panels are the most commonly ordered blood tests worldwide. A model built on these biomarkers can be applied to nearly any blood draw without requiring specialized assays.

2. **Analytical stability**: Blood biomarkers, while subject to pre-analytical variation, generally show lower measurement-to-measurement variance than epigenetic markers (which depend on DNA extraction, bisulfite conversion, and array-specific batch effects). For individual-level longitudinal tracking, this matters.

3. **Reproducibility**: By training exclusively on NHANES — a freely available, well-documented public dataset — every aspect of the training pipeline can be independently audited and replicated.

## Architecture Choice

### What I tried

I experimented with a wide range of model architectures before arriving at the current models:

**Linear models**: Linear regression and elastic net were the starting point. They establish a baseline but fail to capture the nonlinear relationships between biomarkers and aging.

**Neural networks**: I tried simple fully-connected networks (3 layers, no autoencoder, no CNN), deeper architectures, and several variations. The autoencoder + CNN hybrid performed somewhat better than the simpler neural nets — the autoencoder branch acts as a regularizer by requiring the latent space to retain enough information to reconstruct all biomarkers. However, even the best neural architecture did not outperform the tree ensemble models.

**Tree ensembles**: GradientBoostingRegressor and XGBoost consistently outperformed all neural network architectures. After exhaustive hyperparameter tuning (n_estimators, max_depth, learning_rate, min_samples_split, colsample_bytree), the GradientBoosting models became the primary production models.

### Why trees won

The tree-based models achieved better accuracy on this problem likely because:
- The feature space is relatively low-dimensional (21–35 features) and tabular — a regime where gradient-boosted trees typically outperform neural networks
- Some features are categorical (medical history questions), which trees handle natively
- The ensemble approach is naturally robust to the noise and missingness in survey data

### Experimental neural net

The autoencoder + CNN architecture is preserved in `models/experimental/` for research purposes. It represents the best-performing neural architecture I found, but it still underperforms the tree models. It may be useful as a starting point for researchers exploring representation learning on biomarker data.

## Training Strategy

### Primary Model: GradientBoosting

The production models are sklearn `GradientBoostingRegressor` ensembles. I performed exhaustive hyperparameter tuning across n_estimators, max_depth, min_samples_split, learning_rate, and loss function. I also experimented with XGBoost (n_estimators=6000, max_depth=10, colsample_bytree=0.6) which performed comparably. The final hyperparameters represent the best configurations found through this search.

| Parameter | Standard (21-feat) | Extended (35-feat) |
|-----------|-------------------|-------------------|
| n_estimators | 4,000 | 6,000 |
| max_depth | 8 | 10 |
| min_samples_split | 30 | 30 |
| learning_rate | 0.01 | 0.01 |
| loss | least squares | least squares |

### Missing Data Handling

NHANES data contains missing values at variable rates across biomarkers. I handle this with mean imputation (per-feature mean computed on the training set). This is a deliberate simplicity choice:

- Mean imputation is transparent and reproducible.
- It introduces known biases (underestimates variance, can distort correlations).
- More sophisticated approaches (iterative imputation, multiple imputation) could improve performance and are welcome as community contributions.

## Feature Selection

Feature selection was performed through iterative importance-based pruning, not manual curation:

1. **Start from all available biomarkers**: I began with approximately 120 features from NHANES — the full set of lab biomarkers (CBC, CMP, lipids, inflammatory markers) and medical history questionnaire items (MCQ).

2. **Train, measure, prune**: A GradientBoostingRegressor was trained on the full feature set. Feature importances were extracted, and the least important features were removed. The model was retrained on the remaining features.

3. **Iterate until accuracy drops**: This pruning cycle was repeated. At each step, I checked whether removing the bottom features caused a meaningful drop in test-set accuracy (MAE, R²). Features were removed as long as the accuracy remained stable.

4. **35-feature extended model**: The pruning plateau was reached at approximately 35 features. Below this, removing additional features caused noticeable accuracy degradation. This became the **extended model**.

5. **21-feature standard model**: I continued pruning to find the minimal feature set that still maintained acceptable accuracy for practical use — not everyone has access to a comprehensive blood panel. The 21-feature model (16 lab biomarkers + 5 questionnaire items) represents this minimal viable set. It sacrifices some accuracy but can be computed from a standard CBC + basic metabolic panel + HbA1c.

This data-driven approach — rather than manual biomarker selection — ensures that the feature set is optimized for predictive power, not just clinical intuition.

## Target Variable

The model predicts deviation from **chronological age** (NHANES variable RIDAGEYR). This is a design choice with important implications:

- The model learns what "normal aging" looks like across the NHANES population
- Individuals who appear older than their chronological age on blood biomarkers receive higher predicted biological ages
- Whether this deviation is causally linked to health outcomes is an open research question

The survival analysis in [BENCHMARKS.md](BENCHMARKS.md) provides evidence that this deviation is associated with mortality: the univariate Cox Proportional Hazards model yields a hazard ratio of 1.098 per year of biological age (95% CI: 1.095–1.100, concordance = 0.83), and Kaplan-Meier curves show clear separation between individuals classified as accelerated aging (biological age >= chronological age + 5 years) versus decelerated aging. See the hazard ratio tables in BENCHMARKS.md for full details including confidence intervals.
