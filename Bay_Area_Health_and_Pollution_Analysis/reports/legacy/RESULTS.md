# Experiment results

Generated from `b60352ee5854698e85ccfbce2bd11209b405d0eb57977e21c65252f358e4420c` using seed 42. See `run.json` for exact versions.

**Primary result:** the inner-selected combined-feature procedure achieves macro county MAE **26.19**, versus **26.31** for the training-median baseline: **0.5%** lower MAE. Pooled out-of-fold R² is **0.145**. Pooled and equal-county metrics weight geography differently.

## Predeclared feature ablation

| Features | Macro county MAE | Pooled R² | Worst county MAE |
|---|---:|---:|---:|
| air | 26.31 | -0.154 | 69.07 |
| social | 24.46 | 0.273 | 45.92 |
| combined | 26.19 | 0.145 | 46.84 |

## Held-out counties (combined, inner-selected)

| County | Tracts | MAE | R² | Selected family |
|---|---:|---:|---:|---|
| Solano | 96 | 46.84 | -0.283 | ridge |
| Contra Costa | 207 | 38.27 | -0.319 | median |
| Napa | 40 | 32.22 | -11.630 | ridge |
| Alameda | 360 | 26.51 | 0.281 | ridge |
| Santa Clara | 372 | 22.39 | -1.008 | ridge |
| San Francisco | 195 | 21.55 | -0.180 | ridge |
| Marin | 55 | 21.45 | -5.539 | ridge |
| San Mateo | 156 | 13.45 | -0.121 | ridge |
| Sonoma | 99 | 13.04 | -0.954 | ridge |

## Interpretation

The feature ablation is observational: adding socioeconomic variables may improve prediction without establishing causation. Individual county R² can be negative, even when pooled R² is positive. Equal-county MAE is the primary metric; large counties otherwise dominate the pooled score. The nested-selected row evaluates model-family selection inside each training fold. Fixed-family rows are exploratory comparisons and must not be used to select a family and claim an unbiased final score on these same folds.

The separate `split_diagnostic.csv` compares a fixed, untuned boosting model under random five-fold and county-held-out splits. Training sizes differ, so that gap is not a pure estimate of spatial leakage. County boundaries do not create a spatial buffer; nearby tracts across a border and spatially modeled labels can still share information.

Permutation importance uses held-out counties (five shuffles per feature). Correlated predictors can share or mask importance. The error bars describe county variation, not inferential confidence intervals. The county bootstrap range in `stability.json` is descriptive: just nine counties and overlapping training sets limit uncertainty estimates.

The final full-data model is a demonstration artifact, with no fresh independent test set. No result is a forecast, individual diagnosis, causal pollution effect, or validation against current clinical outcomes. The checked-in extract has DRAFT CES columns and undocumented dates; its exact correspondence to the final official release remains unverified.
