# Model card — Bay Area buffered regression

## Purpose

Research and portfolio review of historical environmental-health patterns and the reliability of geographic ML generalization. Scope is **only the nine Bay Area counties**, including fitting, tuning, calibration and evaluation. No validation supports individual clinical risk, future forecasts, causal interventions or public-health resource allocation.

## Models and evaluation

Median baseline, log-target Ridge, Extra Trees (120 trees), histogram gradient boosting (180 iterations). Small explicit grids; seed 42; early stopping disabled. Three feature families contain 4, 5 or 18 predictors. Models use median imputation with missingness indicators, standardization and regression, fitted only on the appropriate training partition.

Outer evaluation holds out one county, nine times. Two other counties are reserved for interval calibration. Fitting points are at least 5 km from calibration/test centroids, and calibration points at least 5 km from test centroids. Three inner county-grouped buffered folds tune parameters and choose family. Inner scoring averages county MAE within a fold; final primary reporting averages nine outer county MAEs. Pooled metrics and every county are also reported.

90% split-conformal ranges use absolute calibration residuals and the finite-sample `(n+1)` quantile. Lower bounds are clipped at zero. Calibration counties never enter fitting or selection. Spatial dependence and shift violate exchangeability, so nominal coverage is not guaranteed; measured coverage and width are required outputs.

## Robustness revision and evidence status

The first official-data run produced extreme exponentiated estimates from log-target Ridge. Its combined macro MAE was 172.20. Original outputs remain in `reports/failed_extrapolation`.

The v3.1 wrapper constrains outputs to `[0, maximum fitting target]`. That maximum is learned **inside each fit**, including inner CV, without calibration/test labels. This prevents catastrophic extrapolation but can underestimate genuinely higher-rate unseen areas. Every test tract remains in evaluation.

**The guard was added after observing held-out results. Revised results are exploratory, not untouched confirmatory validation.** The previous regional study also exposed some of these historical outcomes. See the experiment log.

The revised combined procedure has macro MAE **27.40**, versus baseline **27.30**, pooled R² **0.013**, and interval coverage **83.7%** versus nominal 90%. A social-only ablation has macro MAE **25.08** and pooled R² **0.246**. These comparisons do not establish a deployable model or justify selecting an ablation and claiming a fresh independent test score.

## Diagnostics

Held-out permutation importance measures predictive reliance, not causal effects. Correlated predictors can mask importance. Error bars describe county variation. Poverty slices are diagnostic, not evidence of fairness across protected groups. The county bootstrap is descriptive because there are only nine counties and training folds overlap. The learning curve uses selected hyperparameters and is not an independent evaluation. The shuffled-label control shuffles only within fitting data, keeping calibration/test labels out of training.

Feature-range exceedance and standardized mean shifts flag some support problems. Neither is a comprehensive distribution-shift detector. A 5 km centroid buffer does not guarantee boundary separation or independence of spatially modeled labels.

## Final demonstration artifact

The pipeline is fitted with two Bay Area counties reserved for calibration. `artifacts/official/model_metadata.json` records features, fitting/calibration sizes, interval radius, training ranges, source/protocol hashes and selected parameters. The artifact is not refitted on all tracts and has no fresh independent test set. Outer scores describe the fitting/selection/calibration procedure, not a new test of this artifact.

Batch inference requires all feature columns, permits missing predictors, rejects negative/infinite inputs and all-missing rows, preserves row order/metadata, flags univariate range exceedance, and returns point estimates and nominal ranges. Only load trusted local joblib files. Out-of-range inputs may still be unsupported even when a numeric estimate is returned.

## Reproduction and next evidence

`python run_bayair.py train` regenerates the evidence. Pinned dependencies, source/protocol/code hashes, complete prediction files, calibration residuals and split manifests make it inspectable. CI runs contracts and evidence checks; the full experiment is verified locally.

A confirmatory claim after the robustness revision requires new, appropriately defined observations. All current work stays within the user-requested Bay Area scope.
