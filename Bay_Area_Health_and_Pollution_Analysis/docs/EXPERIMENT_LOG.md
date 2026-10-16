# Experiment log

## v2: draft-data audit

Original draft extract, random-vs-county diagnostics and nested county evaluation. Outputs retained in `reports/legacy`. This established the generalization problem and the need to verify provenance.

## v3.0: official Bay Area source, buffered calibration stress test

Defined `EVALUATION_PROTOCOL.md` before fitting. Used only the nine Bay Area counties. Official data ingestion revealed that asthma and PM2.5 agree with the draft extract, while many other environmental values differ. Exact reconciliation is saved.

The combined procedure selected log-target Ridge in some folds. Out-of-support environmental features caused extreme exponentiated predictions in a held-out county. The failed model's combined macro MAE was **172.20**, pooled R² **−28142.891**. This is a model robustness failure, not a reason to omit observations. Full first-run outputs are preserved in `reports/failed_extrapolation`.

## v3.1: training-derived extrapolation guard

After observing the v3.0 failure, added a model-level output constraint `[0, maximum training target]`. The maximum is learned separately inside every fitting partition, including inner CV. It never uses calibration/test labels. Every test observation remains in the metrics. The guard trades away extrapolation to unseen high target values and can underestimate high-rate counties.

**This is a post-inspection model revision.** Its re-evaluation on the same counties is explicitly exploratory and cannot be presented as an untouched confirmatory test. Keep both runs visible. No source, feature family, buffer, grid or split was revised to optimize a county score. Genuine new-data validation would require data outside the current research scope.
