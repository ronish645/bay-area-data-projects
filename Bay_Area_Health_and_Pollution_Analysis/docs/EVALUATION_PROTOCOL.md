# BayAir v3 evaluation protocol — Bay Area only

Fixed before fitting the v3 benchmark, 2026-10-15 (America/Los_Angeles). This replaces the preliminary statewide-transfer design following the user's restriction to Bay Area counties only. No statewide model was trained.

## Source and scope

Use the official CES 4.0 final workbook as a documented historical source, immediately filtering to Alameda, Contra Costa, Marin, Napa, San Francisco, San Mateo, Santa Clara, Solano and Sonoma. Only these nine counties enter descriptions, training, tuning, calibration and evaluation. The workbook is statewide; the processed dataset is Bay Area only. CES 4.0 is a fixed historical release, not a claim of current conditions or the latest release.

## Fixed evaluation choices

- Target: historical asthma emergency-department visit indicator. Exclude missing labels, never impute labels.
- Outer evaluation: leave one of the nine Bay Area counties out. Every labeled tract receives exactly one held-out prediction for each feature-family procedure.
- Spatial buffer: **5 km** between training points and held-out county points, between calibration and test points, and between fitting and calibration points, using centroid haversine distance. This is point-distance separation, not polygon-boundary separation. Five km is fixed for this densely populated regional study before seeing v3 results.
- For each outer fold, reserve two of the other eight counties for interval calibration with GroupShuffleSplit(seed=42). Use only the remaining six counties for fitting/tuning. Drop fitting/calibration points too close to the outer test county, and fitting points too close to calibration points.
- Three county-grouped, 5 km buffered inner validation folds. Tune using equally weighted mean county MAE within each validation fold, then average across the three folds. Preprocessing is fitted only within the fitting partition.
- Candidate families: median, log-target Ridge, Extra Trees, histogram gradient boosting. Small explicit grids, seed 42, no early-stopping random holdout. Predictions are constrained to nonnegative rates. **Post-inspection v3.1 amendment:** also cap predictions at the maximum training target, learned inside each fit, after v3.0 exposed extreme exponentiated extrapolation. The initial protocol and outputs are preserved in `reports/failed_extrapolation`; v3.1 results are exploratory. See `EXPERIMENT_LOG.md`.
- Fixed ablations: four air/traffic measures; five socioeconomic indicators; thirteen environmental plus five socioeconomic measures. Exclude percentiles, other health labels, CES composites, dates, identifiers and coordinates as predictors.
- Choose the model family and parameters by inner scores only. The combined procedure is the primary result. Ablation comparisons are exploratory.
- Fit the selected model on the outer fitting partition. Calibrate **90% split-conformal residual intervals** on the reserved counties with the finite-sample (n+1) quantile. Clip lower bounds at zero. No calibration/test rows enter fitting or tuning.
- Report empirical coverage and interval width overall and per county. Spatial dependence/shift breaks exchangeability; nominal coverage is not guaranteed.
- Report macro county MAE (primary), pooled MAE/RMSE/R², every county, baseline gains and fixed poverty slices. Do not discard poor folds.
- Compute held-out permutation importance only after each fold's model choices freeze. Correlated features can share predictive reliance; importance does not imply causality.
- Add a negative control by refitting the selected combined pipeline with fitting labels shuffled within that fitting partition. Report its held-out errors separately; never use it to select models.
- Save split manifests, source/code/protocol hashes, all predictions, parameter searches, calibration residuals, runtime versions, learning-curve diagnostics and data reconciliation.
- Fit the final demonstration artifact with two Bay Area counties reserved for calibration; it has no fresh independent test set. Outer metrics describe the training procedure.

## Interpretation

The Bay Area and some target values were explored in v2, so this is **not a blinded new dataset**. The new official source and stronger evaluation improve provenance and separation without erasing prior knowledge. Historical target values are spatially modeled; a 5 km point buffer cannot ensure independent clinical observations. Results are aggregate associations, not individual diagnosis, causal effects, future forecasts or a deployable public-health tool.

Do not tune against outer results and call them fresh test scores. After inspecting these results, new performance claims require genuinely new data or an explicitly exploratory label.
