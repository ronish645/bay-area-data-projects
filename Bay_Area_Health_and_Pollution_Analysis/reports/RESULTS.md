# Bay Area findings — official CES 4.0 final source

## Regional questions

- Highest mean PM2.5: **Napa (9.04 µg/m³)**.
- Highest mean ozone: **Solano (0.04014 ppm)**.
- Highest mean diesel PM: **San Francisco (0.719 tons/year/km²)**.
- Highest mean tract-level asthma ED indicator: **Solano (105.95 age-adjusted visits per 10,000)**.

These are unweighted tract averages in a historical release, not official aggregated county rates or current conditions. One tract has no asthma label. Pollution burden is a broader environmental composite, not an air-only measurement.

## Buffered nested evaluation

Combined selected procedure: macro county MAE **27.40**; median baseline **27.30**. Pooled R² **0.013**. Nominal 90% intervals cover **83.7%** of held-out observations; mean width **82.08** indicator units.

All training, tuning, calibration and testing use only the nine Bay Area counties. Each outer test county is excluded from every fit operation, and fitting/calibration/test points are separated by 5 km. Three county-grouped buffered inner folds select model family/parameters. Two other counties supply interval calibration; they are not used in fitting/tuning. This is a harder and smaller-data task than v2; score changes cannot be attributed to one change alone.

| Feature family | Macro county MAE | Pooled R² | Empirical interval coverage |
|---|---:|---:|---:|
| air | 27.98 | -0.086 | 85.6% |
| social | 25.08 | 0.246 | 84.5% |
| combined | 27.40 | 0.013 | 83.7% |

## Every held-out county, combined procedure

| County | Model family | MAE | R² | Coverage |
|---|---|---:|---:|---:|
| Solano | log_ridge | 68.29 | -1.307 | 65.6% |
| Contra Costa | log_ridge | 36.71 | -0.148 | 65.2% |
| Alameda | log_ridge | 29.83 | 0.081 | 65.6% |
| Napa | log_ridge | 27.84 | -11.666 | 85.0% |
| Marin | log_ridge | 23.13 | -6.724 | 96.4% |
| San Francisco | log_ridge | 19.54 | -0.204 | 93.8% |
| Santa Clara | log_ridge | 18.18 | -0.901 | 97.6% |
| Sonoma | boosting | 11.74 | -0.504 | 100.0% |
| San Mateo | log_ridge | 11.35 | 0.255 | 100.0% |

## Evaluation status: exploratory robustness revision

The first official-data run exposed extreme log-target extrapolation (combined macro MAE 172.20). The model was then constrained to the range [0, maximum fitting target], learned independently inside each fit. First-run outputs remain in `failed_extrapolation/`. Because the guard was added after inspecting held-out results, the revised scores are exploratory, not untouched confirmatory evidence. No observation was removed from scoring. See `docs/EXPERIMENT_LOG.md` for the revision rationale.

## What these results mean

Point predictions and interval calibration face geographic shift. Nominal 90% split-conformal coverage is not guaranteed for dependent, nonexchangeable counties; measured coverage is the relevant diagnostic. Wide intervals or undercoverage are findings, not reasons to hide a fold. The bootstrap in `stability.json` is descriptive because training folds overlap and only nine counties exist.

Permutation importance measures predictive reliance, not causal effects. Socioeconomic slices do not establish clinical fairness. The learning curve uses fixed hyperparameters selected on the same development data and is a diagnostic, not an independent performance estimate. The shuffled-label control refits with fitting labels shuffled only within each fitting partition.

The original draft extract and v2 outputs remain under `reports/legacy`; reconciliation quantifies final-release differences. A new official source does not make already explored regional outcomes blinded. No result here establishes individual clinical risk, pollution causation, future forecasting, or readiness for public-health decisions. The final artifact has separate calibration counties but no fresh independent test set.
