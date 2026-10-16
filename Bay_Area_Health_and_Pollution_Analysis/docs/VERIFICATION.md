# Local verification

Verified on 2026-10-15 (America/Los_Angeles), Python 3.12.14. Actual package versions, source/protocol/code hashes and measured runtime are recorded in `reports/run.json`.

- Full Bay Area-only training, calibration, held-out scoring, artifact serialization and report generation completed using `python run_bayair.py train`.
- Notebook executed from a fresh local Jupyter kernel: **11 code cells, no errors**; outputs are saved.
- Python tests verify official source scope and checksums, missing labels, feature exclusions, county disjointness, actual 5 km distances in outer/inner partitions, finite-sample conformal calculations, extrapolation bounds, inference serialization, preserved failures and agreement of metrics with saved predictions.
- Offline JavaScript tests verify pollutant rankings, feature selection, county filtering and interaction handlers in both report versions.
- Batch inference completed on nine official feature rows, returning predictions, ranges and range-support flags.
- Original draft CSV remains unchanged. Initial extrapolation failure and earlier study outputs are preserved.
- Scientific figures were inspected locally. Automated browser access to local HTML is restricted, so the explorer has not had a real-browser visual verification; JavaScript unit checks are not a substitute for that check.
- CI configuration is provided; no remote CI run or deployment is claimed. Changes are local and have not been pushed to GitHub.
