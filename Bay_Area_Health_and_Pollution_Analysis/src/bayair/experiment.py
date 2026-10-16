"""Run predeclared experiments and persist auditable out-of-fold predictions."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, LeaveOneGroupOut

from bayair.data import AIR, DEFAULT_DATA, FEATURE_SETS, GROUP, PROJECT, TARGET, load_data
from bayair.modeling import SEED, candidates, county_bootstrap, metrics, tune


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(data_path=DEFAULT_DATA, output=PROJECT / "reports", artifacts=PROJECT / "artifacts"):
    start = time.monotonic()
    output, artifacts = Path(output), Path(artifacts)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(data_path)
    write_json(output / "data_audit.json", audit)
    y, groups = data[TARGET], data[GROUP]
    rows, predictions, importances, splits = [], [], [], []
    outer = LeaveOneGroupOut()
    for train, test in outer.split(data, y, groups):
        county = groups.iloc[test].iloc[0]
        print(f"Holding out {county}: {len(test)} tracts", flush=True)
        splits.append({"held_out_county": county,
                       "training_counties": sorted(groups.iloc[train].unique()),
                       "train_tracts": data.iloc[train]["Census Tract"].tolist(),
                       "test_tracts": data.iloc[test]["Census Tract"].tolist()})
        for feature_set, features in FEATURE_SETS.items():
            searches = {name: tune(name, data.iloc[train][features], y.iloc[train], groups.iloc[train])
                        for name in candidates()}
            # Model family is selected using INNER scores, never the outer county.
            winner = max(searches, key=lambda name: searches[name].best_score_)
            for name, search in searches.items():
                pred = search.predict(data.iloc[test][features])
                row = {"county": county, "feature_set": feature_set, "model": name,
                       "n_test": len(test), "inner_mae": -float(search.best_score_),
                       "best_params": json.dumps(search.best_params_, sort_keys=True),
                       **metrics(y.iloc[test], pred)}
                rows.append(row)
                if name == winner:
                    rows.append({**row, "model": "selected"})
                for idx, value in zip(test, pred):
                    predictions.append({"tract": data.loc[idx, "Census Tract"], "county": county,
                                        "feature_set": feature_set, "model": name,
                                        "selected": name == winner, "actual": float(y.iloc[idx]),
                                        "predicted": float(value)})
            if feature_set == "combined":
                importance = permutation_importance(
                    searches[winner], data.iloc[test][features], y.iloc[test],
                    scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED, n_jobs=1)
                for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std):
                    importances.append({"county": county, "selected_model": winner, "feature": feature,
                                        "mae_increase": float(mean), "shuffle_std": float(std)})

    folds, oof = pd.DataFrame(rows), pd.DataFrame(predictions)
    folds.to_csv(output / "county_metrics.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(importances).to_csv(output / "permutation_importance.csv", index=False)
    write_json(output / "split_manifest.json", splits)
    summary_rows = []
    for (feature_set, name), part in folds.groupby(["feature_set", "model"]):
        subset = oof[(oof.feature_set == feature_set) & (oof.selected if name == "selected" else oof.model == name)]
        summary_rows.append({"feature_set": feature_set, "model": name,
                             "macro_county_mae": float(part.mae.mean()),
                             "county_mae_std": float(part.mae.std()),
                             "worst_county_mae": float(part.mae.max()),
                             "pooled_mae": float(metrics(subset.actual, subset.predicted)["mae"]),
                             "pooled_rmse": float(metrics(subset.actual, subset.predicted)["rmse"]),
                             "pooled_r2": float(metrics(subset.actual, subset.predicted)["r2"])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "leaderboard.csv", index=False)
    selected = oof[(oof.feature_set == "combined") & oof.selected].copy()
    selected["absolute_error"] = abs(selected.actual - selected.predicted)
    selected["residual"] = selected.actual - selected.predicted
    baseline = oof[(oof.feature_set == "combined") & (oof.model == "median")].set_index("tract")
    selected["baseline_error"] = [abs(row.actual - baseline.loc[row.tract, "predicted"]) for row in selected.itertuples()]
    metadata = data.set_index("Census Tract")
    for column in ["Longitude", "Latitude", "Poverty"]:
        selected[column] = selected.tract.map(metadata[column])
    selected.to_csv(output / "selected_predictions.csv", index=False)
    # Slice cutoff is fixed before fitting. This is an error audit, not a fairness guarantee.
    selected["poverty_slice"] = np.where(selected.Poverty.isna(), "missing", np.where(selected.Poverty >= 40, ">=40%", "<40%"))
    selected.groupby("poverty_slice").agg(n=("tract", "size"), mae=("absolute_error", "mean"),
                                             mean_residual=("residual", "mean")).to_csv(output / "slice_metrics.csv")
    stability = county_bootstrap(selected)
    write_json(output / "stability.json", stability)

    # A diagnostic of interpolation only: same fixed boosting settings, no outer selection.
    # Deliberately kept separate from the nested county leaderboard.
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    diagnostic = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), candidates()["boosting"][0])
    diagnostic_rows = []
    for scheme, splitter in [("random_5fold", KFold(5, shuffle=True, random_state=SEED)),
                             ("county_held_out", LeaveOneGroupOut())]:
        actual, predicted = [], []
        for train, test in splitter.split(data, y, groups):
            fitted = clone(diagnostic).fit(data.iloc[train][FEATURE_SETS["combined"]], y.iloc[train])
            actual.extend(y.iloc[test])
            predicted.extend(fitted.predict(data.iloc[test][FEATURE_SETS["combined"]]))
        diagnostic_rows.append({"scheme": scheme, **metrics(actual, predicted)})
    pd.DataFrame(diagnostic_rows).to_csv(output / "split_diagnostic.csv", index=False)

    print("Fitting final demonstration model on all labeled tracts", flush=True)
    features = FEATURE_SETS["combined"]
    final_searches = {name: tune(name, data[features], y, groups) for name in candidates()}
    final_name = max(final_searches, key=lambda name: final_searches[name].best_score_)
    joblib.dump(final_searches[final_name].best_estimator_, artifacts / "model.joblib")
    model_card = {"model": final_name, "features": features,
                  "best_params": final_searches[final_name].best_params_,
                  "inner_cv_mae": -float(final_searches[final_name].best_score_),
                  "training_rows": len(data), "data_sha256": audit["sha256"],
                  "feature_min": data[features].min().to_dict(),
                  "feature_max": data[features].max().to_dict(),
                  "warning": "Final full-data model has no independent test set. Report nested OOF metrics, not training scores."}
    write_json(artifacts / "model_metadata.json", model_card)
    run_info = {"seed": SEED, "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
                "elapsed_seconds": round(time.monotonic() - start, 2),
                "primary_metric": "Unweighted mean of nine held-out county MAEs",
                "outer_folds": 9, "inner_folds": 3, "data_sha256": audit["sha256"]}
    write_json(output / "run.json", run_info)
    from bayair.reporting import render
    render(output)
    print(summary[summary.model == "selected"].to_string(index=False), flush=True)
    print(f"Reports: {output.resolve()}", flush=True)
    return summary
