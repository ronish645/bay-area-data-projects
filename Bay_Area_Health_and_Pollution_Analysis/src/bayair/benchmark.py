"""Bay Area only: buffered, nested county evaluation with reserved calibration."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from bayair.advanced import advanced_candidates, conformal_radius, interval_bounds, macro_mae, spatial_tune
from bayair.data import COUNTIES, GROUP, PROJECT, TARGET
from bayair.modeling import county_bootstrap, metrics
from bayair.official import DATA, ENVIRONMENT, OFFICIAL_FEATURES, load_official, reconcile_legacy, sha256
from bayair.spatial import bay_split, buffered_folds, nearest_km


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def partition_manifest(fit, cal, test, inner):
    manifest = {label: {"counties": sorted(part[GROUP].unique()), "tracts": part["Census Tract"].tolist()}
                for label, part in [("fit", fit), ("calibration", cal), ("test", test)]}
    manifest["inner_folds"] = [{"train": fit.iloc[a]["Census Tract"].tolist(),
                                 "validation": fit.iloc[b]["Census Tract"].tolist()} for a, b in inner]
    manifest["minimum_fit_calibration_km"] = float(nearest_km(fit, cal).min())
    if len(test):
        manifest["minimum_fit_test_km"] = float(nearest_km(fit, test).min())
        manifest["minimum_calibration_test_km"] = float(nearest_km(cal, test).min())
    return manifest


def search_rows(searches, county, feature_set):
    records = []
    for name, search in searches.items():
        for i, params in enumerate(search.cv_results_["params"]):
            records.append({"county": county, "feature_set": feature_set, "model": name,
                            "params": json.dumps(params, sort_keys=True),
                            "mean_inner_macro_mae": -float(search.cv_results_["mean_test_score"][i]),
                            "std_inner_fold_mae": float(search.cv_results_["std_test_score"][i])})
    return records


def train_benchmark(output=PROJECT / "reports", artifacts=PROJECT / "artifacts" / "official"):
    started = time.monotonic()
    output, artifacts = Path(output), Path(artifacts)
    output.mkdir(exist_ok=True, parents=True)
    artifacts.mkdir(exist_ok=True, parents=True)
    frame = load_official()
    if set(frame[GROUP]) != COUNTIES:
        raise ValueError("Training is restricted to the nine Bay Area counties")
    protocol = PROJECT / "docs" / "EVALUATION_PROTOCOL.md"
    provenance = json.loads((DATA / "provenance.json").read_text())
    dump(output / "protocol_snapshot.json", {"protocol_sha256": sha256(protocol), "text": protocol.read_text(),
                                             "source_sha256": provenance["processed_sha256"]})
    predictions, fold_rows, searches_log, permutation, calibration_log, manifests, shift_rows = [], [], [], [], [], {}, []
    for county in sorted(COUNTIES):
        fit, cal, test = bay_split(frame, county)
        inner = buffered_folds(fit)
        manifests[county] = partition_manifest(fit, cal, test, inner)
        print(f"{county}: fit={len(fit)}, calibration={len(cal)}, test={len(test)}; all Bay Area, 5 km buffered", flush=True)
        for feature_set, features in OFFICIAL_FEATURES.items():
            searches = {name: spatial_tune(name, fit, features, inner) for name in advanced_candidates()}
            searches_log.extend(search_rows(searches, county, feature_set))
            winner = max(searches, key=lambda name: searches[name].best_score_)
            for procedure in ["selected", "median"]:
                family = winner if procedure == "selected" else "median"
                model = searches[family].best_estimator_
                prediction = model.predict(test[features])
                cal_prediction = model.predict(cal[features])
                radius = conformal_radius(cal[TARGET], cal_prediction)
                lower, upper = interval_bounds(prediction, radius)
                covered = (test[TARGET].to_numpy() >= lower) & (test[TARGET].to_numpy() <= upper)
                fold_rows.append({"county": county, "feature_set": feature_set, "procedure": procedure,
                                  "family": family, "n_fit": len(fit), "n_calibration": len(cal), "n_test": len(test),
                                  "inner_macro_mae": -float(searches[family].best_score_),
                                  "radius_90": radius, "coverage_90": float(covered.mean()),
                                  "mean_interval_width": float((upper - lower).mean()),
                                  "params": json.dumps(searches[family].best_params_, sort_keys=True),
                                  **metrics(test[TARGET], prediction)})
                for i, row in enumerate(test.itertuples(index=False, name=None)):
                    source = test.iloc[i]
                    predictions.append({"tract": source["Census Tract"], "county": county,
                                        "feature_set": feature_set, "procedure": procedure, "family": family,
                                        "actual": float(source[TARGET]), "predicted": float(prediction[i]),
                                        "lower_90": float(lower[i]), "upper_90": float(upper[i]), "covered": bool(covered[i]),
                                        "Longitude": float(source.Longitude), "Latitude": float(source.Latitude),
                                        "Poverty": None if pd.isna(source.Poverty) else float(source.Poverty)})
                for tract, actual, value in zip(cal["Census Tract"], cal[TARGET], cal_prediction):
                    calibration_log.append({"outer_county": county, "feature_set": feature_set, "procedure": procedure,
                                            "tract": tract, "actual": float(actual), "predicted": float(value),
                                            "absolute_residual": float(abs(actual - value))})
            if feature_set == "combined":
                chosen = searches[winner].best_estimator_
                imp = permutation_importance(chosen, test[features], test[TARGET], scoring="neg_mean_absolute_error",
                                             random_state=42, n_repeats=5, n_jobs=1)
                for feature, mean, std in zip(features, imp.importances_mean, imp.importances_std):
                    permutation.append({"county": county, "feature": feature, "mae_increase": float(mean), "shuffle_sd": float(std)})
                shuffled = clone(chosen).fit(fit[features], np.random.default_rng(42).permutation(fit[TARGET]))
                fold_rows.append({"county": county, "feature_set": "combined", "procedure": "shuffled_labels",
                                  "family": winner, "n_fit": len(fit), "n_calibration": len(cal), "n_test": len(test),
                                  **metrics(test[TARGET], shuffled.predict(test[features]))})
                for feature in features:
                    mean, std = fit[feature].mean(), fit[feature].std()
                    delta = (test[feature].mean() - mean) / std if std > 0 else 0.
                    outside = (test[feature] < fit[feature].min()) | (test[feature] > fit[feature].max())
                    shift_rows.append({"county": county, "feature": feature, "standardized_mean_shift": float(delta),
                                       "outside_fit_range_fraction": float(outside.mean()),
                                       "test_missing_fraction": float(test[feature].isna().mean())})

    folds, oof = pd.DataFrame(fold_rows), pd.DataFrame(predictions)
    folds.to_csv(output / "county_metrics.csv", index=False)
    oof.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(searches_log).to_csv(output / "hyperparameter_search.csv", index=False)
    pd.DataFrame(permutation).to_csv(output / "permutation_importance.csv", index=False)
    pd.DataFrame(calibration_log).to_csv(output / "calibration_residuals.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(output / "distribution_shift.csv", index=False)
    dump(output / "split_manifest.json", manifests)
    board = []
    for (feature_set, procedure), rows in folds.groupby(["feature_set", "procedure"]):
        record = {"feature_set": feature_set, "procedure": procedure, "macro_county_mae": float(rows.mae.mean()),
                  "worst_county_mae": float(rows.mae.max())}
        part = oof[(oof.feature_set == feature_set) & (oof.procedure == procedure)]
        if len(part):
            record.update({f"pooled_{k}": v for k, v in metrics(part.actual, part.predicted).items()})
            record.update({"coverage_90": float(part.covered.mean()), "mean_interval_width": float((part.upper_90 - part.lower_90).mean())})
        board.append(record)
    leaderboard = pd.DataFrame(board)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    selected = oof[(oof.feature_set == "combined") & (oof.procedure == "selected")].copy()
    baseline = oof[(oof.feature_set == "combined") & (oof.procedure == "median")].set_index("tract")
    selected["absolute_error"] = abs(selected.actual - selected.predicted)
    selected["residual"] = selected.actual - selected.predicted
    selected["baseline_error"] = [abs(row.actual - baseline.loc[row.tract, "predicted"]) for row in selected.itertuples()]
    selected["poverty_slice"] = np.where(selected.Poverty.isna(), "Missing", np.where(selected.Poverty >= 40, ">=40%", "<40%"))
    selected.groupby("poverty_slice").agg(n=("tract", "size"), mae=("absolute_error", "mean"),
                                             mean_residual=("residual", "mean"), coverage_90=("covered", "mean")).to_csv(output / "slice_metrics.csv")
    dump(output / "stability.json", county_bootstrap(selected))
    reconcile_legacy().to_csv(output / "draft_final_reconciliation.csv", index=False)

    print("Fitting final Bay Area demonstration artifact, with calibration counties reserved", flush=True)
    fit, cal, empty = bay_split(frame)
    inner = buffered_folds(fit)
    features = OFFICIAL_FEATURES["combined"]
    searches = {name: spatial_tune(name, fit, features, inner) for name in advanced_candidates()}
    winner = max(searches, key=lambda name: searches[name].best_score_)
    final = searches[winner].best_estimator_
    radius = conformal_radius(cal[TARGET], final.predict(cal[features]))
    joblib.dump(final, artifacts / "model.joblib")
    dump(artifacts / "model_metadata.json", {"model": winner, "features": features, "best_params": searches[winner].best_params_,
                                            "training_rows": len(fit), "calibration_rows": len(cal), "nominal_coverage": .9,
                                            "interval_radius": radius, "buffer_km": 5,
                                            "feature_min": fit[features].min().to_dict(), "feature_max": fit[features].max().to_dict(),
                                            "data_sha256": provenance["processed_sha256"], "protocol_sha256": sha256(protocol),
                                            "warning": "Spatially shifted split-conformal intervals have no guaranteed coverage. Artifact has no fresh test set."})
    dump(output / "final_model_manifest.json", partition_manifest(fit, cal, empty, inner))
    pd.DataFrame(search_rows(searches, "final_artifact", "combined")).to_csv(output / "final_model_search.csv", index=False)
    learning_rows = []
    for fold, (a, b) in enumerate(inner):
        for fraction in [.25, .5, 1.]:
            subset = np.random.default_rng(42).choice(a, size=max(20, int(len(a) * fraction)), replace=False)
            learned = clone(final).fit(fit.iloc[subset][features], fit.iloc[subset][TARGET])
            learning_rows.append({"fold": fold, "fraction": fraction, "training_rows": len(subset),
                                  "validation_macro_mae": macro_mae(fit.iloc[b], fit.iloc[b][TARGET], learned.predict(fit.iloc[b][features])),
                                  "training_mae": metrics(fit.iloc[subset][TARGET], learned.predict(fit.iloc[subset][features]))["mae"]})
    pd.DataFrame(learning_rows).to_csv(output / "learning_curve.csv", index=False)
    code_files = sorted((PROJECT / "src" / "bayair").glob("*.py"))
    dump(output / "run.json", {"version": "3.1.0", "evaluation_status": "Exploratory post-inspection robustness revision; initial failed results preserved", "scope": "Nine Bay Area counties only", "seed": 42, "buffer_km": 5,
                               "outer_folds": 9, "inner_folds": 3, "nominal_coverage": .9,
                               "raw_bay_area_rows": provenance["processed_rows"], "labeled_rows": len(frame),
                               "data_sha256": provenance["processed_sha256"], "protocol_sha256": sha256(protocol),
                               "code_sha256": {p.name: sha256(p) for p in code_files}, "python": platform.python_version(),
                               "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "openpyxl"]},
                               "elapsed_seconds": round(time.monotonic() - started, 2)})
    from bayair.story import render_story
    render_story(output)
    print(leaderboard.to_string(index=False), flush=True)
    return leaderboard
