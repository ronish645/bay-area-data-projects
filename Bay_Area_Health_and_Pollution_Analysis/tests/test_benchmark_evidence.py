import json

import numpy as np
import pandas as pd

from bayair.advanced import conformal_radius
from bayair.data import COUNTIES, GROUP, PROJECT
from bayair.modeling import metrics
from bayair.official import load_official, sha256
from bayair.story import county_summary

REPORTS = PROJECT / "reports"


def test_every_official_tract_has_one_prediction_per_procedure():
    frame = load_official()
    predictions = pd.read_csv(REPORTS / "predictions.csv", dtype={"tract": str})
    assert set(predictions.county) == COUNTIES
    for _, part in predictions.groupby(["feature_set", "procedure"]):
        assert part.tract.is_unique
        assert len(part) == len(frame)
        assert set(part.tract) == set(frame["Census Tract"])
        assert np.isfinite(part[["actual", "predicted", "lower_90", "upper_90"]]).all().all()
        assert (part.lower_90 <= part.predicted).all()
        assert (part.upper_90 >= part.predicted).all()


def test_saved_macro_metrics_and_coverage_recompute_exactly():
    predictions = pd.read_csv(REPORTS / "predictions.csv")
    board = pd.read_csv(REPORTS / "leaderboard.csv")
    for (feature_set, procedure), part in predictions.groupby(["feature_set", "procedure"]):
        record = board[(board.feature_set == feature_set) & (board.procedure == procedure)].iloc[0]
        macro = part.assign(error=abs(part.actual - part.predicted)).groupby("county").error.mean().mean()
        np.testing.assert_allclose(record.macro_county_mae, macro)
        np.testing.assert_allclose(record.pooled_r2, metrics(part.actual, part.predicted)["r2"])
        covered = part.actual.between(part.lower_90, part.upper_90)
        np.testing.assert_allclose(record.coverage_90, covered.mean())


def test_manifest_matches_calibration_and_fitting_constraints():
    manifests = json.loads((REPORTS / "split_manifest.json").read_text())
    lookup = load_official().set_index("Census Tract")
    residuals = pd.read_csv(REPORTS / "calibration_residuals.csv", dtype={"tract": str})
    folds = pd.read_csv(REPORTS / "county_metrics.csv")
    for county, split in manifests.items():
        a, b, c = [set(split[k]["tracts"]) for k in ["fit", "calibration", "test"]]
        assert not a & b and not a & c and not b & c
        assert set(lookup.loc[list(c), GROUP]) == {county}
        assert split["minimum_fit_calibration_km"] >= 5
        assert split["minimum_fit_test_km"] >= 5
        assert split["minimum_calibration_test_km"] >= 5
        for feature in ["air", "social", "combined"]:
            cal = residuals[(residuals.outer_county == county) & (residuals.feature_set == feature) & (residuals.procedure == "selected")]
            assert set(cal.tract) == b
            expected = conformal_radius(cal.actual, cal.predicted)
            saved = folds[(folds.county == county) & (folds.feature_set == feature) & (folds.procedure == "selected")].radius_90.iloc[0]
            np.testing.assert_allclose(saved, expected)


def test_rankings_come_from_official_rows_not_model_estimates():
    expected = county_summary(load_official(labeled=False)).set_index("county").sort_index()
    saved = pd.read_csv(REPORTS / "county_summary.csv").set_index("county").sort_index()
    pd.testing.assert_frame_equal(expected, saved, check_exact=False, rtol=1e-9)


def test_failure_and_post_inspection_status_are_preserved():
    first = pd.read_csv(REPORTS / "failed_extrapolation" / "leaderboard.csv")
    assert first.query("feature_set == 'combined' and procedure == 'selected'").macro_county_mae.iloc[0] > 100
    run = json.loads((REPORTS / "run.json").read_text())
    assert "exploratory" in run.get("evaluation_status", "").lower()


def test_model_family_selection_and_output_bounds_use_fitting_data():
    search = pd.read_csv(REPORTS / "hyperparameter_search.csv")
    folds = pd.read_csv(REPORTS / "county_metrics.csv")
    predictions = pd.read_csv(REPORTS / "predictions.csv")
    manifests = json.loads((REPORTS / "split_manifest.json").read_text())
    lookup = load_official().set_index("Census Tract")
    for row in folds[folds.procedure == "selected"].itertuples():
        inner = search[(search.county == row.county) & (search.feature_set == row.feature_set)]
        np.testing.assert_allclose(row.inner_macro_mae, inner.mean_inner_macro_mae.min())
        chosen = inner[inner.model == row.family]
        np.testing.assert_allclose(chosen.mean_inner_macro_mae.min(), row.inner_macro_mae)
        max_fit_target = lookup.loc[manifests[row.county]["fit"]["tracts"], "Asthma"].max()
        part = predictions[(predictions.county == row.county) & (predictions.feature_set == row.feature_set)]
        assert part.predicted.between(0, max_fit_target + 1e-9).all()
