"""Verify that committed headline metrics agree with saved held-out predictions."""
import json

import numpy as np
import pandas as pd

from bayair.data import PROJECT, load_data
from bayair.modeling import metrics


def test_saved_predictions_cover_each_tract_once_per_candidate():
    data, audit = load_data()
    reports = PROJECT / "reports" / "legacy"
    assert json.loads((reports / "run.json").read_text())["data_sha256"] == audit["sha256"]
    oof = pd.read_csv(reports / "oof_predictions.csv", dtype={"tract": str})
    for (_, _), rows in oof.groupby(["feature_set", "model"]):
        assert len(rows) == len(data)
        assert set(rows.tract) == set(data["Census Tract"])
        assert rows.tract.is_unique
    for _, rows in oof[oof.selected].groupby("feature_set"):
        assert rows.tract.is_unique
        assert len(rows) == len(data)


def test_saved_splits_have_no_county_or_tract_overlap():
    data, _ = load_data()
    county_lookup = data.set_index("Census Tract")["California County"]
    manifests = json.loads((PROJECT / "reports" / "legacy" / "split_manifest.json").read_text())
    assert len(manifests) == 9
    all_test = []
    for fold in manifests:
        assert not set(fold["train_tracts"]) & set(fold["test_tracts"])
        assert set(county_lookup.loc[fold["test_tracts"]]) == {fold["held_out_county"]}
        assert fold["held_out_county"] not in set(county_lookup.loc[fold["train_tracts"]])
        all_test.extend(fold["test_tracts"])
    assert sorted(all_test) == sorted(data["Census Tract"])


def test_recompute_primary_metrics_from_predictions():
    reports = PROJECT / "reports" / "legacy"
    predictions = pd.read_csv(reports / "selected_predictions.csv")
    board = pd.read_csv(reports / "leaderboard.csv")
    row = board[(board.feature_set == "combined") & (board.model == "selected")].iloc[0]
    calculated = metrics(predictions.actual, predictions.predicted)
    np.testing.assert_allclose(row.pooled_mae, calculated["mae"])
    np.testing.assert_allclose(row.pooled_r2, calculated["r2"])
    macro = predictions.assign(error=abs(predictions.actual - predictions.predicted)).groupby("county").error.mean().mean()
    np.testing.assert_allclose(row.macro_county_mae, macro)


def test_selected_families_follow_inner_scores():
    folds = pd.read_csv(PROJECT / "reports" / "legacy" / "county_metrics.csv")
    for (_, _), part in folds.groupby(["county", "feature_set"]):
        best_inner = part.loc[part.model != "selected", "inner_mae"].min()
        np.testing.assert_allclose(part.loc[part.model == "selected", "inner_mae"].iloc[0], best_inner)
