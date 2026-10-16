import json

import pandas as pd
import pytest

from bayair.data import COUNTIES, GROUP
from bayair.official import DATA, OFFICIAL_CSV, OFFICIAL_FEATURES, load_official, reconcile_legacy, sha256, validate


def test_official_scope_and_provenance():
    frame = load_official(labeled=False)
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert len(frame) == 1581
    assert len(load_official()) == 1580
    assert set(frame[GROUP]) == COUNTIES
    assert frame["Census Tract"].is_unique
    assert sha256(OFFICIAL_CSV) == provenance["processed_sha256"]
    assert provenance["raw_rows"] == 8035  # Publisher workbook, not training scope.
    assert provenance["processed_rows"] == 1581
    assert provenance["counties"] == 9
    assert provenance["source_url"].startswith("https://data.ca.gov/")


def test_feature_contract_avoids_outcomes_and_identifiers():
    features = OFFICIAL_FEATURES["combined"]
    assert len(features) == 18
    assert not set(features) & {"Asthma", "Asthma Pctl", "Cardiovascular Disease", "Low Birth Weight", "Pop. Char.", "CES 4.0 Score", "Census Tract", "California County", "Latitude", "Longitude", "Date", "Pollution Burden"}


def test_source_reconciliation_is_quantified():
    comparisons = reconcile_legacy().set_index("feature")
    assert comparisons.loc["Asthma", "changed_values"] == 0
    assert comparisons.loc["PM2.5", "changed_values"] == 0
    assert comparisons.loc["Drinking Water", "changed_values"] > 0


@pytest.mark.parametrize("coordinate,value", [("Latitude", 0), ("Longitude", float("inf"))])
def test_invalid_geography_cannot_enter_spatial_evaluation(coordinate, value):
    frame = load_official()
    frame.loc[0, coordinate] = value
    with pytest.raises(ValueError, match="Coordinates|coordinates"):
        validate(frame)
