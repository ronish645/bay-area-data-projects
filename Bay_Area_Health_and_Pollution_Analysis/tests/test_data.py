import numpy as np
import pandas as pd
import pytest

from bayair.data import AIR, DEFAULT_DATA, FEATURE_SETS, GROUP, load_data, validate_features


def test_real_extract_contract_and_sentinels():
    data, audit = load_data()
    assert audit["raw_rows"] == 1581
    assert audit["model_rows"] == 1580
    assert audit["missing_target_rows"] == 1
    assert audit["negative_feature_values_replaced"]["Traffic"] > 0
    assert not (data[AIR] < 0).any().any()
    assert data["Census Tract"].str.len().eq(11).all()
    assert data[GROUP].nunique() == 9
    assert data.Asthma.notna().all()


def test_feature_allowlist_excludes_answer_and_geographic_identifiers():
    forbidden = {"Asthma", "Asthma Pctl", "Pop. Char.", "DRAFT CES 4.0 Score", "Census Tract", "Date", GROUP, "ZIP", "Longitude", "Latitude", "Cardiovascular Disease"}
    assert not set(FEATURE_SETS["combined"]) & forbidden


def test_duplicate_tract_rejected(tmp_path):
    frame = pd.read_csv(DEFAULT_DATA)
    frame.loc[1, "Census Tract"] = frame.loc[0, "Census Tract"]
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique census tract"):
        load_data(path)


@pytest.mark.parametrize("value", [-999, float("inf"), "bad"])
def test_invalid_inference_values_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        validate_features(pd.DataFrame({"PM2.5": [value]}), ["PM2.5"])


def test_missing_inference_feature_and_empty_row_rejected():
    with pytest.raises(ValueError, match="Missing prediction"):
        validate_features(pd.DataFrame({"x": [1]}), ["PM2.5"])
    with pytest.raises(ValueError, match="every feature missing"):
        validate_features(pd.DataFrame({"PM2.5": [np.nan]}), ["PM2.5"])
