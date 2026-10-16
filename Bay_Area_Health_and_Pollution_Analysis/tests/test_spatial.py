import numpy as np
import pandas as pd
import pytest

from bayair.data import COUNTIES, GROUP
from bayair.official import load_official
from bayair.spatial import bay_split, buffered_folds, nearest_km


def test_haversine_distance_has_expected_units():
    a = pd.DataFrame({"Latitude": [0.], "Longitude": [0.]})
    b = pd.DataFrame({"Latitude": [0.], "Longitude": [1.]})
    np.testing.assert_allclose(nearest_km(a, b), [111.195], rtol=1e-4)


def test_all_outer_partitions_enforce_counties_and_five_km_buffers():
    frame = load_official()
    seen = []
    for county in sorted(COUNTIES):
        fit, cal, test = bay_split(frame, county)
        groups = [set(p[GROUP]) for p in [fit, cal, test]]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]
        assert groups[2] == {county}
        assert nearest_km(fit, test).min() >= 5
        assert nearest_km(cal, test).min() >= 5
        assert nearest_km(fit, cal).min() >= 5
        seen.extend(test["Census Tract"])
        for train, valid in buffered_folds(fit):
            assert not set(fit.iloc[train][GROUP]) & set(fit.iloc[valid][GROUP])
            assert nearest_km(fit.iloc[train], fit.iloc[valid]).min() >= 5
    assert sorted(seen) == sorted(frame["Census Tract"])


def test_outside_bay_area_county_is_rejected():
    frame = load_official()
    frame.loc[0, GROUP] = "Los Angeles"
    with pytest.raises(ValueError, match="Only Bay Area"):
        bay_split(frame, "Solano")
