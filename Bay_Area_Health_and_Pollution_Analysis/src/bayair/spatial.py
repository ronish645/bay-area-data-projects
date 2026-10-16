"""Distance-buffered splits operate on coordinates only, never outcome values."""
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import BallTree

from bayair.data import COUNTIES, GROUP

EARTH_KM = 6371.0088


def nearest_km(train, test):
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Distance calculation requires two nonempty sets")
    tree = BallTree(np.deg2rad(test[["Latitude", "Longitude"]].to_numpy()), metric="haversine")
    distance, _ = tree.query(np.deg2rad(train[["Latitude", "Longitude"]].to_numpy()), k=1)
    return distance[:, 0] * EARTH_KM


def buffered_folds(frame, n_splits=3, buffer_km=5):
    folds = []
    for train, valid in GroupKFold(n_splits).split(frame, groups=frame[GROUP]):
        retained = train[nearest_km(frame.iloc[train], frame.iloc[valid]) >= buffer_km]
        if len(retained) < 20:
            raise ValueError("Buffer left too few fitting tracts")
        folds.append((retained, valid))
    return folds


def bay_split(frame, held_out_county=None, buffer_km=5):
    if not set(frame[GROUP]).issubset(COUNTIES):
        raise ValueError("Only Bay Area counties are permitted")
    test = frame[frame[GROUP] == held_out_county].copy()
    eligible = frame[frame[GROUP] != held_out_county].copy()
    if len(test):
        eligible["distance_to_test_km"] = nearest_km(eligible, test)
        eligible = eligible[eligible.distance_to_test_km >= buffer_km].copy()
    fitting, calibration = next(GroupShuffleSplit(n_splits=1, test_size=2, random_state=42).split(eligible, groups=eligible[GROUP]))
    fit, cal = eligible.iloc[fitting].copy(), eligible.iloc[calibration].copy()
    fit["distance_to_calibration_km"] = nearest_km(fit, cal)
    fit = fit[fit.distance_to_calibration_km >= buffer_km].copy()
    return fit.reset_index(drop=True), cal.reset_index(drop=True), test.reset_index(drop=True)
