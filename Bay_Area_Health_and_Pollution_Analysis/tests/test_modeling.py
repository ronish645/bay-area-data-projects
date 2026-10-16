import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from bayair.data import FEATURE_SETS, GROUP, load_data
from bayair.modeling import tune


def test_group_boundaries_at_both_levels():
    data, _ = load_data()
    groups = data[GROUP]
    seen = []
    for train, test in LeaveOneGroupOut().split(data, groups=groups):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        seen.extend(test)
        for inner_train, inner_test in GroupKFold(3).split(data.iloc[train], groups=groups.iloc[train]):
            assert not set(groups.iloc[train].iloc[inner_train]) & set(groups.iloc[train].iloc[inner_test])
    assert sorted(seen) == list(range(len(data)))


def test_imputation_uses_training_data_only():
    # A held-out outlier would change the median if preprocessing leaked.
    x = pd.DataFrame({"x": [1., 2., np.nan, 4., 5., 6., 7., 8., 9.]})
    search = tune("ridge", x, np.arange(9.), np.repeat(["a", "b", "c"], 3))
    assert search.best_estimator_.named_steps["impute"].statistics_[0] == 5.5
    before = search.best_estimator_.named_steps["impute"].statistics_.copy()
    search.predict(pd.DataFrame({"x": [1e9, np.nan]}))
    np.testing.assert_array_equal(search.best_estimator_.named_steps["impute"].statistics_, before)


def test_real_data_pipeline_can_handle_missing_features():
    data, _ = load_data()
    search = tune("ridge", data[FEATURE_SETS["combined"]], data.Asthma, data[GROUP])
    assert np.isfinite(search.predict(data[FEATURE_SETS["combined"]].head(20))).all()
