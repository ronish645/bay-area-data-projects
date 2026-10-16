import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from bayair.advanced import PositiveRegressor, conformal_radius, interval_bounds


def test_extrapolation_guard_is_learned_from_fitting_labels():
    x = np.array([[0.], [1.], [2.]])
    fitted = PositiveRegressor(LinearRegression()).fit(x, [2., 4., 6.])
    np.testing.assert_allclose(fitted.predict([[-1e6], [1e6]]), [0., 6.])
    assert fitted.training_target_max_ == 6.


def test_conformal_uses_finite_sample_order_statistic():
    # n=10, ceil((n+1)*0.9)=10 -> maximum residual, not 90th percentile interpolation.
    assert conformal_radius(np.arange(1., 11.), np.zeros(10)) == 10.
    lower, upper = interval_bounds([3., 20.], 10.)
    np.testing.assert_array_equal(lower, [0., 10.])
    np.testing.assert_array_equal(upper, [13., 30.])


@pytest.mark.parametrize("actual,prediction", [([], []), ([1., np.nan], [0., 0.]), ([1.], [0.])])
def test_invalid_or_insufficient_calibration_rejected(actual, prediction):
    with pytest.raises(ValueError):
        conformal_radius(actual, prediction)
