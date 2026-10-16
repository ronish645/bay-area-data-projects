"""Positive-rate regression, spatially grouped tuning and calibrated intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bayair.data import GROUP, TARGET
from bayair.spatial import buffered_folds


class PositiveRegressor(RegressorMixin, BaseEstimator):
    """Constrain predictions to [0, max training target] in CV and inference.

    This deliberately sacrifices unobserved high-rate extrapolation. The upper
    bound is learned inside every fit, never from calibration or test labels.
    """
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, x, y):
        self.regressor_ = clone(self.regressor).fit(x, y)
        self.n_features_in_ = x.shape[1]
        self.training_target_max_ = float(np.max(y))
        return self

    def predict(self, x):
        return np.clip(self.regressor_.predict(x), 0, self.training_target_max_)


def advanced_candidates():
    prefix = "model__regressor__"
    return {
        "median": (DummyRegressor(strategy="median"), {}),
        "log_ridge": (TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1),
                      {prefix + "regressor__alpha": [1., 100., 1000.]}),
        "extra_trees": (ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=1),
                        {prefix + "min_samples_leaf": [3, 12], prefix + "max_features": [.7, 1.]}),
        "boosting": (HistGradientBoostingRegressor(max_iter=180, early_stopping=False, random_state=42),
                     {prefix + "max_leaf_nodes": [7, 15], prefix + "l2_regularization": [1., 20.],
                      prefix + "loss": ["squared_error", "absolute_error"]}),
    }


def macro_mae(frame, y, prediction):
    return float(pd.DataFrame({"county": frame[GROUP].to_numpy(), "error": abs(np.asarray(y) - prediction)}).groupby("county").error.mean().mean())


def spatial_tune(name, frame, features, folds=None):
    estimator, grid = advanced_candidates()[name]
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()), ("model", PositiveRegressor(estimator)),
    ])
    folds = buffered_folds(frame) if folds is None else folds

    def scorer(fitted, x_valid, y_valid):
        return -macro_mae(frame.loc[x_valid.index], y_valid, fitted.predict(x_valid))

    search = GridSearchCV(pipeline, grid, scoring=scorer, cv=folds, n_jobs=1, error_score="raise")
    search.fit(frame[features], frame[TARGET])
    return search


def conformal_radius(y, prediction, coverage=.9):
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one")
    residual = np.abs(np.asarray(y) - np.asarray(prediction))
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Calibration requires nonempty finite residuals")
    rank = int(np.ceil((len(residual) + 1) * coverage))
    if rank > len(residual):
        raise ValueError("Too few calibration rows for the requested finite-sample coverage")
    return float(np.partition(residual, rank - 1)[rank - 1])


def interval_bounds(prediction, radius):
    prediction = np.asarray(prediction)
    return np.maximum(0, prediction - radius), prediction + radius
