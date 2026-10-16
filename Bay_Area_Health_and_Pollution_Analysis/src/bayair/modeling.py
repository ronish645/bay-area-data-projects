"""Nested validation: all preprocessing and selection happen inside training folds."""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def candidates():
    return {
        "median": (DummyRegressor(strategy="median"), {}),
        "ridge": (Ridge(), {"model__alpha": [0.1, 10.0, 1000.0]}),
        "forest": (RandomForestRegressor(n_estimators=80, random_state=SEED, n_jobs=1),
                   {"model__min_samples_leaf": [3, 12], "model__max_features": [0.7, 1.0]}),
        "boosting": (HistGradientBoostingRegressor(max_iter=120, early_stopping=False, random_state=SEED),
                     {"model__max_leaf_nodes": [7, 15], "model__l2_regularization": [1.0, 20.0]}),
    }


def tune(name, x, y, groups):
    estimator, grid = candidates()[name]
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()), ("model", estimator),
    ])
    search = GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=3),
                          scoring="neg_mean_absolute_error", n_jobs=1, error_score="raise")
    search.fit(x, y, groups=groups)
    return search


def metrics(y, prediction):
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "r2": float(r2_score(y, prediction))}


def county_bootstrap(predictions, repeats=2000):
    """Descriptive cluster bootstrap; folds overlap in training, so not a formal CI."""
    county = predictions.groupby("county").agg(
        selected_error=("absolute_error", "mean"), baseline_error=("baseline_error", "mean"))
    gains = (county.baseline_error - county.selected_error).to_numpy()
    rng = np.random.default_rng(SEED)
    draws = rng.choice(gains, size=(repeats, len(gains)), replace=True).mean(axis=1)
    return {"macro_mae_improvement": float(gains.mean()),
            "descriptive_95_percent_range": np.quantile(draws, [0.025, 0.975]).tolist(),
            "counties": len(gains), "repeats": repeats,
            "interpretation": "County-resampled stability range, not a formal confidence interval; outer training sets overlap."}
