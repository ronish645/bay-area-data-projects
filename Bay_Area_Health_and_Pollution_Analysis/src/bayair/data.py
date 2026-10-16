"""Explicit feature allowlists prevent target-derived composites entering a model."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT / "Bay_Area_Data1.csv"
TARGET = "Asthma"
GROUP = "California County"
AIR = ["PM2.5", "Ozone", "Diesel PM", "Traffic"]
SOCIAL = ["Education", "Linguistic Isolation", "Poverty", "Unemployment", "Housing Burden"]
FEATURE_SETS = {"air": AIR, "social": SOCIAL, "combined": AIR + SOCIAL}
COUNTIES = {"Alameda", "Contra Costa", "Marin", "Napa", "San Francisco", "San Mateo", "Santa Clara", "Solano", "Sonoma"}


def load_data(path: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    data = pd.read_csv(path, dtype={"Census Tract": str})
    data.columns = data.columns.str.strip()
    required = ["Census Tract", GROUP, TARGET, "Longitude", "Latitude", *AIR, *SOCIAL]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data["Census Tract"].isna().any() or data["Census Tract"].duplicated().any():
        raise ValueError("Each row must have one unique census tract identifier")
    data["Census Tract"] = data["Census Tract"].str.zfill(11)
    data[GROUP] = data[GROUP].str.strip()
    if data[GROUP].isna().any() or set(data[GROUP]) != COUNTIES:
        raise ValueError("Expected exactly the nine Bay Area counties with no missing labels")
    numeric = [TARGET, "Longitude", "Latitude", *AIR, *SOCIAL]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if np.isinf(data[numeric].to_numpy()).any():
        raise ValueError("Infinite values are invalid")
    # Traffic density cannot be negative; the supplied extract contains -999.
    sentinels = (data[AIR + SOCIAL] < 0).sum().to_dict()
    data[AIR + SOCIAL] = data[AIR + SOCIAL].mask(data[AIR + SOCIAL] < 0)
    if (data[TARGET].dropna() < 0).any():
        raise ValueError("Asthma target cannot be negative")
    audit = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(data), "raw_columns": len(data.columns),
        "missing_target_rows": int(data[TARGET].isna().sum()),
        "negative_feature_values_replaced": {k: int(v) for k, v in sentinels.items() if v},
        "missing_features_after_sentinel_cleaning": data[AIR + SOCIAL].isna().sum().astype(int).to_dict(),
        "county_counts_raw": data[GROUP].value_counts().sort_index().to_dict(),
        "date_used": False,
    }
    # Never manufacture a supervised label through imputation.
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    audit["model_rows"] = len(data)
    return data, audit


def validate_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")
    x = frame[features].apply(pd.to_numeric, errors="raise").copy()
    if np.isinf(x.to_numpy()).any() or (x < 0).any().any():
        raise ValueError("Prediction features must be nonnegative finite numbers or missing")
    if x.isna().all(axis=1).any():
        raise ValueError("A prediction row cannot have every feature missing")
    return x
