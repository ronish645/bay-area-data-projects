"""Pinned official CES release ingestion; original draft data stays unchanged."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from bayair.data import AIR, COUNTIES, GROUP, PROJECT, SOCIAL, TARGET

SOURCE_URL = "https://data.ca.gov/dataset/11eb2b90-f3c1-46b4-bdf2-ba1dab939dac/resource/9a90474a-3912-4955-9552-457ee41cea51/download/calenviroscreen40resultsdatadictionary_f_2021.xlsx"
CATALOG_URL = "https://publish.data.ca.gov/dataset/calenviroscreen-4-0/resource/9a90474a-3912-4955-9552-457ee41cea51"
ENVIRONMENT = ["Ozone", "PM2.5", "Diesel PM", "Drinking Water", "Lead", "Pesticides", "Tox. Release", "Traffic", "Cleanup Sites", "Groundwater Threats", "Haz. Waste", "Imp. Water Bodies", "Solid Waste"]
OFFICIAL_FEATURES = {"air": AIR, "social": SOCIAL, "combined": ENVIRONMENT + SOCIAL}
DATA = PROJECT / "data"
OFFICIAL_CSV = DATA / "official_bay_area.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(workbook=None):
    """Download or ingest a workbook, verify its schema, retain a provenance manifest."""
    raw = DATA / "raw" / "ces4_official.xlsx"
    if workbook is not None:
        raw = Path(workbook)
    elif not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
        if not payload.startswith(b"PK"):
            raise ValueError("Source returned an access page instead of an XLSX workbook")
        raw.write_bytes(payload)
    manifest_path = DATA / "provenance.json"
    digest = sha256(raw)
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["workbook_sha256"]
        if digest != expected:
            raise ValueError("Official source bytes changed; review the release before replacing the pinned snapshot")
    frame = pd.read_excel(raw, sheet_name="CES4.0FINAL_results")
    frame.columns = frame.columns.str.strip()
    required = ["Census Tract", GROUP, "Total Population", "Approximate Location", "Longitude", "Latitude", TARGET, "Pollution Burden", *ENVIRONMENT, *SOCIAL]
    if set(required) - set(frame):
        raise ValueError(f"Official workbook lacks columns: {set(required) - set(frame)}")
    if len(frame) != 8035 or frame[GROUP].str.strip().nunique() != 58:
        raise ValueError("Pinned CES 4.0 release must contain 8,035 tracts and 58 counties")
    frame = frame[required].copy()
    frame["Census Tract"] = frame["Census Tract"].astype("int64").astype(str).str.zfill(11)
    frame[GROUP] = frame[GROUP].str.strip()
    if frame["Census Tract"].duplicated().any():
        raise ValueError("Duplicate official census tract")
    numeric = [c for c in required if c not in ["Census Tract", GROUP, "Approximate Location"]]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    negative = (frame[ENVIRONMENT + SOCIAL] < 0).sum()
    frame[ENVIRONMENT + SOCIAL] = frame[ENVIRONMENT + SOCIAL].mask(frame[ENVIRONMENT + SOCIAL] < 0)
    validate(frame)
    # User scope: the publisher's workbook is statewide; this project is not.
    raw_rows = len(frame)
    frame = frame[frame[GROUP].isin(COUNTIES)].copy()
    DATA.mkdir(exist_ok=True)
    frame.to_csv(OFFICIAL_CSV, index=False, float_format="%.12g")
    # Preserve the publisher's dictionary, independent of our interpretation.
    dictionary = pd.read_excel(raw, sheet_name="Data Dictionary", header=1)
    dictionary.to_csv(DATA / "official_dictionary.csv", index=False)
    provenance = {"release": "CalEnviroScreen 4.0 final (2021)", "source_url": SOURCE_URL,
                  "catalog_url": CATALOG_URL, "workbook_sha256": digest,
                  "processed_sha256": sha256(OFFICIAL_CSV), "raw_rows": raw_rows,
                  "processed_rows": len(frame), "counties": 9, "missing_target": int(frame[TARGET].isna().sum()),
                  "bay_area_rows": int(frame[GROUP].isin(COUNTIES).sum()),
                  "negative_predictors_to_missing": {k: int(v) for k, v in negative.items() if v},
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "rights": "California catalog lists no restrictions on public use; no new license assigned.",
                  "transformations": [f"Select {len(required)} documented fields", "Filter to the nine Bay Area counties before any modeling", "Normalize tract identifiers and county whitespace", "Convert negative predictor placeholders to missing", "Preserve missing targets for descriptive audit; exclude them only in modeling"]}
    manifest_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def validate(frame):
    if not frame["Census Tract"].str.fullmatch(r"06\d{9}").all() or frame["Census Tract"].duplicated().any():
        raise ValueError("Tract identifiers must be unique 2010 California tract IDs")
    if frame[GROUP].isna().any():
        raise ValueError("Missing county")
    coordinates = frame[["Latitude", "Longitude"]]
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Every tract requires finite coordinates for spatial validation")
    if not frame.Latitude.between(32, 43).all() or not frame.Longitude.between(-125, -114).all():
        raise ValueError("Coordinates outside California bounds")
    if (frame[TARGET].dropna() < 0).any():
        raise ValueError("Target must be nonnegative")
    values = frame[ENVIRONMENT + SOCIAL + [TARGET]].to_numpy()
    if np.isinf(values).any():
        raise ValueError("Infinite measurements are invalid")


def load_official(labeled=True):
    if not OFFICIAL_CSV.exists():
        raise FileNotFoundError("Run `python -m bayair.cli fetch-official` first")
    manifest = json.loads((DATA / "provenance.json").read_text())
    if sha256(OFFICIAL_CSV) != manifest["processed_sha256"]:
        raise ValueError("Official CSV checksum mismatch; rebuild from the pinned workbook")
    frame = pd.read_csv(OFFICIAL_CSV, dtype={"Census Tract": str})
    validate(frame)
    if set(frame[GROUP]) != COUNTIES:
        raise ValueError("This study permits only the nine Bay Area counties")
    return frame.dropna(subset=[TARGET]).reset_index(drop=True) if labeled else frame


def reconcile_legacy():
    """Quantify draft-vs-final differences; never silently overwrite the old CSV."""
    official = load_official(labeled=False).set_index("Census Tract")
    legacy = pd.read_csv(PROJECT / "Bay_Area_Data1.csv", dtype={"Census Tract": str})
    legacy.columns = legacy.columns.str.strip()
    legacy["Census Tract"] = legacy["Census Tract"].str.zfill(11)
    legacy = legacy.set_index("Census Tract")
    rows = []
    for feature in [TARGET, *ENVIRONMENT, *SOCIAL, "Pollution Burden"]:
        a, b = legacy[feature].align(official[feature], join="inner")
        valid = a.notna() & b.notna() & (a >= 0) & (b >= 0)
        delta = (a[valid] - b[valid]).abs()
        rows.append({"feature": feature, "matched_nonmissing": int(valid.sum()),
                     "changed_values": int((delta > 1e-6).sum()), "mean_absolute_difference": float(delta.mean())})
    return pd.DataFrame(rows)
