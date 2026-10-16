"""Command-line training and batch inference; no cloud credentials required."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from bayair.data import DEFAULT_DATA, PROJECT, validate_features


def predict(input_path, output_path, artifacts):
    artifacts = Path(artifacts)
    # Only load locally generated, trusted joblib artifacts.
    card = json.loads((artifacts / "model_metadata.json").read_text())
    frame = pd.read_csv(input_path)
    x = validate_features(frame, card["features"])
    model = joblib.load(artifacts / "model.joblib")
    frame["predicted_asthma_indicator"] = model.predict(x)
    frame["outside_training_range"] = ((x < pd.Series(card["feature_min"])) | (x > pd.Series(card["feature_max"]))).any(axis=1)
    frame["missing_feature_count"] = x.isna().sum(axis=1)
    if "interval_radius" in card:
        from bayair.advanced import interval_bounds
        lower, upper = interval_bounds(frame["predicted_asthma_indicator"], card["interval_radius"])
        frame["lower_90"] = lower
        frame["upper_90"] = upper
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="BayAir: county-held-out asthma regression study")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Official Bay Area-only buffered study with calibrated prediction ranges")
    train.add_argument("--output", type=Path, default=PROJECT / "reports")
    train.add_argument("--artifacts", type=Path, default=PROJECT / "artifacts" / "official")
    fetch = sub.add_parser("fetch-official", help="Download/verify official workbook and filter to nine Bay Area counties")
    fetch.add_argument("--workbook", type=Path)
    legacy = sub.add_parser("legacy-train", help="Reproduce the earlier draft-data study")
    legacy.add_argument("--data", type=Path, default=DEFAULT_DATA)
    report = sub.add_parser("report", help="Regenerate figures and the local HTML report")
    report.add_argument("--output", type=Path, default=PROJECT / "reports")
    infer = sub.add_parser("predict", help="Batch prediction using a locally trained model")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--artifacts", type=Path, default=PROJECT / "artifacts" / "official")
    args = parser.parse_args()
    if args.command == "train":
        from bayair.benchmark import train_benchmark
        train_benchmark(args.output, args.artifacts)
    elif args.command == "fetch-official":
        from bayair.official import prepare
        print(json.dumps(prepare(args.workbook), indent=2))
    elif args.command == "legacy-train":
        from bayair.experiment import run
        run(args.data, PROJECT / "reports" / "legacy", PROJECT / "artifacts" / "legacy")
    elif args.command == "report":
        from bayair.story import render_story
        render_story(args.output)
    else:
        predict(args.input, args.output, args.artifacts)


if __name__ == "__main__":
    main()
