"""Answer the regional questions first; expose the ML evidence second."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayair.data import AIR, GROUP, PROJECT, SOCIAL, TARGET
from bayair.official import DATA, load_official

BLUE, TEAL, GRAY, ORANGE = "#3656cf", "#147a72", "#8390a5", "#cc663d"


def county_summary(frame):
    columns = ["Pollution Burden", *AIR, TARGET, "Poverty", "Unemployment"]
    summary = frame.groupby(GROUP)[columns].mean()
    summary["tracts"] = frame.groupby(GROUP).size()
    summary["asthma_observed"] = frame.groupby(GROUP)[TARGET].count()
    return summary.reset_index().rename(columns={GROUP: "county"})


def render_story(output=PROJECT / "reports"):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    frame = load_official(labeled=False)
    counties = county_summary(frame)
    counties.to_csv(output / "county_summary.csv", index=False)
    board = pd.read_csv(output / "leaderboard.csv")
    folds = pd.read_csv(output / "county_metrics.csv")
    predictions = pd.read_csv(output / "predictions.csv", dtype={"tract": str})
    selected = predictions[(predictions.feature_set == "combined") & (predictions.procedure == "selected")].copy()
    importance = pd.read_csv(output / "permutation_importance.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 160,
                         "figure.facecolor": "white", "text.color": "#1a2941", "axes.labelcolor": "#1a2941"})

    def save(name):
        plt.tight_layout()
        plt.savefig(figures / name, bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, feature, color, title in [(axes[0], "PM2.5", BLUE, "Fine-particle pollution"), (axes[1], TARGET, TEAL, "Asthma emergency-visit indicator")]:
        part = counties.sort_values(feature)
        ax.barh(part.county, part[feature], color=color)
        ax.set(title=title, xlabel="Mean across tracts · µg/m³" if feature == "PM2.5" else "Mean tract-level age-adjusted visits per 10,000")
    fig.suptitle("Pollution and asthma: the county rankings differ", fontweight="bold", y=1.01, fontsize=16)
    save("county_comparison.png")

    corr = frame[AIR + SOCIAL + [TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    heat = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i,j]) > .6 else "#1a2941")
    ax.set_title("Spearman associations across Bay Area tracts\nPairwise nonmissing observations; association does not establish causation", pad=16)
    fig.colorbar(heat, ax=ax, shrink=.75, label="Rank correlation")
    save("correlation_heatmap.png")
    corr.to_csv(output / "spearman_correlations.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feature, cmap, title in [(axes[0], "PM2.5", "YlOrRd", "Fine-particle pollution · µg/m³"), (axes[1], TARGET, "viridis", "Asthma indicator · visits per 10,000")]:
        points = ax.scatter(frame.Longitude, frame.Latitude, c=frame[feature], cmap=cmap, s=13, alpha=.8, linewidths=0)
        ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
        ax.set_aspect(1 / np.cos(np.deg2rad(frame.Latitude.mean())))
        fig.colorbar(points, ax=ax, shrink=.65)
    save("regional_maps.png")

    county_errors = folds[(folds.feature_set == "combined") & folds.procedure.isin(["selected", "median"])].pivot(index="county", columns="procedure", values="mae").sort_values("selected")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    county_errors.rename(columns={"median": "Median baseline", "selected": "Selected procedure"}).plot.barh(ax=ax, color=[GRAY, TEAL])
    ax.set(title="Can the model generalize to a held-out Bay Area county?", xlabel="Held-out MAE · 5 km point buffer, separate calibration counties", ylabel="")
    ax.legend(frameon=False)
    save("county_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ablation = board[board.procedure == "selected"].set_index("feature_set").loc[["air", "social", "combined"]]
    axes[0].bar(["Air / traffic", "Socioeconomic", "Combined"], ablation.macro_county_mae, color=[GRAY, BLUE, TEAL])
    axes[0].set(title="Feature-family ablation", ylabel="Macro county MAE · lower is better")
    for i, value in enumerate(ablation.macro_county_mae):
        axes[0].text(i, value + .25, f"{value:.2f}", ha="center")
    axes[1].scatter(selected.actual, selected.predicted, s=10, alpha=.35, color=TEAL, edgecolors="none")
    extent = [0, max(selected.actual.max(), selected.predicted.max())]
    axes[1].plot(extent, extent, "--", color=GRAY)
    axes[1].set(title="Every prediction is out of county", xlabel="Observed indicator", ylabel="Predicted indicator")
    save("ablation_and_fit.png")

    coverage = folds[(folds.feature_set == "combined") & (folds.procedure == "selected")].sort_values("coverage_90")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(coverage.county, coverage.coverage_90 * 100, color=TEAL)
    axes[0].axvline(90, linestyle="--", color=ORANGE, label="Nominal 90%")
    axes[0].set(xlim=(0, 105), xlabel="Observed coverage (%)", title="Do the prediction ranges cover reality?")
    axes[0].legend(frameon=False)
    ordered = selected.sort_values("predicted").iloc[::16]
    axes[1].fill_between(range(len(ordered)), ordered.lower_90, ordered.upper_90, color=BLUE, alpha=.15, label="90% nominal range")
    axes[1].scatter(range(len(ordered)), ordered.actual, s=12, color=TEAL, label="Observed")
    axes[1].plot(range(len(ordered)), ordered.predicted, color=BLUE, linewidth=1, label="Prediction")
    axes[1].set(title="A sample of held-out tracts", xlabel="Tracts ordered by prediction", ylabel="Asthma indicator")
    axes[1].legend(frameon=False, fontsize=8)
    save("interval_coverage.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    imp = importance.groupby("feature").mae_increase.agg(["mean", "std"]).sort_values("mean")
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=BLUE, capsize=2)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set(title="What does the model rely on?", xlabel="Held-out MAE increase after shuffling · mean ± SD across counties")
    save("importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    residual = selected.actual - selected.predicted
    axes[0].scatter(selected.predicted, residual, s=10, color=BLUE, alpha=.4, edgecolors="none")
    axes[0].axhline(0, color=GRAY, linestyle="--")
    axes[0].set(xlabel="Prediction", ylabel="Observed − predicted", title="Underestimation and overestimation")
    error = axes[1].scatter(selected.Longitude, selected.Latitude, c=abs(residual), s=12, cmap="magma_r", alpha=.8)
    axes[1].set(xlabel="Longitude", ylabel="Latitude", title="Where are errors concentrated?")
    fig.colorbar(error, ax=axes[1], shrink=.7, label="Absolute error")
    save("residuals.png")

    learning = pd.read_csv(output / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = learning.groupby("fraction").agg(rows=("training_rows", "mean"), training=("training_mae", "mean"), validation=("validation_macro_mae", "mean"))
    ax.plot(grouped.rows, grouped.training, "o-", color=GRAY, label="Training MAE")
    ax.plot(grouped.rows, grouped.validation, "o-", color=BLUE, label="Validation macro county MAE")
    ax.set(title="Learning-curve diagnostic", xlabel="Mean fitting tracts", ylabel="Error")
    ax.legend(frameon=False)
    save("learning_curve.png")

    manifest = json.loads((output / "split_manifest.json").read_text())["Alameda"]
    lookup = frame.set_index("Census Tract")
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, color in [("fit", GRAY), ("calibration", TEAL), ("test", ORANGE)]:
        part = lookup.loc[manifest[label]["tracts"]]
        ax.scatter(part.Longitude, part.Latitude, color=color, s=13, alpha=.65, label=f"{label} ({len(part)})")
    ax.set(title="Alameda fold: fitting, calibration and test", xlabel="Longitude", ylabel="Latitude")
    ax.legend(frameon=False)
    ax.set_aspect(1 / np.cos(np.deg2rad(frame.Latitude.mean())))
    save("spatial_split.png")

    stats = board[(board.feature_set == "combined") & (board.procedure == "selected")].iloc[0]
    baseline = board[(board.feature_set == "combined") & (board.procedure == "median")].iloc[0]
    pm = counties.loc[counties["PM2.5"].idxmax()]
    asthma = counties.loc[counties[TARGET].idxmax()]
    ozone = counties.loc[counties.Ozone.idxmax()]
    diesel = counties.loc[counties["Diesel PM"].idxmax()]
    payload = {"counties": counties.to_dict("records"), "leaderboard": board.where(board.notna(), None).to_dict("records"),
               "folds": folds[folds.procedure == "selected"].where(folds.notna(), None).to_dict("records"),
               "predictions": json.loads(selected.to_json(orient="records")),
               "stats": {"macro_mae": float(stats.macro_county_mae), "baseline_mae": float(baseline.macro_county_mae),
                         "pooled_r2": float(stats.pooled_r2), "coverage": float(stats.coverage_90)}}
    # Pandas float columns keep NaN when replacing with None; JSON via pandas normalizes them.
    payload["leaderboard"] = json.loads(board.to_json(orient="records"))
    payload["folds"] = json.loads(folds[folds.procedure == "selected"].to_json(orient="records"))
    html = (Path(__file__).parent / "story_template.html").read_text().replace("__DATA__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))
    replacements = {"__PM_COUNTY__": pm["county"], "__PM_VALUE__": f"{pm['PM2.5']:.2f}",
                    "__ASTHMA_COUNTY__": asthma["county"], "__ASTHMA_VALUE__": f"{asthma[TARGET]:.1f}",
                    "__MAE__": f"{stats.macro_county_mae:.2f}", "__BASELINE__": f"{baseline.macro_county_mae:.2f}",
                    "__R2__": f"{stats.pooled_r2:.3f}", "__COVERAGE__": f"{stats.coverage_90:.1%}"}
    for key, value in replacements.items():
        html = html.replace(key, value)
    (output / "index.html").write_text(html)
    text = f"""# Bay Area findings — official CES 4.0 final source

## Regional questions

- Highest mean PM2.5: **{pm['county']} ({pm['PM2.5']:.2f} µg/m³)**.
- Highest mean ozone: **{ozone['county']} ({ozone.Ozone:.5f} ppm)**.
- Highest mean diesel PM: **{diesel['county']} ({diesel['Diesel PM']:.3f} tons/year/km²)**.
- Highest mean tract-level asthma ED indicator: **{asthma['county']} ({asthma[TARGET]:.2f} age-adjusted visits per 10,000)**.

These are unweighted tract averages in a historical release, not official aggregated county rates or current conditions. One tract has no asthma label. Pollution burden is a broader environmental composite, not an air-only measurement.

## Buffered nested evaluation

Combined selected procedure: macro county MAE **{stats.macro_county_mae:.2f}**; median baseline **{baseline.macro_county_mae:.2f}**. Pooled R² **{stats.pooled_r2:.3f}**. Nominal 90% intervals cover **{stats.coverage_90:.1%}** of held-out observations; mean width **{stats.mean_interval_width:.2f}** indicator units.

All training, tuning, calibration and testing use only the nine Bay Area counties. Each outer test county is excluded from every fit operation, and fitting/calibration/test points are separated by 5 km. Three county-grouped buffered inner folds select model family/parameters. Two other counties supply interval calibration; they are not used in fitting/tuning. This is a harder and smaller-data task than v2; score changes cannot be attributed to one change alone.

| Feature family | Macro county MAE | Pooled R² | Empirical interval coverage |
|---|---:|---:|---:|
"""
    for row in ablation.reset_index().itertuples():
        text += f"| {row.feature_set} | {row.macro_county_mae:.2f} | {row.pooled_r2:.3f} | {row.coverage_90:.1%} |\n"
    text += "\n## Every held-out county, combined procedure\n\n| County | Model family | MAE | R² | Coverage |\n|---|---|---:|---:|---:|\n"
    for row in folds[(folds.feature_set == "combined") & (folds.procedure == "selected")].sort_values("mae", ascending=False).itertuples():
        text += f"| {row.county} | {row.family} | {row.mae:.2f} | {row.r2:.3f} | {row.coverage_90:.1%} |\n"
    text += """
## Evaluation status: exploratory robustness revision

The first official-data run exposed extreme log-target extrapolation (combined macro MAE 172.20). The model was then constrained to the range [0, maximum fitting target], learned independently inside each fit. First-run outputs remain in `failed_extrapolation/`. Because the guard was added after inspecting held-out results, the revised scores are exploratory, not untouched confirmatory evidence. No observation was removed from scoring. See `docs/EXPERIMENT_LOG.md` for the revision rationale.

## What these results mean

Point predictions and interval calibration face geographic shift. Nominal 90% split-conformal coverage is not guaranteed for dependent, nonexchangeable counties; measured coverage is the relevant diagnostic. Wide intervals or undercoverage are findings, not reasons to hide a fold. The bootstrap in `stability.json` is descriptive because training folds overlap and only nine counties exist.

Permutation importance measures predictive reliance, not causal effects. Socioeconomic slices do not establish clinical fairness. The learning curve uses fixed hyperparameters selected on the same development data and is a diagnostic, not an independent performance estimate. The shuffled-label control refits with fitting labels shuffled only within each fitting partition.

The original draft extract and v2 outputs remain under `reports/legacy`; reconciliation quantifies final-release differences. A new official source does not make already explored regional outcomes blinded. No result here establishes individual clinical risk, pollution causation, future forecasting, or readiness for public-health decisions. The final artifact has separate calibration counties but no fresh independent test set.
"""
    (output / "RESULTS.md").write_text(text)
