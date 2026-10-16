"""Publication-style figures and a self-contained, offline results explorer."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayair.data import PROJECT

BLUE, TEAL, INK, GRAY = "#3159d9", "#147d79", "#152339", "#68768b"


def render(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    board = pd.read_csv(output / "leaderboard.csv")
    folds = pd.read_csv(output / "county_metrics.csv")
    pred = pd.read_csv(output / "selected_predictions.csv", dtype={"tract": str})
    importance = pd.read_csv(output / "permutation_importance.csv")
    audit = json.loads((output / "data_audit.json").read_text())
    run = json.loads((output / "run.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK,
                         "text.color": INK, "figure.facecolor": "#ffffff",
                         "axes.titleweight": "bold", "font.size": 11, "savefig.dpi": 170})

    def save(name):
        plt.tight_layout()
        plt.savefig(figures / name, bbox_inches="tight")
        plt.close()

    b = board[board.feature_set == "combined"].sort_values("macro_county_mae")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(b.model, b.macro_county_mae, color=[TEAL if x == "selected" else BLUE for x in b.model])
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error · equal weight per held-out county", title="Generalization to an unseen county")
    for i, value in enumerate(b.macro_county_mae):
        ax.text(value + .25, i, f"{value:.2f}", va="center")
    ax.set_xlim(0, b.macro_county_mae.max() * 1.15)
    save("leaderboard.png")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ablation = board[board.model == "selected"].set_index("feature_set").loc[["air", "social", "combined"]]
    ax.bar(["Air pollutants + traffic", "Socioeconomic indicators", "Combined"], ablation.macro_county_mae, color=[GRAY, BLUE, TEAL])
    ax.set(ylabel="Macro county MAE · lower is better", title="What does each feature family contribute?")
    for i, value in enumerate(ablation.macro_county_mae):
        ax.text(i, value + .3, f"{value:.2f}", ha="center")
    save("ablation.png")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    county = folds[(folds.feature_set == "combined") & folds.model.isin(["selected", "median"])].pivot(index="county", columns="model", values="mae").sort_values("selected")
    county.rename(columns={"median": "Median baseline", "selected": "Inner-selected model"}).plot.barh(ax=ax, color=[GRAY, TEAL])
    ax.set(xlabel="Held-out MAE", ylabel="", title="Average gains hide county-level failures")
    ax.legend(loc="lower right", frameon=False)
    save("county_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(pred.actual, pred.predicted, s=11, c=BLUE, alpha=.4, edgecolors="none", rasterized=True)
    extent = [min(pred.actual.min(), pred.predicted.min()), max(pred.actual.max(), pred.predicted.max())]
    axes[0].plot(extent, extent, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set(xlabel="Observed asthma indicator", ylabel="Out-of-fold prediction", title="Observed vs. predicted")
    axes[1].scatter(pred.predicted, pred.residual, s=11, c=TEAL, alpha=.4, edgecolors="none")
    axes[1].axhline(0, color=GRAY, linestyle="--", linewidth=1)
    axes[1].set(xlabel="Out-of-fold prediction", ylabel="Observed − predicted", title="Where does the model miss?")
    save("residuals.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    dots = ax.scatter(pred.Longitude, pred.Latitude, c=pred.absolute_error, cmap="magma_r", s=14, alpha=.8)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Geography of held-out prediction error")
    ax.set_aspect(1 / np.cos(np.deg2rad(pred.Latitude.mean())))
    fig.colorbar(dots, ax=ax, shrink=.65, label="Absolute error · asthma indicator units")
    ax.text(.02, .01, "Tract representative points; no boundary polygons", transform=ax.transAxes, fontsize=9, color=GRAY)
    save("error_map.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    imp = importance.groupby("feature").mae_increase.agg(["mean", "std"]).sort_values("mean")
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=BLUE, capsize=3)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set(xlabel="Held-out MAE increase after shuffling · mean ± SD across counties",
           title="Predictive reliance, not causal effect")
    save("importance.png")

    selected = board[(board.feature_set == "combined") & (board.model == "selected")].iloc[0]
    base = board[(board.feature_set == "combined") & (board.model == "median")].iloc[0]
    reduction = 100 * (1 - selected.macro_county_mae / base.macro_county_mae)
    payload = {"leaderboard": board.to_dict("records"), "folds": folds.drop(columns="best_params").to_dict("records"),
               "predictions": pred[["tract", "county", "actual", "predicted", "absolute_error", "Longitude", "Latitude", "model"]].to_dict("records"),
               "importance": imp.reset_index().fillna(0).to_dict("records"), "audit": audit}
    template = (Path(__file__).parent / "report_template.html").read_text()
    html = template.replace("__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))
    html = html.replace("__ROWS__", str(audit["model_rows"])).replace("__MAE__", f"{selected.macro_county_mae:.2f}")
    html = html.replace("__R2__", f"{selected.pooled_r2:.3f}").replace("__GAIN__", f"{reduction:.1f}%")
    (output / "index.html").write_text(html)
    results = f"""# Experiment results

Generated from `{audit['sha256']}` using seed 42. See `run.json` for exact versions.

**Primary result:** the inner-selected combined-feature procedure achieves macro county MAE **{selected.macro_county_mae:.2f}**, versus **{base.macro_county_mae:.2f}** for the training-median baseline: **{reduction:.1f}%** lower MAE. Pooled out-of-fold R² is **{selected.pooled_r2:.3f}**. Pooled and equal-county metrics weight geography differently.

## Predeclared feature ablation

| Features | Macro county MAE | Pooled R² | Worst county MAE |
|---|---:|---:|---:|
"""
    for row in ablation.reset_index().itertuples():
        results += f"| {row.feature_set} | {row.macro_county_mae:.2f} | {row.pooled_r2:.3f} | {row.worst_county_mae:.2f} |\n"
    results += "\n## Held-out counties (combined, inner-selected)\n\n| County | Tracts | MAE | R² | Selected family |\n|---|---:|---:|---:|---|\n"
    for row in folds[(folds.feature_set == "combined") & (folds.model == "selected")].sort_values("mae", ascending=False).itertuples():
        family = pred.loc[pred.county == row.county, "model"].iloc[0]
        results += f"| {row.county} | {row.n_test} | {row.mae:.2f} | {row.r2:.3f} | {family} |\n"
    results += """
## Interpretation

The feature ablation is observational: adding socioeconomic variables may improve prediction without establishing causation. Individual county R² can be negative, even when pooled R² is positive. Equal-county MAE is the primary metric; large counties otherwise dominate the pooled score. The nested-selected row evaluates model-family selection inside each training fold. Fixed-family rows are exploratory comparisons and must not be used to select a family and claim an unbiased final score on these same folds.

The separate `split_diagnostic.csv` compares a fixed, untuned boosting model under random five-fold and county-held-out splits. Training sizes differ, so that gap is not a pure estimate of spatial leakage. County boundaries do not create a spatial buffer; nearby tracts across a border and spatially modeled labels can still share information.

Permutation importance uses held-out counties (five shuffles per feature). Correlated predictors can share or mask importance. The error bars describe county variation, not inferential confidence intervals. The county bootstrap range in `stability.json` is descriptive: just nine counties and overlapping training sets limit uncertainty estimates.

The final full-data model is a demonstration artifact, with no fresh independent test set. No result is a forecast, individual diagnosis, causal pollution effect, or validation against current clinical outcomes. The checked-in extract has DRAFT CES columns and undocumented dates; its exact correspondence to the final official release remains unverified.
"""
    (output / "RESULTS.md").write_text(results)
