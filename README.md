# Bay Area Data Projects

Projects investigating Bay Area environmental health and relational database design.

## [BayAir: pollution, asthma, and the limits of prediction](Bay_Area_Health_and_Pollution_Analysis/)

**Where is pollution higher? Where are asthma emergency visits higher? Can a model trained on some Bay Area counties estimate the pattern in another?**

BayAir studies **1,581 census tracts across nine Bay Area counties** using the official CalEnviroScreen 4.0 final release. It combines county comparisons, correlation heatmaps and geographic analysis with a reproducible ML experiment that holds out entire counties, enforces 5 km centroid buffers and reserves separate counties for uncertainty calibration.

![Bay Area pollution and asthma county comparisons](Bay_Area_Health_and_Pollution_Analysis/reports/figures/county_comparison.png)

The findings are instructive: Napa has the highest mean PM2.5, Solano has the highest mean asthma emergency-visit indicator, and different pollutants have different county leaders. Socioeconomic inputs outperform pollution-only inputs in the exploratory model comparison; the combined model does not beat a simple baseline on equal-county MAE.

The project includes verified source data, model and data cards, held-out explanations, prediction-range diagnostics, a preserved extrapolation failure, **34 passing Python tests**, an executed notebook, a CLI and a VS Code workspace. Measurements are historical tract averages; the analysis does not establish causal or individual health effects.

**[Live report](https://ronish645.github.io/bay-area-data-projects/)** · [Read the project](Bay_Area_Health_and_Pollution_Analysis/) · [Open the notebook](Bay_Area_Health_and_Pollution_Analysis/Cal_Bay_Area_PollutionAndHealth_analysis.ipynb) · [Inspect the results](Bay_Area_Health_and_Pollution_Analysis/reports/RESULTS.md)

`Python` · `pandas` · `scikit-learn` · `Matplotlib` · `pytest` · `VS Code`

## [Driving School Relational Database](SQl_Driving_School_Database/)

A MySQL database for a driving school, normalized to BCNF. Covers ER modeling, schema design and analytical queries over students, instructors, lessons, vehicles and payments.

`MySQL` · `ER modeling` · `normalization (BCNF)`
