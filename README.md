# Bay Area Data Projects

Two data projects: one analytics and ML, one relational database design.

## 🌫️ [Bay Area Air Pollution & Health Impact Analysis](Bay_Area_Health_and_Pollution_Analysis/)

Does air pollution track with asthma and cardiovascular disease across Bay Area counties?

- Cleaned CalEnviroScreen environmental and health data (median imputation, feature filtering, scaling)
- County-level EDA, correlation analysis and a composite socioeconomic **vulnerability index**
- **Geospatial** PM2.5 mapping with GeoPandas
- **Random Forest** and **Gradient Boosting** regressors to predict asthma rates from pollution levels

`pandas` · `GeoPandas` · `seaborn` · `scikit-learn`

## 🚗 [Driving School Relational Database](SQl_Driving_School_Database/)

A MySQL database for a driving school, normalized to BCNF: ER modeling, schema design, and analytical queries over students, instructors, lessons, vehicles and payments.

`MySQL` · `ER modeling` · `normalization (BCNF)`
