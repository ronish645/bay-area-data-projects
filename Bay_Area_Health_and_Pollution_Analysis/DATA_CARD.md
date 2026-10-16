# Data card — official final release, Bay Area only

The main study now uses the **official CalEnviroScreen 4.0 final workbook**, downloaded from the [California Open Data Portal](https://publish.data.ca.gov/dataset/calenviroscreen-4-0/resource/9a90474a-3912-4955-9552-457ee41cea51). This is a deliberate fixed historical release, not a claim of current conditions or the latest release.

## Source and scope

The publisher's workbook contains 8,035 tracts in 58 counties. **The ingestion pipeline filters it to 1,581 tracts in the nine Bay Area counties before any modeling.** The processed dataset contains no other county. One Bay Area tract lacks an asthma target; 1,580 are labeled.

- Workbook SHA-256: `03f88d03e2fce5a04191dc65f2793e1dd8bf312b8d67a41e99ccf209b62b2f68`.
- Processed CSV SHA-256 and retrieval timestamp: `data/provenance.json`.
- The raw workbook is cached in the ignored `data/raw/` directory. The verified derived CSV and publisher dictionary are committed.
- The catalog states no restrictions on public use and does not specify a separate license. No new data license is assigned here.
- The original draft CSV is unchanged. `reports/draft_final_reconciliation.csv` compares matched tract records: asthma and PM2.5 agree; many other environmental values differ.

## Target and interpretation

The [official asthma indicator](https://oehha.ca.gov/calenviroscreen/indicator/asthma) describes spatially modeled, age-adjusted asthma emergency-department visits per 10,000 people, averaged over 2015–2017. It is not asthma prevalence, an individual probability, a count of affected residents, current hospital demand or a future outcome. The publisher's dictionary is preserved in `data/official_dictionary.csv`.

Geographic points are **2010 census tract centroids**. Other features have their own historical reference periods. Spatial modeling in the label can induce dependence even when fitting and test centroids are buffered.

## Inputs and transformations

Three allowlisted feature families: air/traffic (4); socioeconomic (5); combined (13 environmental + 5 socioeconomic). Environmental features are ozone, PM2.5, diesel PM, drinking water, lead, pesticides, toxic releases, traffic, cleanup sites, groundwater threats, hazardous waste, impaired water bodies and solid waste.

Predictor exclusions: target percentiles, population-characteristic and overall CES composites, pollution-burden composites, other health outcomes, all percentiles, dates, county/ZIP/tract identifiers, coordinates and the CSV row index. Coordinates are used only for geographic separation/description, not as predictors.

The loader normalizes identifiers/whitespace, checks schema and unique California tract IDs, requires valid coordinates and rejects infinite values/negative targets. Negative predictor placeholders become missing, though none occur in the pinned final extract. Missing targets remain in descriptive data but are excluded from supervised modeling. Predictor medians, missingness indicators and scaling are fitted separately inside each fitting partition.

County comparisons are unweighted means across tracts, with nonmissing denominators reported. They are not official population-aggregated county rates. Pollution burden is a broader environmental composite and must not be called air pollution alone.

## Integrity and limits

The source downloader refuses a changed workbook checksum until reviewed. The loader checks the processed checksum. Tests enforce all nine counties and no others. Split manifests preserve fitting, calibration and test IDs; all model roles stay inside the Bay Area.

Missingness may be informative. Exposure measurement error, differing reference periods, modeled labels and ecological aggregation remain limitations. This dataset cannot establish individual or causal health conclusions.
