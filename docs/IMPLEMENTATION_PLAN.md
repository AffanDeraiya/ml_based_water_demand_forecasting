# End-to-End Implementation Plan — DNH Water-Demand Forecasting

## 1. Project purpose

Build a reproducible machine-learning system that forecasts **residential water demand in Dadra and Nagar Haveli (DNH)**. The project will first use realistic synthetic data to validate the complete pipeline, then replace it with real DNH data when it becomes available.

The research paper provides the academic justification and conceptual framework. The DAC-1 presentation provides the initial implementation direction: collect data, preprocess it, train Random Forest/LSTM/ANN models, and evaluate their performance.

## 2. Scope to preserve

- **Current target:** monthly residential water demand.
- **Forecast horizons:** monthly and quarterly.
- **Initial input candidates:** rainfall, population, historical consumption, area-wide water usage where available, and calendar/seasonality features.
- **Core models:** seasonal-naive baseline, Random Forest, ANN, and LSTM.
- **Data-mining component:** K-Means clustering, used only in a defensible and interpretable way.
- **Initial sector scope:** residential only.
- **Data source strategy:** realistic synthetic data first; real DNH data later.

Do not silently expand the project to crop-specific, agricultural, industrial, farm-level, or multi-sector forecasting.

## 3. Core implementation principles

- Treat this as a research-grade forecasting project, not only a model-training exercise.
- Use chronological training, validation, and test splits; do not randomly split time-series data.
- Prevent data leakage: every model feature must be known at the time its forecast is made.
- Keep synthetic and real datasets compatible through one documented data contract.
- Use fixed random seeds, versioned configuration, experiment tracking, and reproducible results.
- Keep synthetic-data results clearly separate from real-world performance claims.

## 4. Phase 1 — Research design and experiment specification

Before coding, document the decisions that determine the dataset, models, and evaluation:

1. Confirm the prediction unit and target unit, for example cubic metres per month.
2. Confirm the data grain:
   - one DNH-wide record per month, or
   - one record per zone/ward per month.
3. Define the one-step monthly forecasting task and forecast origin.
4. Decide that quarterly forecasting initially aggregates three consecutive monthly forecasts.
5. Identify variables that are available at forecast time.
6. Define the chronological train/validation/test methodology.
7. Define primary and secondary evaluation metrics.
8. Specify the fair model-comparison protocol.
9. Define the K-Means objective based on the data grain.
10. State assumptions, risks, exclusions, and success criteria.

The output is a short research-design document that is approved before implementation starts.

### Version 1 working assumptions

The following assumptions allow implementation to begin before the real dataset is confirmed. They are not claims about available DNH data and must be revisited when the team confirms the real-data arrangement.

1. **Spatial grain:** create synthetic monthly data for multiple DNH demand zones/areas and also aggregate it to a DNH-total view. This makes K-Means useful for demand-zone analysis. If real data is DNH-wide only, use a DNH-total forecast and reposition K-Means as exploratory demand-regime analysis.
2. **Target and unit:** define the target as `residential_water_demand_m3`, residential water demand in cubic metres per month. Treat `water_consumption_m3` as a distinct historical feature. Only lagged/past consumption may be used for forecasting.
3. **Synthetic historical period:** generate 15 years of monthly synthetic data because no real-data source or historical coverage is confirmed yet.
4. **Quarterly forecast:** Version 1 will sum three consecutive monthly forecasts. A separately trained quarterly model is a later comparison experiment, if required.
5. **Technical scope:** use open-source technology without a dashboard or web application. The initial final deliverable is a reproducible pipeline, model-comparison results, plots, and a concise report.
6. **Default modelling design:** train pooled/global models across the five synthetic zones while preserving separate chronological series for each zone. Report overall and per-zone metrics; do not train separate per-zone LSTMs in Version 1.
7. **Default split:** for the 15-year synthetic dataset, use years 1–10 for training, years 11–12 for validation, and years 13–15 as the untouched test period.

## 5. Phase 2 — Data contract

Create a documented schema that synthetic and later real data must follow.

### Initial canonical fields

```text
date
area_id
rainfall_mm
population
water_consumption_m3
area_water_usage_m3
residential_water_demand_m3
```

Version 1 uses neutral synthetic zone IDs; the same field can later represent real wards, villages, supply zones, or `DNH_total` without redesigning the pipeline.

### Data-contract requirements

- Units, valid ranges, and allowed data types.
- Data dictionary and source/assumption documentation.
- Definition of the target variable.
- Rules for missing, duplicate, and invalid values.
- Explicit designation of each variable as historical, contemporaneous, or future-known.
- Automated data-validation checks.

## 6. Phase 3 — Realistic synthetic data

Generate a reproducible monthly dataset, ideally spanning 10–15 years.

It must model plausible temporal relationships rather than independent random columns:

- monsoon rainfall seasonality;
- gradual population growth;
- underlying demand growth;
- seasonal residential-demand patterns;
- lagged demand and consumption effects;
- realistic random variation;
- optional controlled anomalies and missing values for preprocessing tests.

Specify every generated relationship and plausible range in a versioned configuration: rainfall, population, demand, growth, zone variation, noise, missingness, and anomalies. Avoid unexplained constants.

If K-Means is to identify demand zones, the synthetic data must include multiple zones with meaningfully different demand patterns. Store the random seed and generation assumptions.

## 7. Phase 4 — Common data pipeline

Implement reusable components:

```text
Raw data
  → schema validation
  → cleaning and missing-value handling
  → chronological sorting
  → feature engineering
  → train / validation / test split
  → model-ready datasets
```

Candidate features include:

- lagged demand: 1, 3, and 12 months;
- lagged rainfall and consumption;
- rolling demand and rainfall averages;
- month and quarter;
- cyclical month encoding using sine/cosine;
- population trend/growth features;
- cluster-derived information only if justified.

For neural models, create sequence/window datasets. Fit scaling only on the training period, then apply it to validation and test periods.

## 8. Phase 5 — K-Means clustering

K-Means must be useful and interpretable.

### When zone-level data exists

Cluster zones using stable historical characteristics, such as average demand, peak demand, demand variability, rainfall sensitivity, and population growth/density. Select the number of clusters using methods such as silhouette score and explain clusters in planning terms.

Fit all clustering transforms and choose the cluster count using training-period data only. Freeze zone assignments before validation and test evaluation.

### When only one DNH-wide series exists

Use K-Means as exploratory demand-regime analysis, such as low/medium/high demand months. Do not treat an arbitrary cluster label as a forecasting feature unless comparative experiments show a real benefit.

Compare cluster-informed and non-cluster-informed forecasting where applicable.

## 9. Phase 6 — Forecasting benchmark and models

Implement a benchmark before the ML models:

- Seasonal-naive forecast: use demand from the equivalent month one year earlier.
- Optional moving-average or trend baseline.

Then train the confirmed models:

- **Random Forest:** tabular ML benchmark using engineered features.
- **ANN:** feed-forward neural network using aligned tabular features.
- **LSTM:** sequence model trained on historical windows.

Every model must use the same forecast origins and only information available at those origins. XGBoost may be added later as a clearly labelled enhancement after the core models are complete.

For Version 1 quarterly output, produce a three-month forecast recursively from one origin, then sum those predictions. Do not use future observed demand, rainfall, consumption, or usage to form later monthly predictions.

## 10. Phase 7 — Evaluation and model selection

Use a final untouched test period. Tune models only using the chronological training and validation periods.

### Metrics

- Primary: MAE and RMSE.
- Secondary: MAPE or sMAPE, and R-squared where meaningful.

### Required analyses

- Overall model comparison.
- Actual-versus-predicted plots.
- Residual/error plots.
- Seasonal performance, especially monsoon versus non-monsoon months.
- Stability across validation periods.
- Training/inference cost and implementation complexity.

Select the best model using accuracy, stability, interpretability, cost, and deployability—not one metric alone.

For quarterly forecasts, initially sum three sequential monthly forecasts. A standalone quarterly model can later be implemented as a controlled comparison experiment.

## 11. Phase 8 — Explainability and decision outputs

For the selected model:

- report feature importance for Random Forest;
- use permutation importance or SHAP where appropriate;
- create a forecast report with expected demand, historical comparison, risk/seasonal context, and limitations;
- phrase outputs as decision support, not automatic water-allocation decisions.

## 12. Phase 9 — Reproducible project packaging

Recommended project structure:

```text
data/
  raw/
  synthetic/
  processed/
  real/
src/
  data_generation/
  validation/
  preprocessing/
  features/
  clustering/
  models/
  evaluation/
  reporting/
configs/
notebooks/
tests/
outputs/
  models/
  figures/
  metrics/
  reports/
```

Include a README, environment/dependency specification, fixed seeds, configuration files, experiment records, and tests for the data and feature logic.

## 13. Phase 10 — Real-data integration

When real DNH data becomes available:

1. Map it to the data contract.
2. Run the same data-validation and preprocessing pipeline.
3. Revisit the spatial grain and K-Means approach.
4. Retrain and evaluate every model.
5. Keep synthetic and real-data results separate in reports, thesis work, and publications.

## 14. Recommended delivery sequence

1. Research-design document.
2. Data contract and data dictionary.
3. Synthetic-data generator.
4. Validation, preprocessing, and feature pipeline.
5. Baseline and Random Forest experiments.
6. ANN and LSTM experiments.
7. K-Means analysis.
8. Evaluation, explainability, and reporting.
9. Real-data integration when data is obtained.

## 15. Decisions still needing confirmation

1. Exact target unit and definition of residential demand.
2. Whether data will be DNH-wide only or available by ward/zone.
3. Whether quarterly forecasting should remain aggregate-only in the first version.
4. Historical period and frequency that will be available in real data.
5. Whether water consumption and residential demand are distinct measurements in the real data, and which is known before the forecast date.
