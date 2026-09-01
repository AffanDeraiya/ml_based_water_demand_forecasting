# Implementation Plan — DNH Monthly Residential Water-Demand Forecasting (Version 2)

## Project direction

The implementation now models one combined **DNH-total monthly residential-demand** series. This replaces the earlier demand-zone design, because the confirmed real-data spatial level is DNH total. The target remains m³/month. The approved canonical schema is in [DATA_CONTRACT.md](DATA_CONTRACT.md); all work must conform to it.

## Phase 1 — Research design and governance — complete

Deliverable: [RESEARCH_DESIGN_DOCUMENT.md](RESEARCH_DESIGN_DOCUMENT.md).

It defines the target, horizons, scope, model-comparison approach, leakage controls, role of K-Means, and limits of synthetic evaluation.

## Phase 2 — Data contract — complete

Deliverable: [DATA_CONTRACT.md](DATA_CONTRACT.md).

It defines the 17-column monthly DNH-total canonical dataset, source-to-month aggregation rules, exclusions, quality gates, and the real-data replacement path.

## Phase 3 — Synthetic data generation — ready to implement

Deliverable: a validated, reproducible 15-year / 180-row monthly DNH-total dataset, its metadata, generator code, configuration, and automated tests.

The detailed, self-contained build specification is [PHASE_3_SYNTHETIC_DATA_GENERATION.md](PHASE_3_SYNTHETIC_DATA_GENERATION.md). It replaces the previous zone-level generator specification. Implement the Version 2 generator and validate it before feature engineering begins.

## Phase 4 — Exploratory data analysis and feature engineering

1. Load and validate the Phase 3 output through one reusable data-loading path.
2. Produce data-quality, distributions, trends, seasonality, and correlation plots.
3. Build strictly leakage-safe features: calendar variables, demand lags (for example 1, 3, 6, 12 months), rolling features using past observations only, and documented lagged operational variables.
4. Create chronological train/validation/test splits (default 10/2/3 years).
5. Persist feature definitions and split boundaries for reproducibility.

Deliverables: EDA notebook/script, feature-building module, data-split artifact, figures, and validation tests.

## Phase 5 — Baselines and forecasting models

1. Establish a seasonal-naive baseline before complex models.
2. Train/tune Random Forest using the common feature matrix.
3. Train/tune an ANN with training-only scaling.
4. Train/tune an LSTM using correctly ordered sequences and training-only scaling.
5. Forecast one month ahead and produce three-month sequential forecasts for the quarterly total.

Deliverables: reproducible model code, saved configurations/models as appropriate, forecasts, and training diagnostics.

## Phase 6 — Evaluation and demand-regime analysis

1. Evaluate all models on the untouched chronological test period with MAE, RMSE, sMAPE, and R² where appropriate.
2. Compare against the seasonal-naive baseline; report uncertainty/limitations rather than generic “accuracy percentage.”
3. Use K-Means only as an exploratory clustering of monthly demand regimes, with scaling and clustering fit on the training period where the analysis feeds a predictive workflow.
4. Produce forecast-versus-actual, residual, seasonal, and model-comparison visuals.

Deliverables: comparison table/plots, K-Means interpretation, and reproducible evaluation outputs.

## Phase 7 — Reporting and reproducibility

1. Write a concise report describing the original planning problem, the baseline limitations, methodology, results, what the models improve, and limitations of synthetic data.
2. Provide a single command or documented sequence that recreates data, features, models, plots, and report artifacts.
3. Record package versions, seeds, configurations, and provenance.

Deliverables: README/runbook, final report, and reproducible outputs.

## Phase 8 — Real-data replacement (when history is available)

1. Ingest confirmed real data through the source-to-month rules in the data contract.
2. Verify residential-demand definition, units, coverage, and aggregation.
3. Re-run validation, EDA, model selection, and final evaluation using real chronological holdout data.
4. Clearly separate synthetic pipeline evidence from real-world performance evidence.

## Non-negotiable controls

- No random train/test split for time-series forecasting.
- No feature may use target-month information unavailable when a forecast is issued.
- No zone-level geography in Version 2 modelling or K-Means claims.
