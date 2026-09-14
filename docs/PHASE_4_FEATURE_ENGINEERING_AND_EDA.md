# Phase 4 — Exploratory Data Analysis and Feature Engineering

## Purpose

Phase 4 converts the validated 15-year DNH-total monthly dataset into a leakage-safe, model-ready forecasting dataset. The work in this phase must preserve the project contract, respect time-series forecasting rules, and prepare a shared feature matrix for later model comparison.

This phase is intentionally limited to:
- validating the Phase 3 output via a single reusable loading path
- running EDA to understand distributions, trend, seasonality, and relationships
- engineering only features available at forecast origin
- splitting data chronologically for reproducible training, validation, and test runs

It does not include final model training or evaluation yet. Those are Phase 5 and later.

---

## Governing rules

The following requirements are non-negotiable:

- Use the canonical schema defined in [DATA_CONTRACT.md](DATA_CONTRACT.md).
- Do not use random train/test splits for time-series forecasting.
- No feature may use target-month information unavailable at the forecast issue date.
- Do not build a separate quarterly model; quarterly output remains the sum of three sequential monthly forecasts.
- Use only DNH_total, not zone-level or sub-area features.
- Persist the feature list, split boundaries, transformations, and metadata for reproducibility.

---

## Phase 4 objectives

1. Load and validate the generated Phase 3 output in a single reusable way.
2. Understand demand behaviour, seasonal structure, trend, and system interactions.
3. Create a clean feature matrix that captures temporal and seasonal structure without leakage.
4. Prepare a chronological split that keeps the project’s evaluation protocol intact.
5. Save artifacts needed for Phase 5 model work.

## Current implementation status

- [x] Tasks 4.1 and 4.2 are implemented and verified through the reusable loader in `src/features/phase4_data_loader.py`.
- [x] Task 4.3 is implemented in the presentation-ready EDA notebook at `notebooks/PHASE_4_TASK_4_3_EDA.ipynb`.
- [x] Task 4.4 is documented in `outputs/reports/dnh_total_data_quality_and_realism.md`.
- [x] Tasks 4.5 and 4.6 are implemented in `src/features/phase4_feature_engineering.py`, specified in `docs/FEATURE_ENGINEERING_SPECIFICATION.md`, and presented visually in `notebooks/FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb`.
- [x] Tasks 4.7 and 4.8 are implemented in `src/features/phase4_dataset_preparation.py`, specified in `docs/FEATURE_MATRIX_AND_SPLIT_SPECIFICATION.md`, and presented visually in `notebooks/SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb`.
- [x] Task 4.9 is implemented through reproducible feature, target, split, row-ID, and metadata artifacts under `data/processed/`.
- [x] Task 4.10 is implemented through `outputs/reports/phase4_handoff_manifest.json`, `outputs/reports/phase4_handoff_manifest.md`, and `notebooks/PREPARATION_HANDOFF_REVIEW.ipynb`.

Quality review update: the generator state dynamics were corrected and the canonical dataset was regenerated. Reservoir and groundwater now vary within their configured operating bands without boundary saturation. The refreshed dataset is ready for feature engineering, subject to the synthetic-data limitations recorded in the realism report.

---

## Task-by-task checklist

### Task 4.1 — Confirm and lock the canonical data path
- [x] Confirm the final source dataset path is the validated Phase 3 CSV.
- [x] Confirm metadata and config are recorded and reproducible.
- [x] Record the exact date range and row count.
- [x] Ensure the loader will read this same path across all later scripts.

Definition of done:
- one loader function or script can read the canonical dataset consistently
- the dataset has 180 rows for 15 years of monthly data

---

### Task 4.2 — Build a reusable data-validation loader
- [x] Create a reusable loader that reads the CSV and validates schema.
- [x] Validate all required columns are present.
- [x] Validate `date` is unique and sorted chronologically.
- [x] Validate `area_id == DNH_total` for all records.
- [x] Validate there are no nulls or non-finite values in required numeric columns.
- [x] Validate the required domain rules (rainfall non-negative, humidity in range, min <= max, etc.).
- [x] Standardize date type and index to monthly frequency.

Definition of done:
- any downstream Phase 4 or Phase 5 component can call the same validated loader
- the loader rejects invalid data before feature generation begins

---

### Task 4.3 — Run EDA on the target and core predictors
- [x] Inspect the target series: `residential_water_demand_m3`.
- [x] Check trend, seasonality, and overall range.
- [x] Inspect rainfall, temperature, humidity, wind, solar radiation, and sunshine patterns by month.
- [x] Compare monsoon vs dry-season behaviour.
- [x] Inspect population and households over time.
- [x] Check reservoir, groundwater, and canal discharge behaviour over time.
- [x] Plot the target vs major drivers to understand visible relationships.
- [x] Summarize correlations and likely lead-lag relationships.

Definition of done:
- a documented EDA summary exists for the dataset
- the main drivers and seasonal patterns are clearly identified
- the project team understands which variables matter most before modelling

---

### Task 4.4 — Document the data-quality and realism findings
- [x] Record whether the synthetic data behaves plausibly across the expected seasonal patterns.
- [x] State whether the dataset is realistic enough for the next phase.
- [x] Note any obvious anomalies, clipping, or oversmoothing in the series.
- [x] Identify whether additional realism fixes are still required before modelling.

The review is complete. The former reservoir and groundwater boundary-clipping issue was corrected before proceeding to feature engineering.

Definition of done:
- a short EDA summary is saved with the phase output or notebook
- any model-risking issues are explicitly documented before building features

---

### Task 4.5 — Create the time-indexed feature engineering plan
- [x] Decide which features are direct predictors and which are lagged features.
- [x] Define all calendar features to be included.
- [x] Define lag features for target and operational variables.
- [x] Define rolling features from past observations only.
- [x] Confirm that no target-month or future information is used.

The approved feature list and forecast-origin availability policy are recorded in `docs/FEATURE_ENGINEERING_SPECIFICATION.md`.

Recommended baseline feature groups:
- calendar: month, quarter, year, cyclical month encoding
- target lags: 1, 2, 3, 6, 12 months
- rolling target stats: 3-month and 6-month rolling mean/std
- weather: rainfall, temp_max, temp_min, humidity_max, humidity_min, wind, solar, sunshine
- system state: reservoir, canal discharge, groundwater
- demographics: total population, urban population, total households

Definition of done:
- feature list is fixed and documented before model work begins
- features are explicitly labelled as “available at forecast origin” versus “not allowed”

---

### Task 4.6 — Build leakage-safe lag and rolling features
- [x] Create lagged demand features using only prior months.
- [x] Create lagged weather/system-state features using prior or contemporaneous values known before the target month, as appropriate.
- [x] Create rolling demand statistics using past observations only.
- [x] Ensure any windowed feature is computed over prior data only.
- [x] Confirm there are no look-ahead features accidentally included.

The implementation is covered by `tests/test_phase4_feature_engineering.py`, including direct perturbation checks that future observations cannot change earlier feature rows.

Definition of done:
- feature matrix has no direct future leakage
- lag calculations are reproducible and documented

---

### Task 4.7 — Prepare the shared feature matrix
- [x] Combine all approved feature columns into one feature dataframe.
- [x] Keep the target variable separate as `residential_water_demand_m3`.
- [x] Preserve the date index and order.
- [x] Avoid introducing missing values from lag/rolling features without a documented strategy.
- [x] Store the final feature matrix in a consistent structure for later model use.

The shared matrix contains 168 complete rows and 28 predictors after the documented 12-month warm-up removal.

Definition of done:
- one feature matrix is ready for all Phase 5 models
- the target and predictors are clearly separated

---

### Task 4.8 — Define chronological train/validation/test splits
- [x] Select the default split: training / validation / test in time order.
- [x] Use the documented default split if no special reason is provided.
- [x] Lock the exact train/validation/test month boundaries.
- [x] Ensure no data leakage between periods.
- [x] Record the date ranges for each split.

Because the first 12 months are removed for complete feature history, the implemented complete-row split is train 108 months (`2011-01` to `2019-12`), validation 24 months (`2020-01` to `2021-12`), and test 36 months (`2022-01` to `2024-12`).

Recommended default split:
- Train: first 120 months (10 years)
- Validation: next 24 months (2 years)
- Test: final 36 months (3 years)

Definition of done:
- split boundaries are fixed and saved
- all future model work uses the same chronological split

---

### Task 4.9 — Create reproducible split artifacts
- [x] Save index lists or row IDs for train/validation/test sets.
- [x] Save the split boundaries and date ranges.
- [x] Save the final feature matrix and target series in reproducible form.
- [x] Save a metadata dictionary describing why the split was chosen.

The saved metadata includes feature columns, target name, warm-up policy, split rationale, row IDs, date boundaries, and the no-shuffle/no-future-fill guarantees.

Definition of done:
- later model code can load the same split without re-deriving it
- the split is fully traceable and repeatable

---

### Task 4.10 — Prepare the Phase 4 handoff package
- [x] Save the final EDA outputs, summary notes, and plots.
- [x] Save the feature engineering script/module.
- [x] Save the prepared training/validation/test datasets and split metadata.
- [x] Confirm all artifacts are ready for Phase 5 baseline and forecasting work.

The handoff manifest checks the complete package and the review notebook presents its contents and limitations.

Definition of done:
- Phase 5 can begin without reworking the feature pipeline
- feature definitions and data splits are stable and documented

---

## Deliverables expected at the end of Phase 4

The following outputs are expected before moving to Phase 5:

- reusable dataset loader/validator
- EDA summary and plots
- leakage-safe feature engineering script or module
- final feature matrix and target series
- time-based train/validation/test split artifacts
- metadata describing the split and feature list
- explicit confirmation that no leakage is present

---

## Exit criteria for Phase 4

Phase 4 is complete only when all of the following are true:

- [x] the Phase 3 dataset is loaded and validated through a single path
- [x] EDA has been performed and summarized
- [x] feature engineering is leakage-safe and documented
- [x] all features are available at forecast origin
- [x] the chronological train/validation/test split is fixed and reproducible
- [x] artifact outputs are stored for Phase 5 use

---

## Phase 5 handoff

Once the checks above are complete, Phase 5 can begin with:
- seasonal-naive baseline
- Random Forest model
- ANN model
- LSTM model
- evaluation on the untouched test period with MAE, RMSE, sMAPE, and R² where appropriate

---

## Notes for implementation discipline

- Keep Phase 4 modular; do not mix EDA, feature creation, and final model code in one script without clear separation.
- Prefer reproducible scripts over ad hoc notebook-only work when the logic is likely to be reused.
- All transformations must be fit only on training data where fitting is required.
- Treat the validation period as a tuning and selection tool, not as a final model report period.

This plan should be reviewed before implementation, and only after approval should Phase 4 coding begin.
