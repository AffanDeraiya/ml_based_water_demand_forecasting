# Shared Feature Matrix and Chronological Split Specification

## Purpose

This specification defines the shared model-ready feature matrix and the time-based train, validation, and test partitions used by later forecasting models.

A presentation-ready walkthrough is available in [SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb](../notebooks/SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb).

## Shared feature matrix

The matrix is built from the validated monthly source data through the leakage-safe feature builder. It contains:

- 28 approved predictor columns
- the target kept separately as `residential_water_demand_m3`
- a unique, ascending month-start index
- no missing values after deterministic warm-up removal
- 168 complete rows, beginning in 2011-01

The first 12 source months are removed because the longest approved historical feature is a 12-month demand lag. No imputation or future-value filling is used.

## Chronological split

The complete 168-row matrix is split without shuffling:

| Split | Rows | Date range | Purpose |
|---|---:|---|---|
| Train | 108 | 2011-01 to 2019-12 | Fit model parameters and transformations |
| Validation | 24 | 2020-01 to 2021-12 | Tune and select modelling choices |
| Test | 36 | 2022-01 to 2024-12 | Final untouched evaluation |

The original project recommendation was 120/24/36 rows over the full 180-month source series. Because complete feature rows begin after the 12-month warm-up, the training portion is adjusted to 108 rows while preserving the final 36 months as the untouched test period. This is the least disruptive adjustment: it keeps the intended three-year test horizon and prevents incomplete warm-up rows from entering training.

## Rules

- The order is always train, then validation, then test.
- No random shuffling is permitted.
- Split boundaries are determined by date order, not target values.
- The test period is not used for feature decisions or tuning.
- Any fitted scaler, imputer, selector, or transformation must be fit on training data only.
- Quarterly outputs remain sums of sequential monthly forecasts; no separate quarterly split is created.

## Handoff outputs

The shared preparation code and explanatory notebook produce:

- the complete feature matrix
- the separate target series
- train, validation, and test views
- split counts and date boundaries
- a visual timeline of the chronological partitions
