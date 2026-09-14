# Monthly Forecast Feature Specification

## Purpose

This specification defines the baseline predictors for monthly DNH_total residential water-demand forecasting. It is written so the feature matrix can be reproduced consistently across later models.

A presentation-ready visual review of these decisions is available in [FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb](../notebooks/FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb), with its generated summary at `outputs/reports/feature_availability_audit.json`.

## Forecast-origin rule

For the target month `t`, a feature is allowed only when its value would be known at the forecasting issue date immediately before month `t` begins.

- Calendar features for month `t` are known in advance and may be used directly.
- Observed demand and operational measurements are treated conservatively as known through month `t-1`; they are therefore lagged by one month.
- No target-month weather, demographics, reservoir, groundwater, canal, or demand values are used directly.
- Rolling demand statistics are calculated from values through month `t-1` only.
- The target remains separate from the feature matrix.

## Approved baseline features

| Feature group | Columns | Availability policy |
|---|---|---|
| Calendar | `month`, `quarter`, `year`, `month_sin`, `month_cos` | Available directly for target month |
| Demand history | `demand_lag_1`, `demand_lag_2`, `demand_lag_3`, `demand_lag_6`, `demand_lag_12` | Prior observed demand only |
| Demand rolling history | `demand_rolling_mean_3`, `demand_rolling_std_3`, `demand_rolling_mean_6`, `demand_rolling_std_6` | Prior demand only; current month excluded |
| Weather history | rainfall, temperatures, humidity, wind, solar radiation, sunshine | One-month lag of observed values |
| System-state history | reservoir level, canal discharge, groundwater level | One-month lag of observed values |
| Demographic history | total population, urban population, total households | One-month lag of observed values |

The implementation uses the feature names `<source>_lag_1` for lagged non-target predictors and the explicit demand names shown above.

## Warm-up policy

The longest baseline lag is 12 months. The feature builder supports two modes:

- retain the full time index with leading `NaN` values for inspection
- drop incomplete warm-up rows when creating a model-ready matrix

Dropping warm-up rows is deterministic and removes the first 12 months only. No imputation or future-value filling is performed.

## Leakage checks

The implementation must satisfy these invariants:

1. Every demand lag at month `t` equals demand from an earlier month.
2. Every rolling demand statistic at month `t` excludes demand from month `t`.
3. Every lagged operational or exogenous feature at month `t` comes from an earlier month.
4. Changing a future observation must not change any earlier feature row.
5. Feature order and names are fixed and reproducible.

## Deferred decisions

Scaling, imputation beyond the deterministic warm-up policy, feature selection, and model-specific transformations are deferred until after the chronological split. Any fitted transformation must use training data only.
