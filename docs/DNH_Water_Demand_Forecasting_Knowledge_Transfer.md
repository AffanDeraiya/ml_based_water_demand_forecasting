# Knowledge Transfer — DNH Residential Water-Demand Forecasting

## Purpose

This is the restart document for this project. If conversation context is lost, read this file first, then the linked authoritative documents. It records the current approved scope and phase; it is not a historical transcript of earlier ideas.

## Project objective

Build a reproducible machine-learning pipeline that forecasts **combined monthly residential water demand for Dadra and Nagar Haveli (DNH)**. The system will first use realistic, reproducible synthetic data and later accept confirmed real historical data through the same canonical schema.

The forecast supports water allocation, supply/storage planning, seasonal preparedness, and water-management decisions. It is a forecast-support system, not an automated allocation decision-maker.

## Source material and its role

- `Imtiyaz DAC-1 (2).pptx` provides the research implementation direction: collect data, preprocess it, develop Random Forest/ANN/LSTM models, and evaluate them.
- `Padarshi_Thakar_AgriWaterDemand_Springer_LNCS (2).docm` and its text extraction provide literature-review and meta-analysis context. They justify data-mining and forecasting ideas but are not an implementation specification.
- `DOC-20260815-WA0016.xlsx` is a schema/example workbook, not a complete 10–15 year historical modelling dataset.

The original material discusses agricultural demand and geographic zones. The confirmed implementation scope below takes precedence whenever it differs.

## Confirmed scope and decisions

- **Spatial grain:** one combined DNH value per calendar month; `area_id` is always `DNH_total`.
- **Target:** `residential_water_demand_m3`, measured in cubic metres per month (m³/month).
- **Forecast horizons:** one-month-ahead forecasting, plus quarterly demand calculated as the sum of three sequential monthly forecasts. Do not build a separate quarterly model in Version 1.
- **Sector:** residential demand only. Do not include agricultural, industrial, crop, village, ward, or `pateload` modelling.
- **Synthetic history:** default 15 years / 180 consecutive monthly rows, initially `2010-01-01` to `2024-12-01`.
- **Core model comparison:** seasonal-naive baseline, Random Forest, ANN, and LSTM.
- **K-Means:** retain only as a later supplementary exploration of monthly demand regimes. It is not geographic clustering, a required feature, or a substitute for forecasting.
- **Platform:** open-source Python tooling. No dashboard/web application is required.
- **Output expectations:** reproducible pipeline, validated data, model-comparison results, plots, and a concise report if required.

## Canonical data contract

The data is one DNH-total monthly time series with exactly these columns, in this order:

```text
date
area_id
rainfall_mm
temp_max_c
temp_min_c
humidity_max_pct
humidity_min_pct
wind_speed_kmh
solar_radiation_mj_m2
sunshine_hours
total_population
urban_population
total_households
reservoir_level_m
canal_discharge_cumecs
groundwater_level_m_bgl
residential_water_demand_m3
```

The complete definitions, aggregation rules, exclusions, quality gates, and leakage requirements are authoritative in [DATA_CONTRACT.md](DATA_CONTRACT.md).

Do not add these fields to the Version 2 modelling dataset unless the project is explicitly redesigned:

- zones, villages, wards, `pateload`, or hour-level fields;
- agricultural fields or groundwater-quality values (`pH`, `TDS`);
- `net_annual_groundwater_availability_mcm` or `annual_groundwater_draft_mcm`;
- water-consumption or area-wide-usage fields. They are excluded because a consistent monthly residential definition has not been confirmed.

## Synthetic-data design

The Phase 3 generator must produce non-random-looking but synthetic data. It must use a supplied seed and create these relationships:

1. Monsoon-led rainfall, humidity, sunshine, solar radiation, temperature, and wind patterns.
2. Gradual growth in total population, urban population, and households.
3. Residential demand driven by demographic scale, weather/seasonality, prior demand, and bounded noise/shocks.
4. Reservoir and groundwater levels with gradual recharge/withdrawal dynamics responding to rainfall and demand/use.
5. Seasonal, supply-aware canal discharge that is not a direct copy of rainfall or demand.
6. Positive monthly demand in m³, with no missing values and no deterministic target formula.

The standalone build instructions, tests, validation rules, file locations, metadata requirements, and acceptance checklist are in [PHASE_3_SYNTHETIC_DATA_GENERATION.md](PHASE_3_SYNTHETIC_DATA_GENERATION.md).

## Forecasting and evaluation rules

- Use chronological train/validation/test periods only. The default synthetic split is 10 years training, 2 years validation, and 3 years test.
- Never random-split the time series.
- Predict month *t* with only information available at the forecast origin. Use lags for historical demand and operational variables when target-month observations would not be available.
- Fit scaling, imputation, feature selection, and tuning on training data only.
- Compare against a seasonal-naive baseline before reporting complex-model results.
- Report MAE, RMSE, sMAPE, and R² where meaningful. Do not use a generic classification-style “accuracy percentage” for continuous demand forecasts.
- Synthetic-data results demonstrate pipeline behaviour only; practical performance claims require a sufficiently long real historical holdout period.

## Phase status

### Phase 1 — Research design: complete

[RESEARCH_DESIGN_DOCUMENT.md](RESEARCH_DESIGN_DOCUMENT.md) defines the scope, research questions, forecasting approach, K-Means role, evaluation approach, and real-data limitations. No unanswered Phase 1 design question blocks implementation.

### Phase 2 — Data contract: complete

[DATA_CONTRACT.md](DATA_CONTRACT.md) fixes the Version 2 monthly DNH-total schema, units, source-mapping guidance, exclusions, and quality rules. This is the implementation contract for every Phase 3 output.

### Phase 3 — Synthetic generator code rework: next

The next logical task is to replace the earlier zone-level generator path with the Version 2 DNH-total generator described in [PHASE_3_SYNTHETIC_DATA_GENERATION.md](PHASE_3_SYNTHETIC_DATA_GENERATION.md). It must produce the canonical CSV, metadata, reusable validator, configuration, and automated tests before Phase 4 begins.

Do not begin modelling or feature engineering until Phase 3 passes its acceptance checklist.

## Project roadmap

1. **Phase 1 — Research design:** complete.
2. **Phase 2 — Data contract:** complete.
3. **Phase 3 — Synthetic generation:** implement and validate the Version 2 DNH-total generator.
4. **Phase 4 — EDA and feature engineering:** validate data again, explore patterns, create leakage-safe calendar and lag features, and create chronological splits.
5. **Phase 5 — Forecasting models:** baseline, Random Forest, ANN, LSTM, monthly forecasts, and sequential quarterly aggregation.
6. **Phase 6 — Evaluation and demand-regime analysis:** compare models and conduct supplementary K-Means demand-regime exploration.
7. **Phase 7 — Reporting and reproducibility:** figures, runbook, report, configuration/version recording.
8. **Phase 8 — Real-data replacement:** ingest actual data through the contract and rerun the full pipeline/evaluation.

The phase deliverables and controls are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Real-data replacement notes

When real history becomes available, aggregate it to the canonical DNH-total monthly grain. Confirm that any raw demand measure is residential before using it as the target. Retain m³/month unless a presentation requirement calls for a simple conversion to litres. Record whether solar radiation is a cumulative-energy or instantaneous measurement before selecting sum versus mean aggregation.

## Restart order for a new agent

1. Read this document.
2. Read [DATA_CONTRACT.md](DATA_CONTRACT.md) and [PHASE_3_SYNTHETIC_DATA_GENERATION.md](PHASE_3_SYNTHETIC_DATA_GENERATION.md).
3. Inspect the repository without relying on obsolete zone-level code or outputs.
4. Implement Phase 3 exactly to the acceptance checklist.
5. Run the generator and tests; then request validation before proceeding to Phase 4.
