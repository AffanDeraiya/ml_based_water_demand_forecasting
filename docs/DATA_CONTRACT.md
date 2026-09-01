# Data Contract — DNH Monthly Residential Water-Demand Forecasting (Version 2)

## Status and purpose

This is the authoritative data contract for the project. It supersedes the earlier multi-zone synthetic-data design: the confirmed real-data spatial level is one combined monthly series for all Dadra and Nagar Haveli (DNH). The Phase 3 generator and all later modelling must use this contract.

The working target unit is cubic metres per month (m³/month), because the supplied water-demand sample uses `Demand_m3`. A future presentation-level conversion to litres is reversible (`1 m³ = 1,000 litres`) and does not require retraining.

## Dataset grain

- One row per calendar month.
- One geographic entity: `DNH_total`.
- Synthetic Version 2 default history: 180 consecutive months (15 years).
- The real-data pipeline will aggregate any finer-grained source data to this monthly DNH-total grain before modelling.

## Canonical modelling schema

| Field | Type / unit | Role | Rules |
|---|---|---|---|
| `date` | date, first day of month | index | Unique, strictly increasing, no missing months. |
| `area_id` | string | identifier | Always `DNH_total` in Version 2. Do not encode it as a predictive feature. |
| `rainfall_mm` | float, mm/month | predictor | Monthly total; monsoon-led seasonality. |
| `temp_max_c` | float, °C | predictor | Monthly mean of daily maximum temperatures where real data is daily. |
| `temp_min_c` | float, °C | predictor | Monthly mean of daily minimum temperatures where real data is daily. |
| `humidity_max_pct` | float, % | predictor | Monthly mean; range 0–100. |
| `humidity_min_pct` | float, % | predictor | Monthly mean; range 0–100 and not above maximum humidity. |
| `wind_speed_kmh` | float, km/h | predictor | Monthly mean; non-negative. |
| `solar_radiation_mj_m2` | float, MJ/m² | predictor | Monthly aggregation must be documented from the source semantics. |
| `sunshine_hours` | float, hours/month | predictor | Monthly total; non-negative. |
| `total_population` | integer | predictor | Monthly value; gradual growth only. |
| `urban_population` | integer | predictor | Monthly value; `0 <= urban_population <= total_population`. |
| `total_households` | integer | predictor | Monthly value; gradual growth only. |
| `reservoir_level_m` | float, m | predictor / system state | Month-end level; responds gradually to rainfall and use. |
| `canal_discharge_cumecs` | float, m³/s (cumecs) | predictor / system state | Monthly mean flow; non-negative. Never sum flow-rate observations. |
| `groundwater_level_m_bgl` | float, metres below ground level | predictor / system state | Monthly mean or month-end, chosen and recorded consistently; lower is shallower. |
| `residential_water_demand_m3` | float, m³/month | target | DNH-total residential demand for the month. |

## Deliberate exclusions

- No demand zones, villages, wards, or `pateload` identifiers in the canonical model data.
- No hour column: it belongs to raw operational data and must be aggregated before use.
- No agricultural variables: the approved scope is residential demand forecasting.
- No groundwater-quality fields (`pH`, `TDS`).
- No `net_annual_groundwater_availability_mcm` or `annual_groundwater_draft_mcm`.
- No consumption or area-wide usage fields until a consistent, confirmed monthly definition is supplied.

## Mapping the supplied workbook to the contract

The workbook is a schema/example workbook, not a 10–15 year modelling dataset. Its weather, demand, groundwater, and demographic columns informed the canonical fields above. Agricultural fields remain out of scope. When real history arrives, apply these rules:

- Aggregate demand across all locations to monthly DNH total only after confirming it represents residential demand.
- Sum daily/sub-daily rainfall to month; average temperature, humidity, and wind speed.
- Sum sunshine hours. Aggregate solar radiation only according to the source's definition (for example, daily energy totals are summed; instantaneous measurements are averaged).
- Use reservoir month-end level and monthly mean canal discharge.
- Use documented monthly-mean or month-end groundwater level consistently.
- Forward-fill annually reported demographic measures within the year, or interpolate them only under a documented policy.

## Time, leakage, and forecasting rules

- Predict month *t* using information that would have been known by the forecast issue date.
- Lag prior demand and any operational measurements when future-month values would not actually be known.
- Fit scalers, imputers, feature selection, and models on training data only.
- Use chronological train/validation/test partitions; never random-shuffle a time series.
- Quarterly forecasts in Version 1 are the sum of three sequential monthly forecasts, not a separate quarterly model.

## Quality gates

Every generated or ingested dataset must pass these checks before modelling:

1. Exactly one row for each consecutive month and a unique `date`.
2. `area_id` is always `DNH_total`.
3. All required columns exist; no nulls or non-finite values.
4. Demand, rainfall, wind, sunshine, solar radiation, and canal discharge are non-negative.
5. Humidity is in [0, 100], with minimum humidity no greater than maximum humidity.
6. `temp_min_c <= temp_max_c`; `urban_population <= total_population`.
7. Population and households move gradually rather than jumping implausibly.
8. Reservoir, canal, groundwater, and demand exhibit plausible seasonal and temporal variation; they must not be deterministic copies of rainfall.

## Locations and versioning
