# Research Design Document — DNH Residential Water-Demand Forecasting

## 1. Study purpose

Develop a reproducible machine-learning pipeline to forecast **combined monthly residential water demand for Dadra and Nagar Haveli (DNH)**. The project uses realistic synthetic data first, then permits replacement by confirmed real history without redesigning the pipeline.

## 2. Confirmed design decisions

- Spatial unit: one DNH-total series (`area_id = DNH_total`), not zones or villages.
- Target: `residential_water_demand_m3`, measured in cubic metres per month.
- Forecast horizons: next-month forecasts and quarterly demand calculated as the sum of three sequential monthly forecasts.
- Main inputs: monthly weather, demographics, reservoir level, canal discharge, groundwater level, and lagged historical demand.
- Scope: residential demand only. Agricultural data and groundwater-quality variables are excluded.
- Data stage: the supplied workbook is a sample schema, so Phase 3 produces 15 years of reproducible monthly synthetic data.
- Technology: open-source Python tooling; no dashboard is required. Deliverables are code, datasets, validation, model-comparison outputs, plots, and a concise report if required.

The canonical fields and quality rules are defined in [DATA_CONTRACT.md](DATA_CONTRACT.md).

## 3. Research questions

1. Can weather, demographic, operational water-system, and lagged-demand data forecast DNH-total monthly residential water demand better than a seasonal-naive baseline?
2. How do baseline, Random Forest, ANN, and LSTM models compare under an identical chronological evaluation protocol?
3. Which predictors and seasonal patterns most influence forecasts?
4. Can K-Means reveal interpretable **monthly demand regimes** (for example, monsoon/high-demand, dry-season, or transition months) without being treated as geographic clustering?

## 4. Forecasting design

The modelling target is monthly demand for month *t*. Forecast features must be restricted to data available at the forecast origin. This typically includes lags of demand and operational variables, calendar features, and known/forecast weather inputs. It must never use observed demand or other unavailable values from the target month.

The default synthetic-data split will be chronological: 10 years training, 2 years validation, and 3 years test. It may be adjusted only with a recorded reason if a future real dataset has a different usable length. Feature transformations, tuning, and model selection use only training and validation periods; the test period remains untouched until final evaluation.

Quarterly output is generated recursively/sequence-wise by forecasting three monthly values and summing them. This maintains the same target definition and avoids pretending quarterly observations are available.

## 5. Synthetic-data realism requirements

The synthetic generator must create plausible relationships without making the target trivially recoverable:

- Monsoon-driven rainfall, humidity, sunshine, solar radiation, temperature, and wind patterns.
- Smooth growth in population, urban population, and households.
- Demand affected by population/households, climate/seasonality, prior demand, and random shocks.
- Reservoir and groundwater levels that respond gradually to rainfall and demand/use.
- Canal discharge that varies by season and supply-system conditions.
- All demand values in m³/month at the DNH-total level.

Exact relationships, files, tests, and acceptance criteria are in [PHASE_3_SYNTHETIC_DATA_GENERATION.md](PHASE_3_SYNTHETIC_DATA_GENERATION.md).

## 6. Models and evaluation (later phases)

At minimum, compare a seasonal-naive baseline with Random Forest, ANN, and LSTM. Report MAE, RMSE, sMAPE, and R² where meaningful. “Accuracy” should not be presented as a generic classification-style percentage for a continuous-demand forecast. Synthetic-data metrics demonstrate the pipeline only; claims about practical DNH forecasting performance require a sufficient real historical holdout period.

K-Means is retained because the research framing includes data mining, but it is a supplementary, leakage-safe exploratory analysis of monthly demand regimes. It must not be described as clustering DNH geographical zones.

## 7. Real-data replacement protocol

When actual historical data becomes available, retain the canonical schema and document all aggregation choices. Confirm that hourly or location-level demand is residential before summing it to a DNH-total monthly target. Reassess `m³` versus litres only as a unit conversion/presentation requirement. Do not make unsupported claims if real history is not long enough for the selected validation design.

## 8. Limitations and assumptions
