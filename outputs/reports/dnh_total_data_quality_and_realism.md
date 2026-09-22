# DNH Total Monthly Dataset: Data Quality and Realism Review

## Dataset reviewed

- Source: validated DNH_total monthly dataset
- Coverage: January 2010 to December 2024
- Observations: 180 monthly rows
- Geography: one combined DNH_total series
- Data type: synthetic data for forecasting-pipeline development, not observed demand

## Data-quality findings

The canonical loader accepted the dataset and confirmed:

- all required columns are present
- there are no null values
- there are no non-finite numeric values
- dates are unique, consecutive, and monthly
- all records use `DNH_total`
- rainfall, wind, solar radiation, sunshine, canal discharge, and demand satisfy their non-negative rules
- humidity and temperature ordering rules are satisfied
- population, urban population, and households satisfy their domain constraints

The dataset is structurally clean and reproducible. Its configuration, seed, metadata, and checksum are recorded in the generated metadata file.

## Plausibility findings

### Weather and seasonality

The weather series shows the intended seasonal behaviour:

- monsoon rainfall is approximately 3.68 times dry-season rainfall
- monsoon maximum humidity is approximately 1.19 times dry-season humidity
- monsoon sunshine is approximately 65% of dry-season sunshine

These patterns are plausible for a monsoon-influenced monthly series and are useful for testing seasonal forecasting features.

### Demand and demographics

The demand series has meaningful variability:

- mean demand: approximately 3.66 million m³/month
- coefficient of variation: approximately 19.1%
- lag-1 demand autocorrelation: approximately 0.77
- overall demand growth: approximately 38.9%

Population and household changes are gradual:

- population growth over the period: approximately 44.9%
- household growth over the period: approximately 38.8%
- largest monthly population change: approximately 0.91%
- largest monthly household change: approximately 1.31%

These properties are suitable for initial time-series feature and model experiments.

## System-state review

The state-update dynamics were corrected to use normalized rainfall and demand drivers with configurable coefficients and gradual mean reversion. The regenerated data now shows useful variation without boundary saturation:

- `reservoir_level_m` ranges from 28.54 m to 40.38 m
- `groundwater_level_m_bgl` ranges from 8.22 m to 11.22 m
- reservoir boundary occupancy: 0% of months
- groundwater boundary occupancy: 0% of months
- reservoir unique values: 151 of 180 months
- groundwater unique values: 120 of 180 months

The state variables now remain inside their configured operating bands while responding gradually to rainfall and demand. The former clipping artefact is no longer present in the regenerated dataset.

Demand also contains large month-to-month movements in a small number of periods. The 95th percentile of absolute monthly demand change is approximately 30.9%, and the maximum is approximately 56.0%. These movements may be useful for stress testing, but they should be reviewed after the state dynamics are corrected.

## Decision

The regenerated dataset is structurally valid and suitable for feature engineering and model-development experiments. The weather, demand, demographic, reservoir, and groundwater series all retain useful variation within their configured ranges.

For modelling:

- weather, calendar, demographic, lagged-demand, and system-state features may be explored
- canal discharge remains subject to the same synthetic-data limitation as every other generated predictor
- no claim should be made that the synthetic relationships represent observed DNH behaviour

## Review status

`QUALITY_REVIEW_COMPLETE_AFTER_STATE_DYNAMICS_CORRECTION`
