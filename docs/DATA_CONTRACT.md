# Data Contract — DNH Residential Water-Demand Forecasting (Version 1)

## 1. Purpose

This data contract defines the canonical monthly dataset used by the project. It is the interface between data generation or real-data ingestion and every downstream component: validation, preprocessing, feature engineering, clustering, modelling, evaluation, and reporting.

The contract applies first to synthetic data and later to real data. Real data must be mapped into this structure rather than requiring the modelling pipeline to be redesigned.

## 2. Contract status and governing design

This is the Phase 2 Version 1 contract. It implements the scope and working assumptions in `RESEARCH_DESIGN_DOCUMENT.md`.

The following are intentional, revisable assumptions:

- The target is monthly residential water demand in cubic metres.
- Each input row represents one synthetic demand zone in one calendar month.
- Five neutral synthetic zones are used: `zone_01` through `zone_05`.
- `DNH_total` is a derived aggregation, not a separately generated zone.
- Water consumption is distinct from residential demand and can be used only through information known at the forecast origin.

If real data has a different spatial level, unit, or definition, change the ingestion/mapping layer and update this contract. Do not silently reinterpret fields in the model pipeline.

## 3. Canonical dataset grain

One row represents:

```text
one calendar month × one area_id
```

For Version 1, the valid area IDs are:

```text
zone_01
zone_02
zone_03
zone_04
zone_05
```

The synthetic dataset contains 15 years × 12 months × 5 zones = **900 canonical rows**.

`DNH_total` is calculated for each month by summing the zone-level demand, consumption, and area-wide usage. Rainfall and population are aggregated or summarised only through a documented method. It must not be mixed into the canonical zone-level training data unless a separate aggregate experiment is explicitly configured.

## 4. Canonical schema and data dictionary

| Field | Required | Type | Unit / format | Definition | Version 1 rules |
| --- | --- | --- | --- | --- | --- |
| `date` | Yes | date | `YYYY-MM-01` | First day of the observation month | One monthly observation; no gaps per area unless explicitly handled |
| `area_id` | Yes | string | `zone_01`–`zone_05` | Neutral synthetic demand-zone identifier | Stable across all periods; no real administrative meaning is implied |
| `rainfall_mm` | Yes | float | millimetres/month | Monthly rainfall associated with the area | Non-negative; may be zero; source/generation method must be logged |
| `population` | Yes | integer | people | Estimated resident population in the area for the month | Positive; should not show implausible month-to-month jumps |
| `water_consumption_m3` | Yes | float | cubic metres/month | Measured or simulated total water consumption for the area | Non-negative; never use target-month/future values to forecast demand |
| `area_water_usage_m3` | No | float | cubic metres/month | Broader area-wide water usage, if distinct from consumption | Non-negative; may be unavailable in future real data |
| `residential_water_demand_m3` | Yes | float | cubic metres/month | Estimated/observed residential water demand; forecasting target | Strictly positive for Version 1; never derived from future features at prediction time |

## 5. Field semantics and leakage policy

`residential_water_demand_m3` is the target at time `t + 1` for the one-step-ahead forecast task. A model making a forecast at the end of month `t` may use historical values through month `t` only.

### Synthetic variable semantics

- `residential_water_demand_m3` is the forecasting target. The synthetic generator will make it depend on population, seasonality, lagged demand, and controlled variation.
- `water_consumption_m3` is a correlated but distinct historical operational measurement. In synthetic data it may reflect demand plus controlled system/loss/usage variation; it must not duplicate the target exactly.
- `area_water_usage_m3` is an optional broader usage proxy. In synthetic data it may include system-level or non-residential variation and remains optional for future real-data ingestion.

These semantics are Version 1 assumptions. If real data defines consumption as the only residential-demand measure, update target semantics and do not use it as a duplicate same-period feature.

### Allowed inputs for a forecast of month `t + 1`

- demand, consumption, rainfall, and usage values through month `t`;
- lagged and rolling features calculated from values through month `t`;
- calendar features known in advance, such as month and quarter;
- a population projection/estimate for `t + 1` only if that value is genuinely available at forecast time and the experiment documents it.

### Prohibited inputs for a forecast of month `t + 1`

- `residential_water_demand_m3` at `t + 1`;
- actual `water_consumption_m3` or `area_water_usage_m3` at `t + 1`;
- rainfall observed after the forecast origin, unless the experiment explicitly uses a separately documented weather forecast feature;
- rolling/scaled/imputed values fitted using validation or test data.

## 6. Required quality rules

The data-validation component must enforce or report the following rules before preprocessing.

### Structural rules

1. All required fields exist exactly once.
2. `date` parses as a valid date and is normalised to the first day of a month.
3. `area_id` is non-empty and belongs to the configured area set.
4. The pair (`date`, `area_id`) is unique.
5. Each area has consecutive monthly dates across its expected coverage period.
6. Rows are sorted chronologically after validation.

### Type and range rules

1. `rainfall_mm`, `water_consumption_m3`, and `area_water_usage_m3` are numeric and non-negative where present.
2. `population` is an integer greater than zero.
3. `residential_water_demand_m3` is numeric and greater than zero in Version 1.
4. No value may be infinite or non-numeric.
5. Values beyond configured plausible ranges are flagged for review, not silently altered.

### Consistency rules

1. The target and consumption fields must not be treated as interchangeable.
2. Area IDs must remain stable across time.
3. Population should follow a plausible trend; exceptional jumps are flagged.
4. Any missing optional `area_water_usage_m3` values must remain explicitly missing until the preprocessing policy handles them.
5. Synthetic generator output must record its seed and parameter configuration.
6. Generator configuration must set documented plausible ranges for rainfall, population, demand, growth, zone variation, noise, and controlled anomalies. Validation uses these configured ranges rather than undocumented constants.

## 7. Missing-data policy

The canonical synthetic dataset should initially contain no uncontrolled missing required values. Controlled missingness may be generated only in a dedicated data-quality test scenario and must be labelled as such.

For eventual real data:

- Missing target values in the final test period must not be imputed for scoring.
- Missing input values may be imputed only using information available before the affected forecast origin.
- Missing optional `area_water_usage_m3` must not prevent a baseline pipeline run; models can use a feature configuration that excludes it.
- Every imputation strategy, affected field, date range, and row count must be logged.

## 8. Aggregation contract for DNH-total reporting

The aggregated DNH-total table is derived separately from canonical zone data.

| Field | Aggregation method |
| --- | --- |
| `residential_water_demand_m3` | Sum across zones |
| `water_consumption_m3` | Sum across zones |
| `area_water_usage_m3` | Sum across zones when all required zone values are present; otherwise record missing and report coverage |
| `population` | Sum across zones |
| `rainfall_mm` | Population-weighted mean across zones; use a simple mean only if population weighting is unavailable and record that fallback |
| `area_id` | `DNH_total` |

An aggregate-only forecasting experiment must use this derived table as its own input dataset and must not mix `DNH_total` rows with zone rows.

## 9. Synthetic-to-real mapping guide

When real data is obtained, create an ingestion mapping file that records the source column, source unit, conversion, target canonical field, temporal frequency, spatial level, and data-owner/source reference.

### Required mapping outcomes

| Real-data condition | Required treatment |
| --- | --- |
| Dates are daily/weekly | Aggregate to calendar month using a documented method before canonical validation |
| Water volume is litres | Convert to cubic metres: divide by 1,000 |
| Demand is reported in million litres | Convert to cubic metres: multiply by 1,000 |
| Real data is DNH-wide only | Use `area_id = DNH_total`; run aggregate pipeline and use K-Means only for demand-regime analysis |
| Real data is ward/village/zone level | Map stable identifiers to `area_id`; update the configured area list; retain zone-level clustering design |
| `area_water_usage_m3` is unavailable | Preserve as missing/omit it from the selected model feature configuration |
| Consumption equals the only available residential measure | Define it as the target only after updating target semantics; do not retain it as a duplicate same-period feature |

## 10. Dataset versions and file conventions

Use these intended locations and names when implementation begins:

```text
data/synthetic/raw/synthetic_zone_monthly_v1.csv
data/synthetic/derived/synthetic_dnh_total_monthly_v1.csv
data/real/raw/<source-specific-file>
data/real/canonical/real_zone_monthly_v1.csv
data/processed/<dataset-version>/...
```

Every processed dataset and experiment must record:

- source dataset version and checksum where applicable;
- seed for synthetic data;
- schema/data-contract version;
- selected feature set;
- split dates;
- preprocessing parameters;
- model and hyperparameter configuration.

## 11. Phase 2 acceptance criteria

Phase 2 is complete when:

1. The canonical schema and field meanings are accepted.
2. The five neutral synthetic zones are accepted as a Version 1 assumption.
3. Validation, missing-data, aggregation, and leakage rules are explicit.
4. The mapping process for real DNH data is documented.
5. Phase 3 can generate data that conforms to this contract without unresolved design ambiguity.
