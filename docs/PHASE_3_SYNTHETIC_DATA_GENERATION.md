# Phase 3 Implementation Brief — Synthetic Data Generation

## 1. Purpose and scope

Implement the Version 1 synthetic-data generator for the DNH residential water-demand forecasting project.

The generator must produce realistic, reproducible **monthly zone-level residential water-demand data** that conforms exactly to `DATA_CONTRACT.md`. It validates the downstream pipeline design while real DNH data is unavailable. It must never be presented as real DNH observations or evidence of real-world forecast accuracy.

This task ends after synthetic-data generation, validation, metadata, and tests are complete. Do **not** implement preprocessing, feature engineering, K-Means, forecasting models, dashboards, or reporting in this phase.

## 2. Governing documents

Read these files before implementation and follow them if there is any conflict:

1. `RESEARCH_DESIGN_DOCUMENT.md` — research scope, assumptions, leakage rules, evaluation design.
2. `DATA_CONTRACT.md` — authoritative schema, field definitions, quality rules, aggregation rules, and file conventions.
3. `IMPLEMENTATION_PLAN.md` — project-wide sequencing and Version 1 assumptions.

Key fixed decisions:

- Forecasting scope: residential demand only.
- Canonical frequency: monthly.
- Canonical unit: cubic metres (`m3`) per month for water-volume fields.
- Spatial design: five **neutral synthetic** zones, not asserted to match real administrative boundaries.
- Zone IDs: `zone_01`, `zone_02`, `zone_03`, `zone_04`, `zone_05`.
- Default coverage: 15 complete calendar years, January 2010 through December 2024.
- Default random seed: `2026`.
- Canonical dataset size: 900 rows (15 years × 12 months × 5 zones).
- `DNH_total` is a separately derived monthly aggregate, not a canonical source-zone row.

The date range and seed must be configurable. The defaults above are Version 1 defaults, not hard-coded assumptions scattered through the code.

## 3. Required outputs

Create these output artifacts using the exact canonical fields and a versioned naming convention:

```text
data/synthetic/raw/synthetic_zone_monthly_v1.csv
data/synthetic/derived/synthetic_dnh_total_monthly_v1.csv
data/synthetic/metadata/synthetic_zone_monthly_v1_metadata.json
```

The agent may add a generation configuration file and automated tests under appropriate project folders. Do not overwrite user-provided data.

### 3.1 Canonical zone-level CSV

The canonical CSV must contain these columns, in this order:

```text
date
area_id
rainfall_mm
population
water_consumption_m3
area_water_usage_m3
residential_water_demand_m3
```

Rules:

- `date` uses `YYYY-MM-01` and represents the month.
- exactly five valid zone IDs appear;
- each zone has consecutive monthly records;
- (`date`, `area_id`) is unique;
- all required values are present and numeric;
- `population` is a positive integer;
- rainfall and all water-volume fields are non-negative;
- demand must be strictly positive;
- sort by `date`, then `area_id`.

The default canonical dataset is clean. Do not inject missing values or anomalies into it.

### 3.2 Derived DNH-total CSV

Create one row per month with `area_id = DNH_total` using the aggregation rules from `DATA_CONTRACT.md`:

- sum: `population`, `water_consumption_m3`, `area_water_usage_m3`, and `residential_water_demand_m3`;
- rainfall: population-weighted mean of zone rainfall.

This file has 180 rows and uses the same column order as the zone-level CSV.

### 3.3 Metadata JSON

The metadata file must include:

- dataset name and version;
- statement that the data is synthetic and not real DNH data;
- generator version;
- generation timestamp;
- random seed;
- start and end dates;
- zone IDs;
- row counts for zone and aggregate outputs;
- generator configuration or a reference to the configuration file;
- field units;
- a concise description of the synthetic relationships;
- output-file paths;
- validation result summary.

## 4. Recommended implementation structure

Use Python with open-source libraries. Prefer a small, modular implementation rather than one large notebook.

Suggested files:

```text
configs/synthetic_data_v1.json
src/data_generation/generate_synthetic_data.py
src/data_generation/synthetic_generator.py
src/validation/validate_dataset.py
tests/test_synthetic_generator.py
```

`numpy`, `pandas`, the Python standard library, and `pytest` are sufficient. Avoid unnecessary dependencies. A JSON configuration file is preferred so configuration can be read without adding a parser dependency.

The exact module names may differ, but the responsibilities must remain separate:

- configuration loading and validation;
- zone-profile definition;
- monthly data generation;
- DNH-total aggregation;
- dataset validation;
- file/metadata writing;
- automated tests.

## 5. Configuration requirements

Put all tunable generation assumptions in one versioned configuration file. The generator must expose or configure at least:

```text
seed
start_date
end_date
zone_ids
zone_population_shares
base_population_total
annual_population_growth_range
zone_growth_adjustments
annual_rainfall_range_mm
monthly_rainfall_seasonality
zone_rainfall_multipliers
rainfall_noise_scale
per_capita_demand_litres_per_day_range
zone_demand_multipliers
demand_seasonality
rainfall_demand_sensitivity
demand_autoregressive_strength
consumption_variation_range
area_usage_multiplier_range
noise_scale
rounding_policy
```

Configuration validation must confirm that:

- zone IDs are unique;
- population shares are positive and sum to one within a small tolerance;
- start/end dates form complete months;
- all generated water-volume values can remain non-negative;
- date range contains at least 15 years by default, or a clearly logged overridden period;
- values representing proportions/multipliers are within meaningful bounds.

## 6. Synthetic-data behaviour to model

The generated data must be correlated and temporally plausible. Do not generate independent random columns.

### 6.1 Zone profiles

Create five distinct synthetic zone profiles. Each profile should differ in some combination of:

- population share and growth rate;
- rainfall multiplier;
- baseline per-capita demand;
- demand seasonality;
- consumption/usage variation.

Do not use real place names or imply administrative mapping. The goal is enough structural diversity for a later, meaningful K-Means experiment.

### 6.2 Population

Generate a positive integer monthly population for each zone.

Expected behaviour:

- DNH-total baseline population should be plausibly close to the documented historical scale, roughly several hundred thousand people, while remaining explicitly synthetic;
- each zone begins with its configured population share;
- population follows a gradual positive long-term growth trend;
- small, controlled variation is acceptable, but there must be no unrealistic month-to-month jumps;
- zone growth rates should differ modestly.

Population is an estimated monthly series. It may be calculated from annual growth and interpolated monthly, but the chosen method must be documented in metadata/configuration.

### 6.3 Rainfall

Generate non-negative monthly rainfall in millimetres with strong monsoon seasonality.

Expected behaviour:

- June through September should form the dominant wet season;
- remaining months should generally be drier;
- annual totals should vary year to year, with most totals in a configurable plausible DNH-inspired range of roughly 2,000–3,500 mm;
- zones should share a regional climate signal but have modest spatial variation;
- noise must not remove the seasonal pattern or create negative rainfall.

The PPT contains approximate annual rainfall context, but do not attempt to recreate or claim exact observed DNH yearly values.

### 6.4 Residential demand — target

Generate `residential_water_demand_m3` as the primary target. It should be a plausible function of:

- zone population;
- base per-capita demand, expressed in litres/person/day then converted to monthly cubic metres;
- month/season effect;
- rainfall-related seasonal effect, with a modest configurable relationship;
- gradual long-term growth;
- prior-month demand persistence/autocorrelation;
- zone-specific behaviour;
- realistic random noise.

A recommended conceptual form is:

```text
demand(t) = population(t)
          × base_per_capita_demand
          × days_in_month
          × seasonal_factor(month)
          × rainfall_adjustment(rainfall(t))
          × zone_factor
          + persistence_from_demand(t-1)
          + bounded_noise
```

The implementation may use a mathematically safer equivalent. It must keep demand positive, avoid extreme values, and record the selected parameter values.

The raw dataset may generate contemporaneous demand from contemporaneous simulated drivers. This is acceptable for data simulation. Later forecasting code must obey the leakage rules and use only lagged/past versions of drivers unless a feature is known at the forecast origin.

### 6.5 Consumption and area-wide usage

Generate related but distinct fields:

- `water_consumption_m3`: a correlated historical operational measure. It must not be exactly equal to target demand. Model controlled supply, loss, metering, or usage variation around demand.
- `area_water_usage_m3`: a broader optional usage proxy. Model it as related to consumption but with an additional controlled system-level/non-residential component.

Both fields must remain non-negative. Their relationship to demand must be strong enough for realistic lagged predictive value but imperfect enough to prevent trivial target reconstruction.

## 7. Reproducibility rules

1. Use one local random-number generator seeded from the configuration.
2. Running the generator twice with the same configuration and seed must create identical CSV content, excluding metadata timestamp if one is included.
3. Do not use external APIs, web data, or current dates as model inputs.
4. Do not use non-deterministic iteration order.
5. Record every config value that affects output in metadata.

## 8. Required validation implementation

Implement reusable validation, not only ad-hoc assertions in the generator.

The validation function must check:

```text
required canonical columns and their order
valid monthly dates normalised to day 1
valid configured area IDs
unique (date, area_id) pairs
continuous monthly coverage per area
expected 900 zone rows and 180 aggregate rows under default config
positive integer population
non-negative rainfall/consumption/usage
strictly positive demand
no null, infinite, or non-numeric values in required fields
sort order
correct aggregate sums and population-weighted rainfall
plausible configured ranges and no unflagged extreme values
```

Validation should return a machine-readable summary that can be written to metadata. It should raise a clear error when a required rule fails.

## 9. Required tests

Add automated tests covering at least:

1. Default generation creates the expected canonical and aggregate schemas.
2. Default generation creates 900 zone rows and 180 aggregate rows.
3. Each zone has 180 consecutive monthly rows.
4. Default generation is deterministic for the configured seed.
5. Different seeds change the generated data.
6. Zone IDs, units, types, and non-negativity/positivity rules conform to the contract.
7. DNH-total monthly demand, population, consumption, and usage equal zone sums.
8. DNH-total rainfall equals the population-weighted zone rainfall.
9. The generator rejects invalid configuration, such as duplicate zones or population shares that do not sum to one.

The implementation should also include a lightweight descriptive check or test showing that the seasonal rainfall pattern exists—for example, mean monsoon rainfall is greater than mean non-monsoon rainfall.

## 10. Optional controlled data-quality scenario

Do not contaminate the default canonical dataset. If implemented, a separate, opt-in quality-test scenario may create controlled missing values or anomalies for future preprocessing tests.

Requirements:

- it uses a distinct output filename and metadata label;
- the default configuration leaves it disabled;
- injected records are documented by field, count, date, and zone;
- it is never used as the normal modelling dataset without explicit configuration.

This feature is optional for Phase 3 and must not delay the clean default generator.

## 11. Acceptance criteria

Phase 3 is complete when all of the following are true:

1. A clean default zone-level dataset and derived DNH-total dataset are generated in the defined paths.
2. Both datasets conform to `DATA_CONTRACT.md`.
3. Metadata and configuration are stored with the outputs.
4. The data has reproducible, realistic temporal and cross-zone structure: seasonality, rainfall variation, gradual population growth, demand trends, persistence, and non-identical related usage variables.
5. Automated validation and the required tests pass.
6. The implementation makes no claim that synthetic results are real DNH results.
7. No code for later project phases is introduced unless it is a minimal reusable validation utility required by this phase.

## 12. Handoff requirements for the implementing agent

When reporting completion, provide:

- files created/changed;
- the command used to generate the default data;
- the command used to run tests;
- test/validation results;
- dataset row counts and date coverage;
- a concise explanation of the generated relationships;
- any deviations from this brief and why.

Do not modify the governing research-design or data-contract decisions without reporting a specific conflict or asking for approval.
