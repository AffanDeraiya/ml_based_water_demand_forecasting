# Phase 3 Implementation Brief — Synthetic DNH-Total Monthly Data (Version 2)

## 1. Mission

Implement a deterministic, validated synthetic-data generator for **combined Dadra and Nagar Haveli (DNH) monthly residential water demand**. This brief is self-contained: a new agent should be able to implement Phase 3 without relying on older zone-level work.

The generator must create a plausible 15-year (180 month) dataset at a single spatial level. It exists to demonstrate and test the subsequent forecasting pipeline until real history is available. It is not a claim that these are observed DNH measurements.

## 2. Fixed decisions and boundaries

- Geography: one series only; every row has `area_id = "DNH_total"`.
- Time grain: monthly; `date` is the first calendar day of each month.
- Default history: `2010-01-01` through `2024-12-01`, inclusive (180 rows). Make dates configurable.
- Target: `residential_water_demand_m3` in m³/month.
- Scope: residential demand only. Do not create villages, zones, wards, `pateload`, hour-level observations, agriculture features, groundwater quality, consumption, or area-wide-usage columns.
- Exclude `net_annual_groundwater_availability_mcm` and `annual_groundwater_draft_mcm`.
- Use only open-source Python libraries already appropriate for the project (standard library, NumPy, pandas; matplotlib/seaborn optional for diagnostics).
- All randomness must be controlled by a supplied seed.
- This Version 2 design replaces the earlier zone-level synthetic generator; do not reuse zone aggregation as its mechanism.

The authoritative schema is [DATA_CONTRACT.md](DATA_CONTRACT.md). If this brief conflicts with that contract, the contract wins.

## 3. Required repository deliverables

Create or replace these Version 2 artifacts (minor naming changes are acceptable only if documented):

```text
configs/synthetic_dnh_total_v2.json
src/data_generation/generate_synthetic_dnh_total.py
src/data_generation/synthetic_dnh_total_generator.py
src/validation/validate_dnh_total_dataset.py
tests/test_synthetic_dnh_total_generator.py
data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv
data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json
```

The command-line entry point must load the JSON configuration, generate the dataset, validate it, write the CSV and metadata, and exit non-zero if validation fails. Do not require notebooks or manual steps to create the core dataset.

## 4. Exact output schema

Output the following columns in this exact order:

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

Use parseable ISO dates (`YYYY-MM-DD`), numeric values for all measured quantities, and integer population/household fields. The generated frame must contain exactly one row per configured month, no duplicates, no gaps, no nulls, and no infinite values.

## 5. Configuration requirements

The JSON config must contain at least:

```json
{
  "schema_version": "2.0",
  "start_date": "2010-01-01",
  "periods": 180,
  "seed": 42,
  "area_id": "DNH_total",
  "output_csv": "data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv",
  "metadata_output": "data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json"
}
```

It should also hold documented, adjustable parameters for baseline population/households, annual growth rates, climate amplitudes/noise, demand coefficients, state-variable response rates, and bounds. Keep parameters interpretable. Do not hard-code a random seed or output path inside the generator.

## 6. Data-generation design

Generate a common `month_of_year` seasonal signal (for example, smooth sinusoidal components), then layer distinct weather, demographic, demand, and system-state processes. Weather and system fields should be correlated where logically appropriate but never exact copies of one another. Use small, seeded random disturbances and enforce physical/logical bounds after generation.

### 6.1 Monsoon-led weather

Create a recognisable wet monsoon period (roughly June–September, with configurable peak/timing) and a dry period. The generated values do not need to match a particular observed year, but they must have plausible ranges and interannual variability.

- `rainfall_mm`: strongly monsoon-peaked, non-negative, with occasional dry/wet year effects and month-level noise. A gamma/lognormal-like positive process is preferable to simple unrestricted Gaussian noise.
- `humidity_max_pct`, `humidity_min_pct`: rise in monsoon months and fall in dry months. Always keep `0 <= minimum <= maximum <= 100`.
- `sunshine_hours`: generally lower during monsoon/cloudy months and higher in dry/sunny months; non-negative.
- `solar_radiation_mj_m2`: broadly related to sunshine, but include independent weather noise so it is not a direct transform of sunshine.
- `temp_max_c`, `temp_min_c`: use annual seasonality, realistic separation, and weather noise. `temp_min_c` must never exceed `temp_max_c`.
- `wind_speed_kmh`: seasonal but weakly/noisily related to monsoon conditions; non-negative.

Diagnostic expectation: rainfall and humidity visibly peak in monsoon months; sunshine and solar radiation generally decline then. The relationship should be visible in aggregates, not perfect row-by-row correlation.

### 6.2 Demographics

Generate `total_population`, `urban_population`, and `total_households` at monthly frequency.

- Start with configurable plausible base values.
- Apply gradual positive growth with modest year-to-year/monthly variation; avoid abrupt drops or step changes.
- Make urban population a non-decreasing or near-non-decreasing share of total population, always within total population.
- Grow households consistently with population, allowing household size to change only gradually.
- Round output population and household counts to integers only at final output.

Diagnostic expectation: plots show smooth long-run upward trends, not obvious deterministic straight lines or implausible shocks.

### 6.3 Residential water demand: target process

Create `residential_water_demand_m3` as a positive DNH-total monthly target. It must be driven by several components, including:

1. A demand baseline that scales with population and/or households.
2. A recurring seasonal component (for example, warmer/drier months and monthly usage behaviour).
3. Weather effects that are plausible but modest relative to base demand; rainfall may reduce some outdoor demand, while heat may increase demand.
4. Autoregressive persistence: current demand partly depends on the previous generated month's demand.
5. Idiosyncratic seeded noise and a small number of bounded shocks so models cannot recover an exact formula.

Do not compute demand as a direct copy of rainfall, a fully deterministic population multiplier, or from target-month variables that later would be unavailable at prediction time. The final relationship should support modelling with historical lags while retaining genuine forecast difficulty.

Use a positive lower bound. Preserve reasonable month-to-month variation—avoid either perfectly smooth demand or erratic negative/near-zero values. Document the selected formula/components and coefficients in metadata.

### 6.4 Supply and groundwater state variables

Generate these after weather and demand so they exhibit gradual state behaviour:

- `reservoir_level_m`: a bounded carry-over state. It rises with rainfall/recharge and falls with withdrawals/use, using a response rate so it does not jump instantly each month. Output a month-end level.
- `groundwater_level_m_bgl`: a bounded carry-over state where a larger number means deeper water below ground. It should tend to improve/shallow after rainfall recharge and worsen/deepen with demand/use, with lagged response and noise.
- `canal_discharge_cumecs`: a non-negative monthly mean flow with seasonal planning/supply behaviour, influenced noisily by season, reservoir availability, and demand. It must not be a simple scalar multiple of demand or rainfall.

All levels must remain within explicit configurable physical bounds. Make the feedback intentionally stable: response rates must prevent runaway reservoir depletion, groundwater deepening, or oscillation over 180 months.

## 7. Recommended generation order

1. Read config, validate required keys, initialise `numpy.random.Generator(seed)`, and create dates.
2. Build calendar columns internally (`month`, year index, monsoon indicator, smooth seasonal functions). Do not include internal helper columns in the final CSV unless explicitly added to a separate diagnostic output.
3. Generate weather with shared interannual shocks plus field-specific noise.
4. Generate demographic trajectories.
5. Iteratively generate demand and carry-over state variables month by month, so demand lag and storage/recharge dynamics are real.
6. Assemble only the canonical 17 columns in exact order.
7. Apply rounding only for output presentation; retain sufficient numeric precision for continuous fields.
8. Run the reusable validator. Write outputs only if validation passes, or clearly mark/remove partial outputs if it fails.
9. Write metadata capturing reproducibility and diagnostic summaries.

## 8. Validation requirements

Implement a reusable validator callable from both the CLI and tests. It must check:

### Structural checks

- exact schema and column order;
- expected row count;
- one `DNH_total` area ID only;
- dates are monthly, unique, ascending, and consecutive;
- no missing, infinite, or non-numeric measurement values.

### Domain checks

- non-negative: rainfall, wind, solar radiation, sunshine, canal discharge, and demand;
- valid humidity order/range;
- `temp_min_c <= temp_max_c`;
- population/households positive and `urban_population <= total_population`;
- configured bounds respected for reservoir and groundwater levels;
- no forbidden zone, village, agriculture, quality, availability, draft, consumption, or hour columns.

### Behavioural checks

- monsoon mean rainfall exceeds dry-season mean rainfall by a meaningful configurable ratio;
- monsoon humidity is higher and sunshine is lower than dry-season aggregates;
- population and households have a positive overall trend without large discontinuities;
- demand has non-zero variance, positive values, and meaningful but not extreme month-to-month change;
- reservoir and groundwater have non-zero variance and are not constant/direct duplicates of rainfall;
- canal discharge has non-zero seasonal variation;
- all required relationships are checked using tolerant aggregate thresholds, not brittle exact correlations.

The validator should return a structured result or raise a clear exception that identifies every failed rule.

## 9. Tests and reproducibility

Write automated tests covering at least:

1. Default configuration produces exactly 180 rows and the exact schema.
2. The same config and seed produce byte-equivalent values/dataframe output.
3. A changed seed changes stochastic output while preserving structural/domain validity.
4. Invalid configurations (bad date, non-positive periods, missing seed/area ID, invalid bounds) fail clearly.
5. Validator rejects missing columns, duplicate/gapped dates, non-DNH area IDs, invalid humidity/temperature/population relationships, and negative demand.
6. Default generated output passes all behavioural checks.

Metadata JSON must include: schema/generator version, timestamp, configured date range/periods, seed, `area_id`, full config or its hash, exact column list, row count, file paths, validation result, summary statistics, and a clear statement that the data is synthetic.

## 10. Acceptance checklist

Phase 3 is complete only when all items hold:

- A documented command produces the CSV and metadata from a clean checkout/environment.
- The output conforms exactly to the Version 2 data contract.
- Re-running with the same config/seed is reproducible.
- Tests and generator validation pass.
- Time-series plots or numerical diagnostics demonstrate monsoon patterns, smooth demographics, non-trivial demand variation, and gradual water-system state response.
- No multi-zone/Village/pateload artifacts remain in the Version 2 dataset or generation path.
- A reader can distinguish synthetic data from real DNH observations from the metadata and documentation.

## 11. Handoff to Phase 4
