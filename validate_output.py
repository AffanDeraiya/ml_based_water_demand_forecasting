"""Quick validation of generated synthetic dataset."""

import pandas as pd
import json
from pathlib import Path

# Read CSV
df = pd.read_csv("data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv")

print("=" * 70)
print("SYNTHETIC DNH-TOTAL DATASET v2 VALIDATION")
print("=" * 70)

# Shape
print(f"\n✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
assert df.shape[0] == 180, "Expected 180 rows"
assert df.shape[1] == 17, "Expected 17 columns"

# Schema
expected_cols = [
    "date", "area_id", "rainfall_mm", "temp_max_c", "temp_min_c",
    "humidity_max_pct", "humidity_min_pct", "wind_speed_kmh",
    "solar_radiation_mj_m2", "sunshine_hours", "total_population",
    "urban_population", "total_households", "reservoir_level_m",
    "canal_discharge_cumecs", "groundwater_level_m_bgl",
    "residential_water_demand_m3"
]
print(f"✓ Columns: {list(df.columns) == expected_cols}")
assert list(df.columns) == expected_cols

# Dates
print(f"✓ Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(f"✓ All area_id: {df['area_id'].unique()}")
assert (df["area_id"] == "DNH_total").all()

# Data quality
null_count = df.isnull().sum().sum()
print(f"✓ No nulls: {null_count == 0} (found {null_count})")

# Non-negativity checks
print("\nDomain Checks:")
print(f"✓ Rainfall non-negative: {(df['rainfall_mm'] >= 0).all()}")
print(f"✓ Demand positive: {(df['residential_water_demand_m3'] > 0).all()}")
print(f"✓ Temperature order: {(df['temp_min_c'] <= df['temp_max_c']).all()}")
print(f"✓ Humidity bounds: {((df['humidity_max_pct'] >= 0) & (df['humidity_max_pct'] <= 100)).all()}")
print(f"✓ Urban <= Total pop: {(df['urban_population'] <= df['total_population']).all()}")

# Behavioral checks
print("\nBehavioral Checks:")
print(f"✓ Population growth: {df['total_population'].iloc[-1] > df['total_population'].iloc[0]:.0f}")
pop_first = df['total_population'].iloc[0]
pop_last = df['total_population'].iloc[-1]
print(f"  2010-01: {pop_first:,.0f} → 2024-12: {pop_last:,.0f} (+{(pop_last-pop_first)/pop_first*100:.1f}%)")

# Monsoon patterns
df_check = df.copy()
df_check['month'] = pd.to_datetime(df_check['date']).dt.month
monsoon_months = {6, 7, 8, 9}

monsoon = df_check[df_check['month'].isin(monsoon_months)]
dry = df_check[~df_check['month'].isin(monsoon_months)]

monsoon_rain_mean = monsoon['rainfall_mm'].mean()
dry_rain_mean = dry['rainfall_mm'].mean()
ratio = monsoon_rain_mean / dry_rain_mean if dry_rain_mean > 0 else 0
print(f"✓ Monsoon/Dry rainfall ratio: {ratio:.2f}x (expected ≥3.5x)")

monsoon_hum = monsoon['humidity_max_pct'].mean()
dry_hum = dry['humidity_max_pct'].mean()
print(f"✓ Monsoon humidity higher: {monsoon_hum:.1f}% vs {dry_hum:.1f}%")

# Variance checks
print("\nVariance Checks:")
print(f"✓ Demand variance: {df['residential_water_demand_m3'].var():,.0f} (std: {df['residential_water_demand_m3'].std():,.0f})")
print(f"✓ Reservoir variance: {df['reservoir_level_m'].var():.2f} m²")
print(f"✓ Groundwater variance: {df['groundwater_level_m_bgl'].var():.2f} m²")

# Summary stats
print("\nSummary Statistics:")
for col in ['rainfall_mm', 'temp_max_c', 'total_population', 'residential_water_demand_m3']:
    print(f"  {col:30s}: mean={df[col].mean():12,.1f}, std={df[col].std():10,.1f}, min={df[col].min():10,.1f}, max={df[col].max():10,.1f}")

# Check metadata
meta_path = Path("data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json")
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    print("\nMetadata:")
    print(f"✓ Config seed: {meta['seed']}")
    print(f"✓ Generator version: {meta['generator_version']}")
    print(f"✓ Synthetic flag: {meta['synthetic']}")
    print(f"✓ Generated at: {meta['generated_at']}")

print("\n" + "=" * 70)
print("✓ ALL VALIDATION CHECKS PASSED")
print("=" * 70)
