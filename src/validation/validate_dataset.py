"""Dataset validation utilities (skeleton).

Validation functions will be implemented to enforce the DATA_CONTRACT rules.
"""
import pandas as pd


def validate_zone_dataframe(df: pd.DataFrame, zone_ids: list):
    errors = []
    required_cols = [
        "date",
        "area_id",
        "rainfall_mm",
        "population",
        "water_consumption_m3",
        "area_water_usage_m3",
        "residential_water_demand_m3",
    ]
    for c in required_cols:
        if c not in df.columns:
            errors.append(f"Missing column: {c}")
    # more validations to be added
    return errors
