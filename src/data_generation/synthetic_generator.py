"""Synthetic data generator implementation (skeleton).

This file will be extended to implement the full Phase 3 generator logic.
"""
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd


class SyntheticGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config.get("seed", 2026))
        np.random.seed(self.seed)

        self.start_date = pd.to_datetime(config.get("start_date"))
        self.end_date = pd.to_datetime(config.get("end_date"))
        self.zone_ids = config.get("zone_ids", [])

        self.out_raw = Path("data/synthetic/raw/synthetic_zone_monthly_v1.csv")
        self.out_agg = Path("data/synthetic/derived/synthetic_dnh_total_monthly_v1.csv")
        self.out_meta = Path("data/synthetic/metadata/synthetic_zone_monthly_v1_metadata.json")

    def generate_all(self):
        # placeholder generating an empty dataframe to be replaced by real logic
        dates = pd.date_range(self.start_date, self.end_date, freq="MS")
        rows = []
        for d in dates:
            for z in self.zone_ids:
                rows.append({
                    "date": d.strftime("%Y-%m-01"),
                    "area_id": z,
                    "rainfall_mm": 0.0,
                    "population": 1000,
                    "water_consumption_m3": 0.0,
                    "area_water_usage_m3": 0.0,
                    "residential_water_demand_m3": 1.0,
                })
        df = pd.DataFrame(rows)
        df.to_csv(self.out_raw, index=False)

        # write an empty aggregate and metadata for now
        agg = df.groupby("date").agg({
            "population": "sum",
            "rainfall_mm": "mean",
            "water_consumption_m3": "sum",
            "area_water_usage_m3": "sum",
            "residential_water_demand_m3": "sum",
        }).reset_index()
        agg["area_id"] = "DNH_total"
        agg = agg[["date", "area_id", "rainfall_mm", "population", "water_consumption_m3", "area_water_usage_m3", "residential_water_demand_m3"]]
        agg.to_csv(self.out_agg, index=False)

        meta = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "seed": self.seed,
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "zone_ids": self.zone_ids,
            "rows_zone": len(df),
            "rows_agg": len(agg),
        }
        self.out_meta.write_text(json.dumps(meta, indent=2))

        print(f"Wrote {self.out_raw} ({len(df)} rows) and {self.out_agg} ({len(agg)} rows)")
