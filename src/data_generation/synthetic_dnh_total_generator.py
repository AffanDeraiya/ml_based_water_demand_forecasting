"""Phase 3 DNH-total synthetic water-demand generator (Version 2).

Generates a single 180-month DNH-total time series with weather, demographics,
and system-state variables. All randomness is seeded for reproducibility.

Generation order:
1. Calendar and seasonal helper functions
2. Weather with shared interannual shocks
3. Demographic trajectories
4. Iterative month-by-month demand and carry-over state variables
5. Assemble 17 canonical columns
"""

import json
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import numpy as np
import pandas as pd


class SyntheticDNHTotalGenerator:
    """Deterministic generator for Version 2 synthetic DNH-total monthly data."""

    def __init__(self, config: dict):
        """Initialize with a configuration dict."""
        self.config = config
        self._validate_config()

        # Brief-compatible top-level keys; keep nested config compatibility for older structures.
        self.schema_version = str(config.get("schema_version", config.get("metadata", {}).get("schema_version", "2.0")))
        self.seed = int(config.get("seed", config["randomization"]["seed"]))
        if "randomization" in config and "seed" in config["randomization"]:
            config["randomization"]["seed"] = self.seed
        config["seed"] = self.seed
        self.rng = np.random.default_rng(self.seed)  # numpy 1.17+ API

        # Dates and periods
        self.start_date = pd.to_datetime(config.get("start_date", config["dates"]["start_date"]))
        self.periods = int(config.get("periods", config["dates"]["periods"]))
        self.end_date = self.start_date + pd.DateOffset(months=self.periods - 1)
        self.dates = pd.date_range(start=self.start_date, periods=self.periods, freq="MS")

        # Geography and outputs
        self.area_id = config.get("area_id", config["geography"]["area_id"])
        self.csv_path = Path(config.get("output_csv", config["output"]["csv_path"]))
        self.meta_path = Path(config.get("metadata_output", config["output"]["metadata_path"]))

        # Demographics baseline
        self.baseline_population = float(config["demographics"]["baseline_total_population"])
        self.baseline_urban = float(config["demographics"]["baseline_urban_population"])
        self.baseline_households = float(config["demographics"]["baseline_total_households"])
        self.population_growth_rate = float(config["demographics"]["annual_population_growth_rate"])

        # Weather config
        weather = config["weather"]
        self.monsoon_months = set(weather["monsoon_months"])
        self.rainfall_annual_baseline = float(weather["rainfall"]["annual_baseline_mm"])
        self.rainfall_annual_noise_std = float(weather["rainfall"]["annual_noise_std"])
        self.monsoon_peak_fraction = float(weather["rainfall"]["monsoon_peak_fraction"])
        self.temp_max_baseline = float(weather["temperature"]["temp_max_baseline_c"])
        self.temp_min_baseline = float(weather["temperature"]["temp_min_baseline_c"])
        self.temp_amplitude = float(weather["temperature"]["annual_cycle_amplitude"])
        self.temp_noise_std = float(weather["temperature"]["daily_variance_noise"])
        self.humidity_max_baseline = float(weather["humidity"]["humidity_max_baseline_pct"])
        self.humidity_min_baseline = float(weather["humidity"]["humidity_min_baseline_pct"])
        self.monsoon_humidity_max_shift = float(weather["humidity"]["monsoon_max_shift"])
        self.monsoon_humidity_min_shift = float(weather["humidity"]["monsoon_min_shift"])

        realism = config.get("realism", {})
        self.extreme_year_probability = float(realism.get("extreme_year_probability", 0.18))
        self.drought_multiplier_min = float(realism.get("drought_multiplier_range", [0.65, 0.88])[0])
        self.drought_multiplier_max = float(realism.get("drought_multiplier_range", [0.65, 0.88])[1])
        self.flood_multiplier_min = float(realism.get("flood_multiplier_range", [1.3, 1.75])[0])
        self.flood_multiplier_max = float(realism.get("flood_multiplier_range", [1.3, 1.75])[1])
        self.extreme_month_probability = float(realism.get("extreme_month_probability", 0.04))
        self.demand_shock_probability = float(realism.get("demand_shock_probability", 0.035))
        self.demand_shock_min = float(realism.get("demand_shock_range", [0.82, 1.25])[0])
        self.demand_shock_max = float(realism.get("demand_shock_range", [0.82, 1.25])[1])
        self.state_noise_std = float(realism.get("state_noise_std", 0.03))
        self.sunshine_annual_baseline = float(weather["sunshine"]["sunshine_hours_annual_baseline"])
        self.monsoon_sunshine_reduction = float(weather["sunshine"]["monsoon_reduction_fraction"])
        self.solar_annual_baseline = float(weather["solar_radiation"]["solar_radiation_annual_baseline_mj_m2"])
        self.monsoon_solar_reduction = float(weather["solar_radiation"]["monsoon_reduction_fraction"])
        self.solar_noise_std = float(weather["solar_radiation"]["independent_noise_std"])
        self.wind_baseline = float(weather["wind"]["wind_speed_baseline_kmh"])
        self.monsoon_wind_increase = float(weather["wind"]["monsoon_increase_fraction"])
        self.wind_noise_std = float(weather["wind"]["noise_std"])

        # Demand config
        demand = config["demand"]
        self.baseline_lpd = float(demand["baseline_demand_lpd"])
        self.monsoon_demand_mult = float(demand["seasonality"]["monsoon_multiplier"])
        self.dry_demand_mult = float(demand["seasonality"]["dry_multiplier"])
        self.rainfall_elasticity = float(demand["rainfall_elasticity"])
        self.ar_strength = float(demand["autoregressive_strength"])
        self.demand_noise_std = float(demand["noise_std"])

        # System state config
        sys_state = config["system_state"]
        self.reservoir_baseline = float(sys_state["reservoir"]["baseline_level_m"])
        self.reservoir_max = float(sys_state["reservoir"]["max_level_m"])
        self.reservoir_min = float(sys_state["reservoir"]["min_level_m"])
        self.reservoir_recharge_rate = float(sys_state["reservoir"]["recharge_response_rate"])
        self.reservoir_discharge_rate = float(sys_state["reservoir"]["discharge_response_rate"])
        self.gw_baseline = float(sys_state["groundwater"]["baseline_level_m_bgl"])
        self.gw_max = float(sys_state["groundwater"]["max_level_m_bgl"])
        self.gw_min = float(sys_state["groundwater"]["min_level_m_bgl"])
        self.gw_recharge_rate = float(sys_state["groundwater"]["recharge_response_rate"])
        self.gw_discharge_rate = float(sys_state["groundwater"]["discharge_response_rate"])
        self.canal_baseline = float(sys_state["canal_discharge"]["baseline_discharge_cumecs"])
        self.canal_max = float(sys_state["canal_discharge"]["max_discharge_cumecs"])
        self.canal_min = float(sys_state["canal_discharge"]["min_discharge_cumecs"])
        self.canal_dry_mult = float(sys_state["canal_discharge"]["seasonal_variation"]["dry_multiplier"])
        self.canal_monsoon_mult = float(sys_state["canal_discharge"]["seasonal_variation"]["monsoon_multiplier"])
        self.canal_demand_elasticity = float(sys_state["canal_discharge"]["demand_elasticity"])

        # Compute config checksum for provenance
        config_json = json.dumps(self.config, sort_keys=True)
        self.config_checksum = hashlib.sha256(config_json.encode()).hexdigest()

    def _validate_config(self):
        """Basic config validation aligned with Phase 3 brief and backward-compatible with nested config."""
        required_sections = ["randomization", "dates", "geography", "output", "demographics", "weather", "demand", "system_state"]
        required_top_level = ["schema_version", "start_date", "periods", "seed", "area_id", "output_csv", "metadata_output"]

        for key in required_sections:
            if key not in self.config:
                raise ValueError(f"Config missing required section: {key}")

        for key in required_top_level:
            if key not in self.config:
                # Accept older nested config structures for backward compatibility during transition.
                if key in ["start_date", "periods", "seed", "area_id", "output_csv", "metadata_output"]:
                    if key == "start_date" and "dates" in self.config and "start_date" in self.config["dates"]:
                        continue
                    if key == "periods" and "dates" in self.config and "periods" in self.config["dates"]:
                        continue
                    if key == "seed" and "randomization" in self.config and "seed" in self.config["randomization"]:
                        continue
                    if key == "area_id" and "geography" in self.config and "area_id" in self.config["geography"]:
                        continue
                    if key == "output_csv" and "output" in self.config and "csv_path" in self.config["output"]:
                        continue
                    if key == "metadata_output" and "output" in self.config and "metadata_path" in self.config["output"]:
                        continue
                raise ValueError(f"Config missing required brief key: {key}")

        try:
            pd.to_datetime(self.config.get("start_date", self.config["dates"]["start_date"]))
        except Exception as exc:
            raise ValueError(f"Invalid start_date in config: {exc}") from exc

        periods = self.config.get("periods", self.config["dates"]["periods"])
        if int(periods) <= 0:
            raise ValueError("Config periods must be a positive integer")

        seed = self.config.get("seed", self.config["randomization"]["seed"])
        try:
            int(seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("Config seed must be an integer-valued value") from exc

        area_id = self.config.get("area_id", self.config["geography"]["area_id"])
        if not str(area_id):
            raise ValueError("Config area_id must be non-empty")

        if self.config.get("schema_version") is not None and str(self.config.get("schema_version")) != "2.0":
            raise ValueError("Config schema_version must be '2.0'")

        monsoon_months = self.config["weather"].get("monsoon_months", [])
        if not monsoon_months or any(int(m) < 1 or int(m) > 12 for m in monsoon_months):
            raise ValueError("Config weather.monsoon_months must contain valid month numbers 1..12")

        reservoir_cfg = self.config["system_state"]["reservoir"]
        gw_cfg = self.config["system_state"]["groundwater"]
        if float(reservoir_cfg["min_level_m"]) >= float(reservoir_cfg["max_level_m"]):
            raise ValueError("Reservoir min_level_m must be less than max_level_m")
        if float(gw_cfg["min_level_m_bgl"]) >= float(gw_cfg["max_level_m_bgl"]):
            raise ValueError("Groundwater min_level_m_bgl must be less than max_level_m_bgl")

    def _is_monsoon(self, month: int) -> bool:
        """Check if month is in monsoon period."""
        return month in self.monsoon_months

    def _seasonal_factor(self, month: int) -> float:
        """Return a smooth seasonal factor (0-1 scale) for any month."""
        # Use cosine wave: peak at June (month 6), trough at December (month 12)
        # Shifted so monsoon months have higher values
        angle = 2 * np.pi * (month - 1) / 12
        factor = 0.5 + 0.5 * np.cos(angle - np.pi / 2)  # peaks at month 6
        return factor

    def generate_all(self) -> pd.DataFrame:
        """Generate the complete 180-month DNH-total dataset."""

        # Step 1: Sample year-level shocks (shared across weather variables).
        # This preserves the 15-year monthly scope while introducing rare drought/flood years
        # and more realistic interannual variability that real water systems exhibit.
        years = sorted(set(d.year for d in self.dates))
        year_rainfall_shocks = {}
        for year in years:
            if self.rng.random() < self.extreme_year_probability:
                if self.rng.random() < 0.5:
                    shock = self.rng.uniform(self.drought_multiplier_min, self.drought_multiplier_max)
                else:
                    shock = self.rng.uniform(self.flood_multiplier_min, self.flood_multiplier_max)
            else:
                shock = self.rng.uniform(0.85, 1.15)
            year_rainfall_shocks[year] = float(np.clip(shock, 0.5, 2.0))

        # Use configured annual noise to create realistic year-to-year rainfall variability.
        # The raw annual_noise_std is held in mm and is converted to a proportional factor.
        self._annual_noise_scale = max(self.rainfall_annual_noise_std / max(self.rainfall_annual_baseline, 1.0), 0.0)

        # Step 2: Generate weather time series
        weather_df = self._generate_weather(year_rainfall_shocks)

        # Step 3: Generate demographics
        demo_df = self._generate_demographics()

        # Step 4: Generate demand and system state iteratively
        state_df = self._generate_demand_and_state(weather_df, demo_df)

        # Step 5: Assemble final dataframe with 17 canonical columns in exact order
        df = pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in self.dates],
            "area_id": self.area_id,
            "rainfall_mm": weather_df["rainfall_mm"].values,
            "temp_max_c": weather_df["temp_max_c"].values,
            "temp_min_c": weather_df["temp_min_c"].values,
            "humidity_max_pct": weather_df["humidity_max_pct"].values,
            "humidity_min_pct": weather_df["humidity_min_pct"].values,
            "wind_speed_kmh": weather_df["wind_speed_kmh"].values,
            "solar_radiation_mj_m2": weather_df["solar_radiation_mj_m2"].values,
            "sunshine_hours": weather_df["sunshine_hours"].values,
            "total_population": demo_df["total_population"].values,
            "urban_population": demo_df["urban_population"].values,
            "total_households": demo_df["total_households"].values,
            "reservoir_level_m": state_df["reservoir_level_m"].values,
            "canal_discharge_cumecs": state_df["canal_discharge_cumecs"].values,
            "groundwater_level_m_bgl": state_df["groundwater_level_m_bgl"].values,
            "residential_water_demand_m3": state_df["residential_water_demand_m3"].values,
        })

        # Round output
        df["rainfall_mm"] = df["rainfall_mm"].round(2)
        df["temp_max_c"] = df["temp_max_c"].round(2)
        df["temp_min_c"] = df["temp_min_c"].round(2)
        df["humidity_max_pct"] = df["humidity_max_pct"].round(1)
        df["humidity_min_pct"] = df["humidity_min_pct"].round(1)
        df["wind_speed_kmh"] = df["wind_speed_kmh"].round(2)
        df["solar_radiation_mj_m2"] = df["solar_radiation_mj_m2"].round(2)
        df["sunshine_hours"] = df["sunshine_hours"].round(1)
        df["total_population"] = df["total_population"].astype(int)
        df["urban_population"] = df["urban_population"].astype(int)
        df["total_households"] = df["total_households"].astype(int)
        df["reservoir_level_m"] = df["reservoir_level_m"].round(2)
        df["canal_discharge_cumecs"] = df["canal_discharge_cumecs"].round(3)
        df["groundwater_level_m_bgl"] = df["groundwater_level_m_bgl"].round(2)
        df["residential_water_demand_m3"] = df["residential_water_demand_m3"].round(1)

        return df

    def _generate_weather(self, year_shocks: dict) -> pd.DataFrame:
        """Generate 180 months of weather variables."""
        records = []

        for i, date in enumerate(self.dates):
            year = date.year
            month = date.month
            is_monsoon = self._is_monsoon(month)
            seasonal = self._seasonal_factor(month)

            # Rainfall: lognormal-like positive with explicit annual-noise contribution.
            # Add a small shock probability for extreme months to mimic real monsoon volatility.
            annual_total = self.rainfall_annual_baseline * year_shocks[year]
            annual_total *= 1.0 + float(self.rng.normal(0, self._annual_noise_scale))
            annual_total = max(0.0, annual_total)
            if is_monsoon:
                monsoon_total = annual_total * self.monsoon_peak_fraction
                monsoon_months_count = len(self.monsoon_months)
                base_rain = monsoon_total / monsoon_months_count
            else:
                dry_total = annual_total * (1 - self.monsoon_peak_fraction)
                dry_months = 12 - len(self.monsoon_months)
                base_rain = dry_total / dry_months
            rainfall = max(0.0, base_rain * (1.0 + float(self.rng.normal(0, 0.1))))

            if self.rng.random() < self.extreme_month_probability:
                if is_monsoon:
                    rainfall *= self.rng.uniform(1.25, 1.7)
                else:
                    rainfall *= self.rng.uniform(0.7, 0.9)

            # Temperature: annual cycle with noise
            angle = 2 * np.pi * (month - 1) / 12
            temp_offset = self.temp_amplitude * 0.5 * np.cos(angle)  # peak in summer
            temp_max = self.temp_max_baseline + temp_offset + float(self.rng.normal(0, self.temp_noise_std))
            temp_min = self.temp_min_baseline + temp_offset + float(self.rng.normal(0, self.temp_noise_std))
            # Ensure min <= max
            if temp_min > temp_max:
                temp_min, temp_max = temp_max, temp_min

            # Humidity: monsoon-driven
            if is_monsoon:
                humidity_max = self.humidity_max_baseline + self.monsoon_humidity_max_shift + float(self.rng.normal(0, 2))
                humidity_min = self.humidity_min_baseline + self.monsoon_humidity_min_shift + float(self.rng.normal(0, 2))
            else:
                humidity_max = self.humidity_max_baseline + float(self.rng.normal(0, 2))
                humidity_min = self.humidity_min_baseline + float(self.rng.normal(0, 2))
            # Bounds
            humidity_max = np.clip(humidity_max, 0, 100)
            humidity_min = np.clip(humidity_min, 0, humidity_max)

            # Sunshine: monsoon reduction
            if is_monsoon:
                sunshine = self.sunshine_annual_baseline / 12 * (1 - self.monsoon_sunshine_reduction) + float(self.rng.normal(0, 10))
            else:
                sunshine = self.sunshine_annual_baseline / 12 + float(self.rng.normal(0, 10))
            sunshine = max(0.0, sunshine)

            # Solar radiation: related to sunshine but independent noise
            if is_monsoon:
                solar = self.solar_annual_baseline / 12 * (1 - self.monsoon_solar_reduction)
            else:
                solar = self.solar_annual_baseline / 12
            solar = solar * (1.0 + float(self.rng.normal(0, self.solar_noise_std)))
            solar = max(0.0, solar)

            # Wind speed: weakly tied to monsoon
            if is_monsoon:
                wind = self.wind_baseline * (1 + self.monsoon_wind_increase) + float(self.rng.normal(0, self.wind_noise_std))
            else:
                wind = self.wind_baseline + float(self.rng.normal(0, self.wind_noise_std))
            wind = max(0.0, wind)

            records.append({
                "date": date,
                "rainfall_mm": rainfall,
                "temp_max_c": temp_max,
                "temp_min_c": temp_min,
                "humidity_max_pct": humidity_max,
                "humidity_min_pct": humidity_min,
                "sunshine_hours": sunshine,
                "solar_radiation_mj_m2": solar,
                "wind_speed_kmh": wind,
            })

        return pd.DataFrame(records)

    def _generate_demographics(self) -> pd.DataFrame:
        """Generate smooth demographic trajectories."""
        records = []

        for i, date in enumerate(self.dates):
            # Monthly growth rate from annual
            monthly_growth = (1.0 + self.population_growth_rate) ** (1 / 12) - 1
            months_from_start = i

            # Total population with small monthly noise
            total_pop = self.baseline_population * ((1 + monthly_growth) ** months_from_start)
            total_pop = total_pop * (1.0 + float(self.rng.normal(0, 0.002)))  # 0.2% noise
            total_pop = max(1, total_pop)

            # Urban population: grows with total, urban share increases slowly
            urban_share_start = self.baseline_urban / self.baseline_population
            urban_share_end = min(0.50, urban_share_start + 0.15 * (months_from_start / self.periods))
            urban_pop = total_pop * urban_share_end
            urban_pop = urban_pop * (1.0 + float(self.rng.normal(0, 0.003)))
            urban_pop = np.clip(urban_pop, 1, total_pop)

            # Households: scales with population, household size changes gradually
            household_base = self.baseline_households
            household_size_start = self.baseline_population / self.baseline_households
            # Slight increase in household size (gradual decline per capita)
            household_size_trend = household_size_start * (1 + 0.0002 * months_from_start)
            total_households = total_pop / household_size_trend
            total_households = total_households * (1.0 + float(self.rng.normal(0, 0.002)))
            total_households = max(1, total_households)

            records.append({
                "date": date,
                "total_population": total_pop,
                "urban_population": urban_pop,
                "total_households": total_households,
            })

        return pd.DataFrame(records)

    def _generate_demand_and_state(self, weather_df: pd.DataFrame, demo_df: pd.DataFrame) -> pd.DataFrame:
        """Generate demand and system state iteratively (month by month)."""
        records = []

        # Initialize state variables within configured operating bands.
        reservoir_level = np.clip(self.reservoir_baseline, self.reservoir_min, self.reservoir_max)
        gw_level = np.clip(self.gw_baseline, self.gw_min, self.gw_max)
        last_demand = None

        for i, date in enumerate(self.dates):
            year = date.year
            month = date.month
            is_monsoon = self._is_monsoon(month)

            # Get weather and demographics for this month
            rainfall = weather_df.iloc[i]["rainfall_mm"]
            temp_max = weather_df.iloc[i]["temp_max_c"]
            temp_min = weather_df.iloc[i]["temp_min_c"]
            wind = weather_df.iloc[i]["wind_speed_kmh"]
            solar = weather_df.iloc[i]["solar_radiation_mj_m2"]
            total_pop = demo_df.iloc[i]["total_population"]
            total_hh = demo_df.iloc[i]["total_households"]

            # Demand generation: multi-component.
            # Add a small probability of operational shocks (policy, restrictions, heatwaves)
            # to reproduce the variability seen in real utility demand series.
            days_in_month = (date + pd.DateOffset(months=1) - date).days
            base_demand_m3 = (total_pop * self.baseline_lpd / 1000.0) * days_in_month

            if is_monsoon:
                seasonal_mult = self.monsoon_demand_mult
            else:
                seasonal_mult = self.dry_demand_mult

            temp_factor = 1.0 + (temp_max - self.temp_max_baseline) * 0.01
            rainfall_factor = 1.0 + rainfall * self.rainfall_elasticity
            weather_mult = temp_factor * rainfall_factor

            shock_factor = 1.0
            if self.rng.random() < self.demand_shock_probability:
                shock_factor *= self.rng.uniform(self.demand_shock_min, self.demand_shock_max)

            if last_demand is None:
                ar_component = base_demand_m3 * seasonal_mult * weather_mult * shock_factor
            else:
                ar_component = self.ar_strength * last_demand + (1 - self.ar_strength) * (base_demand_m3 * seasonal_mult * weather_mult * shock_factor)

            noise = float(self.rng.normal(1.0, self.demand_noise_std))
            demand = ar_component * noise
            demand = max(100.0, demand)

            last_demand = demand

            # Reservoir level: carry-over state with a more natural, noisy transition.
            rainfall_recharge = rainfall * 0.01
            demand_withdrawal = demand / 100000
            reservoir_level = reservoir_level + self.reservoir_recharge_rate * (rainfall_recharge - self.reservoir_discharge_rate * demand_withdrawal)
            reservoir_level += float(self.rng.normal(0, self.state_noise_std))
            reservoir_level = np.clip(reservoir_level, self.reservoir_min, self.reservoir_max)

            # Groundwater level: carry-over state (larger number = deeper), slower and noisier.
            rainfall_recharge_gw = rainfall * 0.001
            demand_depletion = demand / 50000
            gw_level = gw_level - self.gw_recharge_rate * rainfall_recharge_gw + self.gw_discharge_rate * demand_depletion
            gw_level += float(self.rng.normal(0, self.state_noise_std * 0.75))
            gw_level = np.clip(gw_level, self.gw_min, self.gw_max)

            # Canal discharge: supply-aware, seasonal with occasional perturbations.
            if is_monsoon:
                canal_target = self.canal_baseline * self.canal_monsoon_mult
            else:
                canal_target = self.canal_baseline * self.canal_dry_mult
            demand_factor = 1.0 + (demand / 100000) * self.canal_demand_elasticity
            canal = canal_target * demand_factor
            if self.rng.random() < self.extreme_month_probability:
                canal *= self.rng.uniform(0.8, 1.25)
            canal = np.clip(canal, self.canal_min, self.canal_max)
            canal = canal * (1.0 + float(self.rng.normal(0, 0.05)))
            canal = np.clip(canal, self.canal_min, self.canal_max)

            records.append({
                "date": date,
                "reservoir_level_m": reservoir_level,
                "groundwater_level_m_bgl": gw_level,
                "canal_discharge_cumecs": canal,
                "residential_water_demand_m3": demand,
            })

        return pd.DataFrame(records)

    def write_output(self, df: pd.DataFrame, validation_result: dict | None = None):
        """Write CSV and metadata."""
        # Ensure output directories exist
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        df.to_csv(self.csv_path, index=False)
        print(f"Wrote {self.csv_path} ({len(df)} rows)")

        summary = {
            "row_count": int(len(df)),
            "date_start": str(df["date"].iloc[0]),
            "date_end": str(df["date"].iloc[-1]),
            "area_id": self.area_id,
            "population_start": int(df["total_population"].iloc[0]),
            "population_end": int(df["total_population"].iloc[-1]),
            "demand_mean_m3": float(df["residential_water_demand_m3"].mean()),
            "demand_std_m3": float(df["residential_water_demand_m3"].std()),
            "rainfall_mean_mm": float(df["rainfall_mm"].mean()),
            "reservoir_mean_m": float(df["reservoir_level_m"].mean()),
            "groundwater_mean_m_bgl": float(df["groundwater_level_m_bgl"].mean()),
        }

        # Write metadata
        metadata = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator_version": "2.0.0",
            "schema_version": self.schema_version,
            "seed": self.seed,
            "area_id": self.area_id,
            "start_date": str(self.start_date.date()),
            "end_date": str(self.end_date.date()),
            "periods": self.periods,
            "row_count": len(df),
            "config_checksum": self.config_checksum,
            "config": self.config,
            "columns": list(df.columns),
            "validation_summary": validation_result if validation_result is not None else {"ok": None, "errors": []},
            "summary_statistics": summary,
            "synthetic": True,
            "note": "This is synthetic data for demonstrating the forecasting pipeline. It is not claimed to represent observed DNH water demand.",
        }
        self.meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"Wrote {self.meta_path}")
