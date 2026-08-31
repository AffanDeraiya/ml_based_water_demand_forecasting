import json
from pathlib import Path
import pandas as pd

from src.data_generation.synthetic_generator import SyntheticGenerator


def test_default_generation_creates_files(tmp_path, monkeypatch):
    cfg_path = Path("configs/synthetic_data_v1.json")
    cfg = json.loads(cfg_path.read_text())
    # set outputs to tmp_path via monkeypatching if needed; for now run and assert files exist
    gen = SyntheticGenerator(cfg)
    gen.out_raw = tmp_path / "synthetic_zone_monthly_v1.csv"
    gen.out_agg = tmp_path / "synthetic_dnh_total_monthly_v1.csv"
    gen.out_meta = tmp_path / "synthetic_zone_monthly_v1_metadata.json"
    gen.generate_all()

    assert gen.out_raw.exists()
    df = pd.read_csv(gen.out_raw)
    assert not df.empty
    assert set(["date", "area_id"]).issubset(df.columns)

    assert gen.out_agg.exists()
    agg = pd.read_csv(gen.out_agg)
    assert not agg.empty
    assert (agg["area_id"] == "DNH_total").all()

    assert gen.out_meta.exists()
    meta = json.loads(gen.out_meta.read_text())
    assert meta["seed"] == cfg.get("seed")
