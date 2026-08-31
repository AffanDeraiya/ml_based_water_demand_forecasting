# DNH Water Demand Forecasting

Repository for reproducible water-demand forecasting pipeline (synthetic first, real later).

Quickstart

1. Create a Python 3.14 venv and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Generate synthetic data:

```bash
python -m src.data_generation.generate_synthetic_data --config configs/synthetic_data_v1.json
```

4. Run tests:

```bash
pytest -q
```

See `docs/` for project documentation.