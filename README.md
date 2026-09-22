# DNH Water Demand Forecasting

Repository containing the complete synthetic-first water-demand forecasting implementation, source data, generated features, trained models, evaluation outputs, figures, notebooks, and final reports. The committed `data/` and `outputs/` trees are intentional sharing artifacts so a clone includes the current dataset and results.

Quickstart

1. Create a Python 3.14 venv and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Generate the Version 2 synthetic DNH-total dataset:

```bash
python -m src.data_generation.generate_synthetic_dnh_total --config configs/synthetic_dnh_total_v2.json
```

4. Run tests:

```bash
pytest -q
```

5. Rebuild the prepared feature artifacts and chronological splits:

```bash
python -m pytest -q tests/test_phase4_dataset_preparation.py tests/test_phase4_feature_engineering.py tests/test_phase4_loading_validation.py
```

6. Rebuild the forecasting package and quarterly horizon outputs:

```bash
python scripts/package_phase5.py
python -c "from src.evaluation.horizon_forecasting import save_horizon_forecasts; save_horizon_forecasts('.')"
```

7. Execute the notebooks if their outputs need refreshing. Run package generation last because it checksums the final model and horizon artifacts.

```bash
python -m jupyter execute notebooks/PHASE_6_EVALUATION_REVIEW.ipynb --output phase6_review_executed.ipynb
```

8. Rebuild the evaluation tables, visuals, checksums, and handoff as the final write operation:

```bash
python -c "from src.evaluation.phase6_evaluation import save_phase6_package; save_phase6_package('.')"
```

9. For a full notebook sweep, execute each notebook into a temporary output directory:

```bash
for notebook in notebooks/*.ipynb; do python -m jupyter execute "$notebook" --output "$(basename "$notebook" .ipynb)_executed.ipynb"; done
```

Use `requirements-lock.txt` when reproducing the validated environment exactly.

The committed artifacts are organized as follows:

- `data/synthetic/`: canonical synthetic DNH-total dataset and metadata
- `data/processed/`: feature matrix, target, and chronological split artifacts
- `outputs/models/`: saved forecasting models and the persisted regime pipeline
- `outputs/metrics/`: model metrics, diagnostics, comparisons, and evaluation summaries
- `outputs/reports/`: predictions, quarterly forecasts, regime assignments, manifests, and handoff files
- `outputs/figures/`: EDA, feature, model, horizon, residual, and demand-regime figures

See `docs/` for project specifications and phase documentation.

Key reader-facing documents:

- [Final academic report](FINAL_REPORT.md)
- [HTML report](docs/FINAL_REPORT.html)
- [Repository walkthrough](REPOSITORY_WALKTHROUGH.md)
- [HTML walkthrough](docs/REPOSITORY_WALKTHROUGH.html)