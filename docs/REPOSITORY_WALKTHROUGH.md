# Repository Walkthrough

## 1. Project Purpose

This repository implements a synthetic-first monthly residential water-demand forecasting pipeline for the combined Dadra and Nagar Haveli study area. The approved modelling grain is one row per month for one entity, `DNH_total`, with residential water demand measured in cubic metres per month.

The project is organized as a sequence of validated stages:

1. Research design and governance
2. Data contract definition
3. Synthetic DNH-total data generation
4. Exploratory analysis and feature engineering
5. Forecasting models and baseline comparison
6. Evaluation, horizon review, and demand-regime analysis
7. Final reporting and reproducibility documentation

The current repository contains the completed implementation and the final report materials. Real-data replacement is outside the current synthetic benchmark.

## 2. Start Here

Read these files first:

- [README.md](../README.md): short setup and reproducibility entry point
- [FINAL_REPORT.md](FINAL_REPORT.md): academic summary of the problem, methods, results, visuals, and limitations
- [DATA_CONTRACT.md](DATA_CONTRACT.md): authoritative schema, units, exclusions, and quality rules
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): original project roadmap
- [DNH_Water_Demand_Forecasting_Knowledge_Transfer.md](DNH_Water_Demand_Forecasting_Knowledge_Transfer.md): project context and design rationale

## 3. Top-Level Folders

### `configs/`

Configuration files for deterministic synthetic data generation. The main configuration is [synthetic_dnh_total_v2.json](../configs/synthetic_dnh_total_v2.json).

It defines the date range, random seed, DNH-total geography, weather behavior, demographic growth, demand generation, system-state dynamics, realism settings, and validation thresholds.

### `data/`

Data at different pipeline stages:

- `data/synthetic/raw/`: generated synthetic canonical data
- `data/synthetic/metadata/`: generator metadata and provenance
- `data/processed/features/`: persisted feature matrix and target
- `data/processed/splits/`: chronological split metadata
- `data/real/`: reserved location for future confirmed real data

### `src/`

Reusable implementation modules grouped by responsibility:

- `src/data_generation/`: synthetic DNH-total generator and command-line entry point
- `src/validation/`: schema, domain, chronology, and realism validation
- `src/features/`: canonical loading, feature engineering, shared matrix preparation, and split creation
- `src/models/`: seasonal-naive, Random Forest, ANN, and LSTM model implementations
- `src/evaluation/`: metrics, saved prediction comparison, horizon forecasts, Phase 5 handoff, and Phase 6 evaluation/package generation
- `src/clustering/`: clustering-related utilities
- `src/preprocessing/`: preprocessing utilities
- `src/utils/`: general support utilities

### `tests/`

Automated regression coverage for the implementation:

- Synthetic data generation and validation
- Phase 4 loading, feature engineering, preparation, and chronological splits
- Baseline, Random Forest, ANN, and LSTM behavior
- Model comparison and packaging
- Sequential horizon forecasts and quarterly sums
- Phase 6 scorecards, residuals, regimes, visuals, manifests, and handoff

### `notebooks/`

Reader-facing demonstrations and review notebooks. The notebooks are useful for understanding the workflow visually; the reusable Python modules remain the source of truth for computation.

## 4. Notebook Guide

### Data and feature preparation

- [FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb](../notebooks/FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb): feature availability and inclusion decisions
- [PHASE_4_TASK_4_3_EDA.ipynb](../notebooks/PHASE_4_TASK_4_3_EDA.ipynb): exploratory analysis of demand, weather, demographics, and system state
- [SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb](../notebooks/SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb): shared predictors, target separation, and chronological partitions
- [PREPARATION_HANDOFF_REVIEW.ipynb](../notebooks/PREPARATION_HANDOFF_REVIEW.ipynb): preparation outputs and handoff review

### Model reviews

- [SEASONAL_NAIVE_BASELINE_REVIEW.ipynb](../notebooks/SEASONAL_NAIVE_BASELINE_REVIEW.ipynb): seasonal reference model
- [RANDOM_FOREST_FORECAST_REVIEW.ipynb](../notebooks/RANDOM_FOREST_FORECAST_REVIEW.ipynb): Random Forest training and diagnostics
- [ANN_FORECAST_REVIEW.ipynb](../notebooks/ANN_FORECAST_REVIEW.ipynb): ANN training, tuning, and diagnostics
- [LSTM_FORECAST_REVIEW.ipynb](../notebooks/LSTM_FORECAST_REVIEW.ipynb): LSTM sequence preparation, training, and diagnostics
- [PHASE_5_MODEL_COMPARISON_REVIEW.ipynb](../notebooks/PHASE_5_MODEL_COMPARISON_REVIEW.ipynb): common model comparison and residual review
- [PHASE_5_HORIZON_FORECAST_REVIEW.ipynb](../notebooks/PHASE_5_HORIZON_FORECAST_REVIEW.ipynb): one-month and three-month sequential forecasts
- [PHASE_6_EVALUATION_REVIEW.ipynb](../notebooks/PHASE_6_EVALUATION_REVIEW.ipynb): final scorecards, residuals, horizon behavior, and demand regimes

## 5. Important Outputs

### Processed data

- `data/processed/features/dnh_total_monthly_features_v1.csv`
- `data/processed/features/dnh_total_monthly_target_v1.csv`
- `data/processed/splits/dnh_total_split_metadata_v1.json`

### Model outputs

Stored mainly under `outputs/models/`, `outputs/metrics/`, and `outputs/reports/`:

- Saved model files for all four forecasting approaches
- Validation and test predictions
- Model configuration and metric JSON files
- Residual files and diagnostics
- Tuning results and training histories

### Horizon outputs

- `outputs/reports/phase5_horizon_forecasts.csv`: three sequential monthly forecasts per model
- `outputs/reports/phase5_quarterly_forecasts.csv`: summed three-month forecast and actual totals per model
- `outputs/metrics/phase5_horizon_metrics.json`: horizon error summaries
- `outputs/metrics/phase6_horizon_review.csv`: one-month versus aggregate three-month review

### Evaluation outputs

- `outputs/metrics/phase6_scorecard.csv`: validation and test metrics
- `outputs/metrics/phase6_baseline_comparison.csv`: model differences against seasonal naive
- `outputs/metrics/phase6_residual_diagnostics.csv`: residual summaries
- `outputs/reports/phase6_seasonal_errors.csv`: calendar-month error summaries
- `outputs/reports/phase6_large_errors.csv`: largest-error months
- `outputs/metrics/phase6_stability_review.csv`: validation/test ranking stability
- `outputs/reports/phase6_regime_assignments.csv`: monthly regime labels
- `outputs/metrics/phase6_regime_metadata.json`: scaler, features, cluster selection, and names
- `outputs/models/phase6_regime_kmeans.joblib`: persisted scaler-plus-K-Means pipeline

### Figures

Phase 6 figures are under `outputs/figures/phase6/`:

- Forecast-versus-actual comparisons
- Residual time series and distributions
- RMSE comparison
- Sequential horizon forecasts
- Regime sizes, centroids, and timeline

## 6. Reproducibility Sequence on Windows

Run commands from the repository root in PowerShell.

### Install the validated environment

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

### Generate and validate synthetic data

```powershell
.\venv\Scripts\python.exe -m src.data_generation.generate_synthetic_dnh_total --config configs/synthetic_dnh_total_v2.json
```

### Run the complete test suite

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

### Execute notebooks into a temporary directory

```powershell
$validationDir = Join-Path $env:TEMP 'ml_based_irrigation_notebook_validation'
New-Item -ItemType Directory -Force -Path $validationDir | Out-Null
foreach ($notebook in Get-ChildItem notebooks\*.ipynb) {
    $output = Join-Path $validationDir ($notebook.BaseName + '_executed.ipynb')
    .\venv\Scripts\python.exe -m jupyter execute $notebook.FullName --output $output --timeout=240
}
```

### Regenerate the Phase 5 package

```powershell
.\venv\Scripts\python.exe scripts\package_phase5.py
.\venv\Scripts\python.exe -c "from src.evaluation.horizon_forecasting import save_horizon_forecasts; save_horizon_forecasts('.')"
```

### Regenerate the Phase 6 package last

```powershell
.\venv\Scripts\python.exe -c "from src.evaluation.phase6_evaluation import save_phase6_package; save_phase6_package('.')"
```

The Phase 6 package refreshes the Phase 5 handoff before checksumming the final evaluation package. Run it after any command that rewrites forecasts, metrics, reports, or model artifacts.

## 7. How to Read the Implementation

A practical code-reading order is:

1. Read the data contract and configuration.
2. Read `src/data_generation/synthetic_dnh_total_generator.py` and `src/validation/validate_dnh_total_dataset.py`.
3. Read `src/features/phase4_data_loader.py`, `phase4_feature_engineering.py`, and `phase4_dataset_preparation.py`.
4. Read the four model modules under `src/models/`.
5. Read `src/evaluation/forecasting_evaluation.py` and `model_comparison.py`.
6. Read `src/evaluation/horizon_forecasting.py` for recursive forecasts and quarterly sums.
7. Read `src/evaluation/phase6_evaluation.py` for scorecards, residual analysis, regimes, visuals, manifests, and handoff.
8. Use the tests to see the intended contracts and failure behavior.
9. Use the notebooks to see the same workflow presented visually.

## 8. Interpretation Rules

- Synthetic benchmark performance is not operational accuracy.
- Validation and test windows are chronological and must remain untouched by later tuning.
- K-Means regimes describe exploratory monthly behavior, not geography.
- Quarterly values are sums of sequential monthly forecasts, not predictions from a separate quarterly model.
- Residual quantiles are descriptive error summaries, not confidence intervals.
- Real-data replacement requires renewed validation, feature-availability review, model selection, and evaluation.

## 9. Current Handoff

The completed evaluation package is marked ready for final reporting in [phase6_handoff_manifest.md](../outputs/reports/phase6_handoff_manifest.md). The final academic report is [FINAL_REPORT.md](FINAL_REPORT.md).
