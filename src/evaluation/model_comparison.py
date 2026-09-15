"""Reproducible comparison and diagnostics for completed Phase 5 models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .forecasting_evaluation import (
    compute_regression_metrics,
    compute_residual_diagnostics,
    load_phase5_artifacts,
)


DEFAULT_MODEL_RUNS = {
    "Seasonal naive": "seasonal_naive_baseline",
    "Random Forest": "random_forest_forecaster",
    "ANN": "ann_forecaster",
    "LSTM": "lstm_forecaster",
}


def load_saved_predictions(
    project_root: Path | str = ".",
    model_runs: Mapping[str, str] = DEFAULT_MODEL_RUNS,
) -> dict[str, dict[str, pd.Series]]:
    """Load persisted validation/test predictions and enforce the shared indexes."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    predictions: dict[str, dict[str, pd.Series]] = {}

    for model_name, run_name in model_runs.items():
        path = root / "outputs" / "reports" / f"{run_name}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Saved predictions not found: {path}")
        frame = pd.read_csv(path, index_col="date", parse_dates=True)
        model_predictions: dict[str, pd.Series] = {}
        for split in ("validation", "test"):
            if split not in frame.columns:
                raise ValueError(f"Missing {split} predictions for {model_name}")
            target = artifacts[f"{split}_y"]
            series = frame[split].dropna()
            if not series.index.equals(target.index):
                raise ValueError(f"{model_name} {split} prediction index does not match target")
            model_predictions[split] = series.rename(model_name)
        predictions[model_name] = model_predictions

    return predictions


def build_comparison_tables(
    project_root: Path | str = ".",
    model_runs: Mapping[str, str] = DEFAULT_MODEL_RUNS,
) -> dict[str, pd.DataFrame]:
    """Recompute comparable metric, residual, and ranking tables from saved predictions."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    predictions = load_saved_predictions(root, model_runs)
    metric_rows = []
    residual_rows = []

    for model_name, split_predictions in predictions.items():
        for split, prediction in split_predictions.items():
            target = artifacts[f"{split}_y"]
            metric_rows.append({
                "model": model_name,
                "split": split,
                **compute_regression_metrics(target.to_numpy(), prediction.to_numpy()),
            })
            residual_rows.append({
                "model": model_name,
                "split": split,
                **compute_residual_diagnostics(target.to_numpy(), prediction.to_numpy()),
            })

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "rmse", "mae"]).reset_index(drop=True)
    residuals = pd.DataFrame(residual_rows).sort_values(["split", "residual_rmse"]).reset_index(drop=True)
    rankings = metrics.assign(
        rmse_rank=metrics.groupby("split")["rmse"].rank(method="min"),
        mae_rank=metrics.groupby("split")["mae"].rank(method="min"),
    ).sort_values(["split", "rmse_rank", "mae_rank"]).reset_index(drop=True)
    return {"metrics": metrics, "residuals": residuals, "rankings": rankings}


def save_comparison_outputs(
    project_root: Path | str = ".",
    model_runs: Mapping[str, str] = DEFAULT_MODEL_RUNS,
    excluded_models: Sequence[str] = (),
) -> dict[str, Path]:
    """Save comparison tables, plots, and an explicit model-coverage note."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    predictions = load_saved_predictions(root, model_runs)
    tables = build_comparison_tables(root, model_runs)
    metrics_dir = root / "outputs" / "metrics"
    report_dir = root / "outputs" / "reports"
    figure_dir = root / "outputs" / "figures" / "model_comparison"
    for directory in (metrics_dir, report_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "phase5_model_comparison_metrics.csv"
    residuals_path = metrics_dir / "phase5_model_comparison_residuals.csv"
    rankings_path = metrics_dir / "phase5_model_comparison_rankings.csv"
    coverage_path = metrics_dir / "phase5_model_comparison_coverage.json"
    tables["metrics"].to_csv(metrics_path, index=False)
    tables["residuals"].to_csv(residuals_path, index=False)
    tables["rankings"].to_csv(rankings_path, index=False)
    coverage_path.write_text(json.dumps({
        "included_models": list(model_runs.keys()),
        "excluded_models": list(excluded_models),
        "reason": "All included models have completed their training and persisted prediction workflow.",
    }, indent=2))

    for split in ("validation", "test"):
        figure, axis = plt.subplots(figsize=(14, 6))
        actual = artifacts[f"{split}_y"]
        axis.plot(actual.index, actual.values, color="#1f2937", linewidth=2.5, label="Actual")
        for model_name, split_predictions in predictions.items():
            axis.plot(split_predictions[split].index, split_predictions[split].values, linewidth=1.8, label=model_name)
        axis.set_title(f"{split.title()} period: model forecasts compared with actual demand")
        axis.set_ylabel("Residential water demand (m3)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / f"01_{split}_forecast_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for axis, split in zip(axes, ("validation", "test")):
        actual = artifacts[f"{split}_y"]
        for model_name, split_predictions in predictions.items():
            residual = actual - split_predictions[split]
            axis.axhline(0, color="#6b7280", linewidth=1)
            axis.plot(residual.index, residual.values, linewidth=1.6, label=model_name)
        axis.set_title(f"{split.title()} residuals")
        axis.set_ylabel("Actual minus forecast")
        axis.legend()
    figure.tight_layout()
    residual_figure_path = figure_dir / "02_residual_comparison.png"
    figure.savefig(residual_figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {
        "metrics": metrics_path,
        "residuals": residuals_path,
        "rankings": rankings_path,
        "coverage": coverage_path,
        "residual_figure": residual_figure_path,
        "validation_figure": figure_dir / "01_validation_forecast_comparison.png",
        "test_figure": figure_dir / "01_test_forecast_comparison.png",
    }