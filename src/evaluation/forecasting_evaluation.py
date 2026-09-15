"""Shared metrics and data-loading utilities for Phase 5 forecasting models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


def load_phase5_artifacts(project_root: Path | str = ".") -> Dict[str, Any]:
    """Load the Phase 4 shared matrix and split data for Phase 5 model work."""
    root = Path(project_root)

    feature_path = root / "data" / "processed" / "features" / "dnh_total_monthly_features_v1.csv"
    target_path = root / "data" / "processed" / "features" / "dnh_total_monthly_target_v1.csv"
    metadata_path = root / "data" / "processed" / "splits" / "dnh_total_split_metadata_v1.json"

    for path in (feature_path, target_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 4 artifact not found: {path}")

    features = pd.read_csv(feature_path, index_col="date", parse_dates=True)
    target = pd.read_csv(target_path, index_col="date", parse_dates=True)
    metadata = json.loads(metadata_path.read_text())

    split_summary = pd.DataFrame(metadata["split_summary"]) 
    train_rows = split_summary.loc[split_summary["split"] == "train", "rows"].iloc[0]
    val_rows = split_summary.loc[split_summary["split"] == "validation", "rows"].iloc[0]
    test_rows = split_summary.loc[split_summary["split"] == "test", "rows"].iloc[0]

    feature_columns = list(features.columns)
    target_name = target.columns[0]

    # use the same contiguous row ordering defined by Phase 4
    train_end = train_rows
    val_end = train_end + val_rows
    test_end = val_end + test_rows

    train_X = features.iloc[:train_end].copy()
    validation_X = features.iloc[train_end:val_end].copy()
    test_X = features.iloc[val_end:test_end].copy()

    train_y = target.iloc[:train_end].copy().iloc[:, 0]
    validation_y = target.iloc[train_end:val_end].copy().iloc[:, 0]
    test_y = target.iloc[val_end:test_end].copy().iloc[:, 0]

    return {
        "features": features,
        "target": target,
        "train_X": train_X,
        "validation_X": validation_X,
        "test_X": test_X,
        "train_y": train_y,
        "validation_y": validation_y,
        "test_y": test_y,
        "target_name": target_name,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "split_summary": split_summary,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return standard regression metrics for demand forecasting comparison."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))) * 100.0
    smape = float(
        np.mean(
            2.0 * np.abs(y_true - y_pred) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)
        )
        * 100.0
    )
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
    }


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return compact bias and error diagnostics for a forecast series."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residuals.shape != np.asarray(y_true).shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "mean_residual": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_mae": float(np.mean(np.abs(residuals))),
        "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_absolute_residual": float(np.max(np.abs(residuals))),
    }


def evaluate_supervised_model(
    model: Any,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    test_X: pd.DataFrame | None = None,
    test_y: pd.Series | None = None,
    *,
    fit_model: bool = True,
) -> Dict[str, Any]:
    """Fit a supervised model and score validation, optionally test, predictions."""
    if (test_X is None) != (test_y is None):
        raise ValueError("test_X and test_y must be provided together")

    if fit_model:
        model.fit(train_X, train_y)
    predictions: Dict[str, pd.Series] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for split_name, features, target in (
        ("validation", validation_X, validation_y),
        ("test", test_X, test_y),
    ):
        if features is None or target is None:
            continue
        prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"{split_name} predictions must match target length")
        predictions[split_name] = pd.Series(prediction, index=target.index, name="prediction")
        metrics[split_name] = compute_regression_metrics(target.to_numpy(), prediction)

    return {"model": model, "predictions": predictions, "metrics": metrics}


def save_model_run(
    run_name: str,
    model: Any,
    predictions: Mapping[str, pd.Series],
    metrics: Mapping[str, Mapping[str, float]],
    configuration: Mapping[str, Any],
    project_root: Path | str = ".",
    targets: Mapping[str, pd.Series] | None = None,
) -> Dict[str, Path]:
    """Save a fitted model, predictions, metrics, and configuration metadata."""
    from joblib import dump

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    output_root = Path(project_root)
    model_dir = output_root / "outputs" / "models"
    metrics_dir = output_root / "outputs" / "metrics"
    report_dir = output_root / "outputs" / "reports"
    for directory in (model_dir, metrics_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{run_name}.joblib"
    predictions_path = report_dir / f"{run_name}_predictions.csv"
    metrics_path = metrics_dir / f"{run_name}_metrics.json"
    configuration_path = metrics_dir / f"{run_name}_configuration.json"
    residuals_path = report_dir / f"{run_name}_residuals.csv"
    diagnostics_path = metrics_dir / f"{run_name}_residual_diagnostics.json"

    dump(model, model_path)
    prediction_frame = pd.concat(predictions.values(), axis=1)
    prediction_frame.columns = list(predictions.keys())
    prediction_frame.to_csv(predictions_path, index_label="date")
    metrics_path.write_text(json.dumps(json_safe(metrics), indent=2))
    configuration_path.write_text(json.dumps(json_safe(configuration), indent=2))

    output_paths = {
        "model": model_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "configuration": configuration_path,
    }
    if targets is not None:
        residual_frame = pd.concat(
            {
                split: targets[split] - predictions[split]
                for split in predictions
            },
            axis=1,
        )
        residual_frame.to_csv(residuals_path, index_label="date")
        diagnostics = {
            split: compute_residual_diagnostics(
                targets[split].to_numpy(), predictions[split].to_numpy()
            )
            for split in predictions
        }
        diagnostics_path.write_text(json.dumps(json_safe(diagnostics), indent=2))
        output_paths.update({"residuals": residuals_path, "residual_diagnostics": diagnostics_path})

    return output_paths
