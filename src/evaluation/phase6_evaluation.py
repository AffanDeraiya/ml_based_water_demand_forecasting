"""Phase 6 validation, scorecard, residual, horizon, and regime analysis."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .forecasting_evaluation import compute_regression_metrics, compute_residual_diagnostics, load_phase5_artifacts
from .model_comparison import DEFAULT_MODEL_RUNS, load_saved_predictions
from .phase5_handoff import write_phase5_handoff


def validate_phase5_inputs(project_root: Path | str = ".") -> dict[str, Any]:
    """Validate Phase 5 manifests, four-model predictions, horizons, and split sizes."""
    root = Path(project_root)
    manifest_path = root / "outputs" / "reports" / "phase5_artifact_manifest.json"
    handoff_path = root / "outputs" / "reports" / "phase5_handoff_manifest.json"
    horizon_path = root / "outputs" / "reports" / "phase5_horizon_forecasts.csv"
    quarterly_path = root / "outputs" / "reports" / "phase5_quarterly_forecasts.csv"
    for path in (manifest_path, handoff_path, horizon_path, quarterly_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 5 input is missing: {path}")

    manifest = json.loads(manifest_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    artifacts = load_phase5_artifacts(root)
    predictions = load_saved_predictions(root)
    horizon = pd.read_csv(horizon_path)
    quarterly = pd.read_csv(quarterly_path)
    required_models = {"Seasonal naive", "Random Forest", "ANN", "LSTM"}
    if manifest.get("status") != "READY_FOR_HANDOFF" or manifest.get("missing_artifacts"):
        raise ValueError("Phase 5 artifact manifest is not ready")
    if handoff.get("status") != "READY_FOR_PHASE_6":
        raise ValueError("Phase 5 handoff is not ready for Phase 6")
    if set(predictions) != required_models or set(horizon["model"]) != required_models:
        raise ValueError("Phase 5 model coverage is incomplete")
    if [len(artifacts[name]) for name in ("train_y", "validation_y", "test_y")] != [108, 24, 36]:
        raise ValueError("Phase 5 split sizes do not match the accepted contract")
    if len(horizon) != 12 or set(horizon["horizon_step"]) != {1, 2, 3}:
        raise ValueError("Phase 5 horizon outputs are incomplete")
    if len(quarterly) != 4 or set(quarterly["model"]) != required_models:
        raise ValueError("Phase 5 quarterly outputs are incomplete")
    integrity_issues: list[str] = []
    source_of_truth = handoff.get("source_of_truth", {})
    expected_sources = {
        "feature_matrix": "data/processed/features/dnh_total_monthly_features_v1.csv",
        "target": "data/processed/features/dnh_total_monthly_target_v1.csv",
        "split_metadata": "data/processed/splits/dnh_total_split_metadata_v1.json",
    }
    for name, expected_path in expected_sources.items():
        if source_of_truth.get(name) != expected_path:
            integrity_issues.append(f"Handoff source-of-truth path mismatch for {name}")

    feature_index = artifacts["features"].index
    target_index = artifacts["target"].index
    if not feature_index.equals(target_index):
        integrity_issues.append("Feature and target indexes do not align")
    if len(feature_index) != 168 or len(target_index) != 168:
        integrity_issues.append("Complete feature and target rows do not equal 168")
    split_summary = artifacts["split_summary"]
    if split_summary["rows"].sum() != 168:
        integrity_issues.append("Split row counts do not sum to 168")
    for split in ("train", "validation", "test"):
        split_rows = int(split_summary.loc[split_summary["split"] == split, "rows"].iloc[0])
        split_index = artifacts[f"{split}_y"].index
        if len(split_index) != split_rows:
            integrity_issues.append(f"{split} target row count does not match split metadata")
        if not artifacts[f"{split}_X"].index.equals(split_index):
            integrity_issues.append(f"{split} feature and target indexes do not align")

    manifest_artifacts = {item["path"]: item for item in manifest.get("artifacts", [])}
    for relative_path in expected_sources.values():
        path = root / relative_path
        entry = manifest_artifacts.get(relative_path)
        if entry is None or not path.exists():
            integrity_issues.append(f"Manifest entry or source artifact is missing: {relative_path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            integrity_issues.append(f"Source artifact checksum mismatch: {relative_path}")
    if integrity_issues:
        raise ValueError("Phase 5 input integrity checks failed: " + "; ".join(integrity_issues))
    return {
        "manifest": manifest,
        "handoff": handoff,
        "artifacts": artifacts,
        "predictions": predictions,
        "horizon": horizon,
        "quarterly": quarterly,
        "models": sorted(required_models),
        "integrity_issues": integrity_issues,
        "integrity_checks": {
            "prediction_indexes_aligned": True,
            "feature_target_indexes_aligned": True,
            "source_checksums_match": True,
        },
    }


def build_phase6_scorecard(project_root: Path | str = ".") -> pd.DataFrame:
    """Recompute the authoritative validation/test scorecard from saved predictions."""
    inputs = validate_phase5_inputs(project_root)
    rows = []
    for model_name, split_predictions in inputs["predictions"].items():
        for split, prediction in split_predictions.items():
            target = inputs["artifacts"][f"{split}_y"]
            from .forecasting_evaluation import compute_regression_metrics

            rows.append({
                "model": model_name,
                "split": split,
                **compute_regression_metrics(target.to_numpy(), prediction.to_numpy()),
            })
    return pd.DataFrame(rows).sort_values(["split", "rmse", "mae"]).reset_index(drop=True)


def build_phase6_selection_review(project_root: Path | str = ".") -> pd.DataFrame:
    """Identify objective leaders without changing the Phase 5 selection policy."""
    scorecard = build_phase6_scorecard(project_root)
    rows = []
    for split, frame in scorecard.groupby("split", sort=True):
        rmse_leader = frame.sort_values(["rmse", "mae", "model"]).iloc[0]
        mae_leader = frame.sort_values(["mae", "rmse", "model"]).iloc[0]
        rows.append({
            "split": split,
            "rmse_leader": rmse_leader["model"],
            "rmse": float(rmse_leader["rmse"]),
            "mae_leader": mae_leader["model"],
            "mae": float(mae_leader["mae"]),
            "rmse_leader_matches_mae_leader": bool(rmse_leader["model"] == mae_leader["model"]),
        })
    review = pd.DataFrame(rows)
    test_leader = review.loc[review["split"] == "test", "rmse_leader"].iloc[0]
    validation_leader = review.loc[review["split"] == "validation", "rmse_leader"].iloc[0]
    review["test_rmse_leader_matches_validation"] = test_leader == validation_leader
    return review.sort_values("split").reset_index(drop=True)


def build_phase6_baseline_comparison(project_root: Path | str = ".") -> pd.DataFrame:
    """Compare every completed model with the persisted seasonal-naive baseline."""
    scorecard = build_phase6_scorecard(project_root)
    baseline = scorecard[scorecard["model"] == "Seasonal naive"].set_index("split")
    rows = []
    for _, row in scorecard[scorecard["model"] != "Seasonal naive"].iterrows():
        reference = baseline.loc[row["split"]]
        rows.append({
            "model": row["model"],
            "split": row["split"],
            "baseline_mae": float(reference["mae"]),
            "model_mae": float(row["mae"]),
            "mae_delta_vs_baseline": float(row["mae"] - reference["mae"]),
            "mae_relative_change_pct": float((row["mae"] - reference["mae"]) / reference["mae"] * 100.0),
            "baseline_rmse": float(reference["rmse"]),
            "model_rmse": float(row["rmse"]),
            "rmse_delta_vs_baseline": float(row["rmse"] - reference["rmse"]),
            "rmse_relative_change_pct": float((row["rmse"] - reference["rmse"]) / reference["rmse"] * 100.0),
        })
    return pd.DataFrame(rows).sort_values(["split", "rmse_delta_vs_baseline", "mae_delta_vs_baseline"]).reset_index(drop=True)


def build_phase6_residual_analysis(project_root: Path | str = ".") -> dict[str, pd.DataFrame]:
    """Summarize residual diagnostics, month-level errors, and horizon degradation."""
    root = Path(project_root)
    inputs = validate_phase5_inputs(root)
    diagnostics = []
    monthly_rows = []
    for model_name, split_predictions in inputs["predictions"].items():
        run_name = DEFAULT_MODEL_RUNS[model_name]
        residual_path = root / "outputs" / "reports" / f"{run_name}_residuals.csv"
        diagnostics_path = root / "outputs" / "metrics" / f"{run_name}_residual_diagnostics.json"
        if not residual_path.exists() or not diagnostics_path.exists():
            raise FileNotFoundError(f"Saved residual artifacts are missing for {model_name}")
        saved_residuals = pd.read_csv(residual_path, index_col="date", parse_dates=True)
        saved_diagnostics = json.loads(diagnostics_path.read_text())
        for split, prediction in split_predictions.items():
            target = inputs["artifacts"][f"{split}_y"]
            residual = target - prediction
            saved_series = saved_residuals[split].dropna() if split in saved_residuals.columns else None
            if saved_series is None or not saved_series.index.equals(target.index):
                raise ValueError(f"{model_name} {split} residual artifact does not match target index")
            if not np.allclose(saved_series.to_numpy(), residual.to_numpy()):
                raise ValueError(f"{model_name} {split} residual artifact does not match saved predictions")
            if split not in saved_diagnostics:
                raise ValueError(f"{model_name} {split} residual diagnostics are missing")
            diagnostics.append({
                "model": model_name,
                "split": split,
                **compute_residual_diagnostics(target.to_numpy(), prediction.to_numpy()),
                "p05_residual": float(residual.quantile(0.05)),
                "p95_residual": float(residual.quantile(0.95)),
            })
            for timestamp, value in residual.items():
                monthly_rows.append({
                    "model": model_name,
                    "split": split,
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "month": timestamp.month,
                    "residual": float(value),
                    "absolute_error": float(abs(value)),
                })
    horizon = inputs["horizon"].copy()
    horizon["absolute_error"] = abs(horizon["actual_m3"] - horizon["forecast_m3"])
    horizon_summary = horizon.groupby(["model", "horizon_step"], as_index=False).agg(
        forecast_m3=("forecast_m3", "first"),
        actual_m3=("actual_m3", "first"),
        absolute_error=("absolute_error", "first"),
    )
    seasonal_summary = (
        pd.DataFrame(monthly_rows)
        .groupby(["model", "split", "month"], as_index=False)
        .agg(
            count=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("residual", lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))),
        )
    )
    large_errors = pd.DataFrame(monthly_rows).sort_values(
        ["model", "split", "absolute_error"], ascending=[True, True, False]
    )
    large_errors["error_rank"] = large_errors.groupby(["model", "split"]).cumcount() + 1
    large_errors = large_errors[large_errors["error_rank"] <= 3].reset_index(drop=True)
    diagnostics_frame = pd.DataFrame(diagnostics)
    ranking = diagnostics_frame.assign(
        rmse_rank=diagnostics_frame.groupby("split")["residual_rmse"].rank(method="min"),
        mae_rank=diagnostics_frame.groupby("split")["residual_mae"].rank(method="min"),
    )
    stability = ranking.pivot(index="model", columns="split", values=["rmse_rank", "mae_rank"]).reset_index()
    stability.columns = ["_".join(column).strip("_") for column in stability.columns]
    stability["rmse_rank_change"] = stability["rmse_rank_test"] - stability["rmse_rank_validation"]
    stability["mae_rank_change"] = stability["mae_rank_test"] - stability["mae_rank_validation"]
    stability["ranking_stable"] = (stability["rmse_rank_change"] == 0) & (stability["mae_rank_change"] == 0)
    return {
        "diagnostics": pd.DataFrame(diagnostics),
        "monthly_errors": pd.DataFrame(monthly_rows),
        "horizon_errors": horizon_summary,
        "seasonal_errors": seasonal_summary,
        "large_errors": large_errors,
        "stability": stability,
    }


def build_phase6_horizon_review(project_root: Path | str = ".") -> pd.DataFrame:
    """Summarize one-month error and aggregate three-month error for each model."""
    root = Path(project_root)
    horizon = pd.read_csv(root / "outputs" / "reports" / "phase5_horizon_forecasts.csv")
    rows: list[dict[str, float | str | int]] = []
    for model_name in sorted(horizon["model"].unique()):
        model_horizon = horizon[horizon["model"] == model_name].sort_values("horizon_step")
        if set(model_horizon["horizon_step"]) != {1, 2, 3}:
            continue
        errors = model_horizon["actual_m3"] - model_horizon["forecast_m3"]
        rows.append({
            "model": model_name,
            "one_month_absolute_error": float(np.abs(errors.iloc[0])),
            "three_month_mae": float(np.mean(np.abs(errors))),
            "three_month_rmse": float(np.sqrt(np.mean(errors.to_numpy() ** 2))),
        })
    review = pd.DataFrame(rows)
    return review.sort_values("model").reset_index(drop=True)


def build_phase6_regime_analysis(project_root: Path | str = ".") -> dict[str, Any]:
    """Fit leakage-safe K-Means demand regimes using training-period scaling only."""
    root = Path(project_root)
    artifacts = load_phase5_artifacts(root)
    features = artifacts["features"].copy()
    split_summary = artifacts["split_summary"].copy()

    if len(features) != 168:
        raise ValueError("Expected 168 complete feature rows for regime analysis")

    feature_columns = [column for column in features.columns if column not in {"date"}]
    if not feature_columns:
        raise ValueError("No feature columns available for regime analysis")

    train_dates = pd.Index(features.index[:split_summary.loc[split_summary["split"] == "train", "rows"].iloc[0]])
    validation_dates = pd.Index(features.index[split_summary.loc[split_summary["split"] == "train", "rows"].iloc[0]:split_summary.loc[split_summary["split"] == "train", "rows"].iloc[0] + split_summary.loc[split_summary["split"] == "validation", "rows"].iloc[0]])
    test_dates = pd.Index(features.index[split_summary.loc[split_summary["split"] == "train", "rows"].iloc[0] + split_summary.loc[split_summary["split"] == "validation", "rows"].iloc[0]:])

    train_X = features.loc[train_dates, feature_columns]
    validation_X = features.loc[validation_dates, feature_columns]
    test_X = features.loc[test_dates, feature_columns]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X)
    candidate_ks = range(2, 7)
    best_model: KMeans | None = None
    best_score = -np.inf
    best_k = 2
    candidate_scores: list[dict[str, float | int]] = []
    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(train_scaled)
        score = silhouette_score(train_scaled, labels)
        candidate_scores.append({"k": int(k), "silhouette_score": float(score), "inertia": float(model.inertia_)})
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
    if best_model is None:
        raise RuntimeError("No valid KMeans model could be fit for regime analysis")

    train_labels = best_model.predict(train_scaled)
    validation_labels = best_model.predict(scaler.transform(validation_X))
    test_labels = best_model.predict(scaler.transform(test_X))

    demand_level_columns = [
        column for column in feature_columns
        if column.startswith("demand_lag_") or column.startswith("demand_rolling_mean_")
    ]
    variability_columns = [column for column in feature_columns if column.startswith("demand_rolling_std_")]
    level_indexes = [feature_columns.index(column) for column in demand_level_columns]
    variability_indexes = [feature_columns.index(column) for column in variability_columns]
    level_scores = best_model.cluster_centers_[:, level_indexes].mean(axis=1)
    variability_scores = best_model.cluster_centers_[:, variability_indexes].mean(axis=1)
    level_median = float(np.median(level_scores))
    variability_median = float(np.median(variability_scores))
    regime_names = {
        cluster: (
            f"{'higher-demand' if level_scores[cluster] >= level_median else 'lower-demand'} "
            f"{'variable' if variability_scores[cluster] >= variability_median else 'stable'} months"
        )
        for cluster in range(best_k)
    }

    assignments = []
    for dates, split_name, labels in (
        (train_dates, "train", train_labels),
        (validation_dates, "validation", validation_labels),
        (test_dates, "test", test_labels),
    ):
        for timestamp, cluster in zip(dates, labels):
            assignments.append({
                "date": timestamp.strftime("%Y-%m-%d"),
                "split": split_name,
                "cluster": int(cluster),
                "regime_name": regime_names[int(cluster)],
            })
    assignments_df = pd.DataFrame(assignments).sort_values(["split", "date"]).reset_index(drop=True)

    return {
        "selected_k": int(best_k),
        "silhouette_score": float(best_score),
        "scaler": scaler,
        "model": best_model,
        "pipeline": Pipeline([("scaler", scaler), ("kmeans", best_model)]),
        "assignments": assignments_df,
        "feature_columns": feature_columns,
        "train_shape": train_X.shape,
        "validation_shape": validation_X.shape,
        "test_shape": test_X.shape,
        "candidate_scores": candidate_scores,
        "regime_names": regime_names,
        "scaler_parameters": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "random_state": 42,
    }


def build_phase6_regime_error_summary(project_root: Path | str = ".") -> pd.DataFrame:
    """Summarize absolute error and bias by model and demand regime."""
    root = Path(project_root)
    regimes = build_phase6_regime_analysis(root)
    assignments = regimes["assignments"].copy()
    assignments["date"] = pd.to_datetime(assignments["date"])
    assignments = assignments.set_index("date")
    artifacts = load_phase5_artifacts(root)
    predictions = load_saved_predictions(root)
    rows: list[dict[str, Any]] = []

    for model_name, split_predictions in predictions.items():
        for split, prediction in split_predictions.items():
            target = artifacts[f"{split}_y"]
            residual = target - prediction
            combined = pd.DataFrame({
                "residual": residual.values,
                "abs_residual": np.abs(residual.values),
                "cluster": assignments.loc[target.index, "cluster"].values,
                "regime_name": assignments.loc[target.index, "regime_name"].values,
            }, index=target.index)
            for cluster, frame in combined.groupby("cluster"):
                rows.append({
                    "model": model_name,
                    "split": split,
                    "cluster": int(cluster),
                    "regime_name": str(frame["regime_name"].iloc[0]),
                    "count": int(len(frame)),
                    "bias": float(frame["residual"].mean()),
                    "mae": float(frame["abs_residual"].mean()),
                    "rmse": float(np.sqrt(np.mean(frame["residual"].to_numpy() ** 2))),
                })
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No regime-by-model summary rows were produced")
    return summary.sort_values(["model", "split", "cluster"]).reset_index(drop=True)


def save_phase6_evaluation(project_root: Path | str = ".") -> dict[str, Path]:
    """Persist scorecards, residual summaries, horizon review, and regime package outputs."""
    root = Path(project_root)
    metrics_dir = root / "outputs" / "metrics"
    report_dir = root / "outputs" / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    scorecard = build_phase6_scorecard(root)
    analysis = build_phase6_residual_analysis(root)
    horizon_review = build_phase6_horizon_review(root)
    regime_analysis = build_phase6_regime_analysis(root)
    regime_summary = build_phase6_regime_error_summary(root)

    paths = {
        "scorecard": metrics_dir / "phase6_scorecard.csv",
        "residual_diagnostics": metrics_dir / "phase6_residual_diagnostics.csv",
        "monthly_errors": report_dir / "phase6_monthly_errors.csv",
        "horizon_errors": report_dir / "phase6_horizon_errors.csv",
        "horizon_review": metrics_dir / "phase6_horizon_review.csv",
        "baseline_comparison": metrics_dir / "phase6_baseline_comparison.csv",
        "selection_review": metrics_dir / "phase6_selection_review.csv",
        "seasonal_errors": report_dir / "phase6_seasonal_errors.csv",
        "large_errors": report_dir / "phase6_large_errors.csv",
        "stability": metrics_dir / "phase6_stability_review.csv",
        "regime_assignments": report_dir / "phase6_regime_assignments.csv",
        "regime_error_summary": metrics_dir / "phase6_regime_error_summary.csv",
        "regime_metadata": metrics_dir / "phase6_regime_metadata.json",
        "regime_model": root / "outputs" / "models" / "phase6_regime_kmeans.joblib",
    }

    scorecard.to_csv(paths["scorecard"], index=False)
    analysis["diagnostics"].to_csv(paths["residual_diagnostics"], index=False)
    analysis["monthly_errors"].to_csv(paths["monthly_errors"], index=False)
    analysis["horizon_errors"].to_csv(paths["horizon_errors"], index=False)
    horizon_review.to_csv(paths["horizon_review"], index=False)
    build_phase6_baseline_comparison(root).to_csv(paths["baseline_comparison"], index=False)
    build_phase6_selection_review(root).to_csv(paths["selection_review"], index=False)
    analysis["seasonal_errors"].to_csv(paths["seasonal_errors"], index=False)
    analysis["large_errors"].to_csv(paths["large_errors"], index=False)
    analysis["stability"].to_csv(paths["stability"], index=False)
    regime_analysis["assignments"].to_csv(paths["regime_assignments"], index=False)
    regime_summary.to_csv(paths["regime_error_summary"], index=False)
    paths["regime_metadata"].write_text(json.dumps({
        "selected_k": regime_analysis["selected_k"],
        "silhouette_score": regime_analysis["silhouette_score"],
        "candidate_scores": regime_analysis["candidate_scores"],
        "feature_columns": regime_analysis["feature_columns"],
        "regime_names": {str(key): value for key, value in regime_analysis["regime_names"].items()},
        "scaler_parameters": regime_analysis["scaler_parameters"],
        "train_shape": regime_analysis["train_shape"],
        "validation_shape": regime_analysis["validation_shape"],
        "test_shape": regime_analysis["test_shape"],
        "random_state": regime_analysis["random_state"],
    }, indent=2))
    import joblib
    paths["regime_model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(regime_analysis["pipeline"], paths["regime_model"])
    return paths


def _save_phase6_visuals(
    project_root: Path,
    inputs: dict[str, Any],
    scorecard: pd.DataFrame,
    analysis: dict[str, pd.DataFrame],
    horizon: pd.DataFrame,
    regime_analysis: dict[str, Any],
) -> dict[str, Path]:
    """Create the final Phase 6 visual set from persisted evaluation artifacts."""
    figure_dir = project_root / "outputs" / "figures" / "phase6"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    colors = {"validation": "#2f6f9f", "test": "#d9480f"}

    for split in ("validation", "test"):
        figure, axis = plt.subplots(figsize=(13, 5.5))
        actual = inputs["artifacts"][f"{split}_y"]
        axis.plot(actual.index, actual.values, color="#1f2937", linewidth=2.6, label="Actual demand")
        for model_name, predictions in inputs["predictions"].items():
            axis.plot(predictions[split].index, predictions[split].values, linewidth=1.5, label=model_name)
        axis.set_title(f"{split.title()} demand: observed values and model forecasts")
        axis.set_ylabel("Residential water demand (m3)")
        axis.legend(ncol=2)
        figure.tight_layout()
        path = figure_dir / f"01_forecast_vs_actual_{split}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths[f"forecast_vs_actual_{split}"] = path

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    for axis, split in zip(axes, ("validation", "test")):
        for model_name in inputs["models"]:
            frame = analysis["monthly_errors"]
            frame = frame[(frame["model"] == model_name) & (frame["split"] == split)]
            axis.plot(pd.to_datetime(frame["date"]), frame["residual"], linewidth=1.3, label=model_name)
        axis.axhline(0, color="#6b7280", linewidth=1)
        axis.set_title(f"{split.title()} residual time series")
        axis.set_ylabel("Actual minus forecast")
        axis.legend(ncol=2)
    figure.tight_layout()
    paths["residual_timeseries"] = figure_dir / "02_residual_timeseries.png"
    figure.savefig(paths["residual_timeseries"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for axis, split in zip(axes, ("validation", "test")):
        for model_name in inputs["models"]:
            values = analysis["monthly_errors"].loc[
                lambda frame: (frame["model"] == model_name) & (frame["split"] == split), "residual"
            ]
            axis.hist(values, bins=10, alpha=0.38, label=model_name)
        axis.axvline(0, color="#1f2937", linewidth=1)
        axis.set_title(f"{split.title()} residual distribution")
        axis.set_ylabel("Months")
        axis.legend(ncol=2)
    axes[-1].set_xlabel("Residual: actual minus forecast")
    figure.tight_layout()
    paths["residual_distributions"] = figure_dir / "03_residual_distributions.png"
    figure.savefig(paths["residual_distributions"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    metric_plot = scorecard.pivot(index="model", columns="split", values="rmse")
    axis = metric_plot.plot(kind="bar", figsize=(11, 5.5), color=[colors[split] for split in ("test", "validation")])
    axis.set_title("RMSE comparison across evaluation windows")
    axis.set_ylabel("RMSE")
    axis.set_xlabel("")
    axis.legend(title="Window")
    figure = axis.get_figure()
    figure.tight_layout()
    paths["metric_comparison"] = figure_dir / "04_metric_comparison.png"
    figure.savefig(paths["metric_comparison"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5.5))
    actual_horizon = horizon.drop_duplicates("horizon_step").sort_values("horizon_step")
    axis.plot(actual_horizon["horizon_step"], actual_horizon["actual_m3"], color="#1f2937", linewidth=2.6, marker="o", label="Actual demand")
    for model_name, frame in horizon.groupby("model"):
        ordered = frame.sort_values("horizon_step")
        axis.plot(ordered["horizon_step"], ordered["forecast_m3"], marker="o", linewidth=1.5, label=model_name)
    axis.set_title("Three-month sequential forecast review")
    axis.set_xlabel("Months ahead")
    axis.set_ylabel("Residential water demand (m3)")
    axis.legend(ncol=2)
    figure.tight_layout()
    paths["horizon_forecasts"] = figure_dir / "05_horizon_forecasts.png"
    figure.savefig(paths["horizon_forecasts"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    assignments = regime_analysis["assignments"].copy()
    assignments["date"] = pd.to_datetime(assignments["date"])
    counts = assignments.groupby(["split", "cluster"]).size().unstack(fill_value=0)
    axis = counts.plot(kind="bar", stacked=True, figsize=(10, 5.5), colormap="tab10")
    axis.set_title("Demand-regime membership by evaluation window")
    axis.set_ylabel("Months")
    axis.set_xlabel("")
    figure = axis.get_figure()
    figure.tight_layout()
    paths["regime_sizes"] = figure_dir / "06_regime_sizes.png"
    figure.savefig(paths["regime_sizes"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    centroids = pd.DataFrame(
        regime_analysis["model"].cluster_centers_,
        columns=regime_analysis["feature_columns"],
    )
    figure, axis = plt.subplots(figsize=(14, 5.5))
    image = axis.imshow(centroids.to_numpy(), aspect="auto", cmap="coolwarm")
    axis.set_title("Standardized demand-regime centroids")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Regime")
    axis.set_xticks(range(len(centroids.columns)))
    axis.set_xticklabels(centroids.columns, rotation=75, ha="right", fontsize=8)
    axis.set_yticks(range(len(centroids)))
    axis.set_yticklabels([f"Regime {index}" for index in centroids.index])
    figure.colorbar(image, ax=axis, label="Standardized centroid value")
    figure.tight_layout()
    paths["regime_centroids"] = figure_dir / "07_regime_centroids.png"
    figure.savefig(paths["regime_centroids"], dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.5))
    axis.scatter(assignments["date"], assignments["cluster"], c=assignments["cluster"], cmap="tab10", s=22)
    for boundary in ("2020-01-01", "2022-01-01"):
        axis.axvline(pd.Timestamp(boundary), color="#6b7280", linestyle="--", linewidth=1)
    axis.set_title("Monthly demand-regime timeline")
    axis.set_ylabel("Regime")
    axis.set_yticks(sorted(assignments["cluster"].unique()))
    axis.set_yticklabels([f"Regime {cluster}" for cluster in sorted(assignments["cluster"].unique())])
    figure.tight_layout()
    paths["regime_timeline"] = figure_dir / "08_regime_timeline.png"
    figure.savefig(paths["regime_timeline"], dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_metadata() -> dict[str, Any]:
    package_names = (
        "pandas", "numpy", "matplotlib", "pytest", "python-dateutil",
        "scikit-learn", "torch", "joblib", "nbclient",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "packages": packages}


def save_phase6_package(project_root: Path | str = ".") -> dict[str, Path]:
    """Generate the complete Phase 6 evaluation package and Phase 7 handoff."""
    root = Path(project_root)
    write_phase5_handoff(root)
    inputs = validate_phase5_inputs(root)
    table_paths = save_phase6_evaluation(root)
    scorecard = build_phase6_scorecard(root)
    analysis = build_phase6_residual_analysis(root)
    regime_analysis = build_phase6_regime_analysis(root)
    horizon = inputs["horizon"]
    visual_paths = _save_phase6_visuals(root, inputs, scorecard, analysis, horizon, regime_analysis)

    centroids = pd.DataFrame(regime_analysis["model"].cluster_centers_, columns=regime_analysis["feature_columns"])
    centroid_path = root / "outputs" / "reports" / "phase6_regime_centroids.csv"
    centroids.to_csv(centroid_path, index_label="cluster")

    package_paths = {**table_paths, **visual_paths, "regime_centroids": centroid_path}
    environment_path = root / "outputs" / "reports" / "phase6_environment.json"
    environment_path.write_text(json.dumps(_environment_metadata(), indent=2))
    package_paths["environment"] = environment_path
    notebook_path = root / "notebooks" / "PHASE_6_EVALUATION_REVIEW.ipynb"
    source_paths = [
        root / "outputs" / "reports" / "phase5_artifact_manifest.json",
        root / "outputs" / "reports" / "phase5_handoff_manifest.json",
        root / "outputs" / "reports" / "phase5_horizon_forecasts.csv",
        root / "outputs" / "reports" / "phase5_quarterly_forecasts.csv",
        root / "data" / "processed" / "features" / "dnh_total_monthly_features_v1.csv",
        root / "data" / "processed" / "features" / "dnh_total_monthly_target_v1.csv",
        root / "data" / "processed" / "splits" / "dnh_total_split_metadata_v1.json",
    ]
    artifact_entries = []
    for label, path in {**package_paths, "notebook": notebook_path}.items():
        if path.exists():
            artifact_entries.append({"name": label, "path": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    source_entries = []
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(f"Phase 6 source artifact is missing: {path}")
        source_entries.append({
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        })
    manifest_path = root / "outputs" / "reports" / "phase6_artifact_manifest.json"
    manifest = {
        "package": "DNH_total Phase 6 evaluation package",
        "status": "READY_FOR_PHASE_7",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_artifacts": [str(path.relative_to(root)) for path in source_paths],
        "artifact_count": len(artifact_entries),
        "artifacts": artifact_entries,
        "source_artifacts": source_entries,
        "regime": {"selected_k": regime_analysis["selected_k"], "silhouette_score": regime_analysis["silhouette_score"], "random_state": 42},
        "rebuild_command": ".\\venv\\Scripts\\python.exe -c \"from src.evaluation.phase6_evaluation import save_phase6_package; save_phase6_package('.')\"",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    selection = build_phase6_selection_review(root)
    baseline = build_phase6_baseline_comparison(root)
    test_leader = selection.loc[selection["split"] == "test", "rmse_leader"].iloc[0]
    stability = analysis["stability"]
    handoff_path = root / "outputs" / "reports" / "phase6_handoff_manifest.json"
    handoff = {
        "package": "DNH_total Phase 6 evaluation handoff",
        "status": "READY_FOR_PHASE_7",
        "source_of_truth": "Saved Phase 5 predictions, diagnostics, horizon forecasts, and chronological artifacts",
        "selected_test_rmse_leader": test_leader,
        "selected_validation_rmse_leader": selection.loc[selection["split"] == "validation", "rmse_leader"].iloc[0],
        "validation_test_rmse_leader_consistent": bool(selection["test_rmse_leader_matches_validation"].iloc[0]),
        "test_set_policy": inputs["handoff"].get("test_set_policy"),
        "regime_summary": {"selected_k": regime_analysis["selected_k"], "silhouette_score": regime_analysis["silhouette_score"]},
        "horizon_summary": "One-month and three-month sequential forecasts and persisted quarterly sums are included; no separate quarterly model is fitted.",
        "residual_caveat": "Residual distributions and regime relationships describe this synthetic benchmark and are not operational confidence intervals or causal findings.",
        "baseline_comparison_rows": int(len(baseline)),
        "stable_model_rankings": int(stability["ranking_stable"].sum()),
        "stable_models": stability.loc[stability["ranking_stable"], "model"].tolist(),
        "unresolved_real_data_questions": [
            "Confirm the real-data target definition and reporting calendar.",
            "Reassess feature availability, missingness, and leakage controls after data replacement.",
            "Revalidate model rankings and demand regimes on real observations.",
        ],
        "phase7_ready": True,
        "artifact_manifest": "outputs/reports/phase6_artifact_manifest.json",
        "rebuild_command": manifest["rebuild_command"],
    }
    handoff_path.write_text(json.dumps(handoff, indent=2))
    handoff_md_path = root / "outputs" / "reports" / "phase6_handoff_manifest.md"
    handoff_md_path.write_text(
        "# Phase 6 Evaluation Handoff\n\n"
        "Status: READY_FOR_PHASE_7\n\n"
        f"Validation-period RMSE leader: {handoff['selected_validation_rmse_leader']}\n\n"
        f"Test-period RMSE leader: {test_leader}\n\n"
        f"Selected demand regimes: {regime_analysis['selected_k']} (training silhouette score: {regime_analysis['silhouette_score']:.3f})\n\n"
        f"Stable model rankings across validation and test: {', '.join(handoff['stable_models']) or 'None'}\n\n"
        "The package contains reproducible scorecards, residual and uncertainty summaries, sequential horizon review, demand-regime analysis, and final evaluation visuals. Results describe a synthetic benchmark and must be reassessed after real-data replacement.\n\n"
        "Open questions for real-data replacement:\n\n"
        "- Confirm the real-data target definition and reporting calendar.\n"
        "- Reassess feature availability, missingness, and leakage controls.\n"
        "- Revalidate model rankings and demand regimes on real observations.\n\n"
        f"Rebuild command: `{manifest['rebuild_command']}`\n"
    )
    for label, path in (("handoff_json", handoff_path), ("handoff_markdown", handoff_md_path)):
        manifest["artifacts"].append({
            "name": label,
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        })
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "artifact_manifest": manifest_path,
        "handoff_json": handoff_path,
        "handoff_markdown": handoff_md_path,
        **package_paths,
    }