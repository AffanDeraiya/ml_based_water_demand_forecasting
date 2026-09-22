"""Build reproducible Phase 5 artifact and handoff packages."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE5_REQUIRED_ARTIFACTS = (
    "requirements.txt",
    "docs/PHASE_5_FORECASTING_MODELS.md",
    "data/processed/features/dnh_total_monthly_features_v1.csv",
    "data/processed/features/dnh_total_monthly_target_v1.csv",
    "data/processed/splits/dnh_total_split_metadata_v1.json",
    "src/evaluation/forecasting_evaluation.py",
    "src/evaluation/model_comparison.py",
    "src/evaluation/phase5_handoff.py",
    "scripts/package_phase5.py",
    "src/models/baselines.py",
    "src/models/random_forest.py",
    "src/models/ann.py",
    "src/models/lstm.py",
    "tests/test_phase5_evaluation_and_baseline.py",
    "tests/test_phase5_random_forest.py",
    "tests/test_phase5_ann.py",
    "tests/test_phase5_lstm.py",
    "tests/test_phase5_model_comparison.py",
    "notebooks/SEASONAL_NAIVE_BASELINE_REVIEW.ipynb",
    "notebooks/RANDOM_FOREST_FORECAST_REVIEW.ipynb",
    "notebooks/ANN_FORECAST_REVIEW.ipynb",
    "notebooks/LSTM_FORECAST_REVIEW.ipynb",
    "notebooks/PHASE_5_MODEL_COMPARISON_REVIEW.ipynb",
    "notebooks/PHASE_5_HORIZON_FORECAST_REVIEW.ipynb",
    "src/evaluation/horizon_forecasting.py",
    "tests/test_phase5_horizon_forecasting.py",
)

PHASE5_MODEL_ARTIFACTS = (
    "outputs/models/seasonal_naive_baseline.joblib",
    "outputs/models/random_forest_forecaster.joblib",
    "outputs/models/ann_forecaster.joblib",
    "outputs/models/lstm_forecaster.pt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_entry(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    entry: dict[str, Any] = {
        "path": relative_path,
        "exists": path.is_file(),
        "kind": path.suffix.lstrip(".") or "file",
    }
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = _sha256(path)
    return entry


def phase5_artifact_paths(project_root: Path | str = ".") -> list[str]:
    """Return the complete source, model, scorecard, report, and plot package."""
    root = Path(project_root)
    paths = list(PHASE5_REQUIRED_ARTIFACTS) + list(PHASE5_MODEL_ARTIFACTS)
    for directory, patterns in {
        "outputs/metrics": ("*.json", "*.csv"),
        "outputs/reports": ("*forecaster*", "*baseline*", "phase5*"),
        "outputs/figures/model_baselines": ("*.png",),
        "outputs/figures/model_random_forest": ("*.png",),
        "outputs/figures/model_ann": ("*.png",),
        "outputs/figures/model_lstm": ("*.png",),
        "outputs/figures/model_comparison": ("*.png",),
    }.items():
        folder = root / directory
        for pattern in patterns:
            for path in folder.glob(pattern):
                relative_path = str(path.relative_to(root)).replace("\\", "/")
                if directory == "outputs/metrics" and Path(relative_path).name.startswith("phase6_"):
                    continue
                paths.append(relative_path)
    manifest_path = "outputs/reports/phase5_artifact_manifest.json"
    return sorted(path for path in set(paths) if path != manifest_path)


def build_phase5_artifact_manifest(project_root: Path | str = ".") -> dict[str, Any]:
    """Build a checksummed manifest for all reproducible Phase 5 artifacts."""
    root = Path(project_root)
    artifacts = [_artifact_entry(root, path) for path in phase5_artifact_paths(root)]
    missing = [entry["path"] for entry in artifacts if not entry["exists"]]
    return {
        "package": "DNH_total Phase 5 forecasting model package",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_HANDOFF" if not missing else "INCOMPLETE",
        "artifact_count": len(artifacts),
        "missing_artifacts": missing,
        "artifacts": artifacts,
        "source_of_truth": {
            "feature_matrix": "data/processed/features/dnh_total_monthly_features_v1.csv",
            "target": "data/processed/features/dnh_total_monthly_target_v1.csv",
            "split_metadata": "data/processed/splits/dnh_total_split_metadata_v1.json",
        },
        "quarterly_forecast_artifact": "outputs/reports/phase5_quarterly_forecasts.csv",
        "chronological_splits": {"train_months": 108, "validation_months": 24, "test_months": 36},
        "regeneration_command": ".\\venv\\Scripts\\python.exe scripts\\package_phase5.py",
        "test_command": ".\\venv\\Scripts\\python.exe -m pytest -q",
        "checksum_scope": "The manifest is excluded from its own artifact list because a file cannot contain its final checksum before it is written.",
    }


def build_phase5_handoff(project_root: Path | str = ".") -> dict[str, Any]:
    """Build the final Phase 5 handoff summary from saved comparison artifacts."""
    root = Path(project_root)
    manifest = build_phase5_artifact_manifest(root)
    comparison_path = root / "outputs" / "metrics" / "phase5_model_comparison_metrics.csv"
    coverage_path = root / "outputs" / "metrics" / "phase5_model_comparison_coverage.json"
    metrics = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    return {
        "package": "DNH_total Phase 5 forecasting handoff",
        "status": "READY_FOR_PHASE_6" if manifest["status"] == "READY_FOR_HANDOFF" and not coverage.get("excluded_models") else "INCOMPLETE",
        "phase5_tasks": {f"5.{task}": "complete" for task in range(1, 9)},
        "source_of_truth": manifest["source_of_truth"],
        "chronological_splits": manifest["chronological_splits"],
        "included_models": coverage.get("included_models", []),
        "excluded_models": coverage.get("excluded_models", []),
        "test_set_policy": "Test predictions were generated after validation-based selection and were not used for tuning.",
        "best_models_by_rmse": (
            metrics.sort_values(["split", "rmse"]).groupby("split", as_index=False).first()[["split", "model", "rmse"]].to_dict("records")
            if not metrics.empty else []
        ),
        "limitations": [
            "All results are based on synthetic data and are not operational evidence.",
            "The test period was used only for final comparison after validation-based selection.",
        ],
        "next_phase": "Phase 6 evaluation, residual review, and demand-regime analysis",
        "artifact_manifest": "outputs/reports/phase5_artifact_manifest.json",
    }


def write_phase5_handoff(project_root: Path | str = ".") -> dict[str, Path]:
    """Write JSON and Markdown Phase 5 manifest and handoff reports."""
    root = Path(project_root)
    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_phase5_artifact_manifest(root)
    handoff = build_phase5_handoff(root)
    manifest_path = report_dir / "phase5_artifact_manifest.json"
    handoff_path = report_dir / "phase5_handoff_manifest.json"
    markdown_path = report_dir / "phase5_handoff_manifest.md"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    handoff_path.write_text(json.dumps(handoff, indent=2))
    lines = [
        "# DNH Total Phase 5 Forecasting Handoff",
        "",
        f"Status: `{handoff['status']}`",
        "",
        "## Completed model package",
        "",
        *[f"- {model}" for model in handoff["included_models"]],
        "",
        "## Best model by split",
        "",
        *[f"- {item['split']}: {item['model']} (RMSE {item['rmse']:.3f})" for item in handoff["best_models_by_rmse"]],
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in handoff["limitations"]],
        "",
        f"Artifact manifest: `{handoff['artifact_manifest']}`",
        "",
        f"Next phase: {handoff['next_phase']}",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    return {"artifact_manifest": manifest_path, "handoff_json": handoff_path, "handoff_markdown": markdown_path}