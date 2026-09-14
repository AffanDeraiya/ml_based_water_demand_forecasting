"""Build a traceable handoff manifest for the completed preparation work."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


REQUIRED_HANDOFF_ARTIFACTS = (
    "data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv",
    "data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json",
    "outputs/reports/dnh_total_eda_summary.json",
    "outputs/reports/dnh_total_data_quality_and_realism.md",
    "outputs/reports/feature_availability_audit.json",
    "outputs/figures/eda/01_target_demand_trend.png",
    "outputs/figures/eda/02_weather_monthly_profiles.png",
    "outputs/figures/eda/03_monsoon_dry_comparison.png",
    "outputs/figures/eda/04_demographics_and_system_state.png",
    "outputs/figures/eda/05_driver_correlations.png",
    "outputs/figures/eda/06_lagged_driver_relationships.png",
    "outputs/figures/feature_audit/01_features_by_group.png",
    "outputs/figures/feature_audit/02_warmup_missingness.png",
    "outputs/figures/feature_audit/03_approved_feature_relationships.png",
    "outputs/figures/matrix_splits/01_chronological_split_timeline.png",
    "data/processed/splits/dnh_total_split_metadata_v1.json",
    "docs/DATA_CONTRACT.md",
    "docs/FEATURE_ENGINEERING_SPECIFICATION.md",
    "docs/FEATURE_MATRIX_AND_SPLIT_SPECIFICATION.md",
    "notebooks/PHASE_4_TASK_4_3_EDA.ipynb",
    "notebooks/FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb",
    "notebooks/SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb",
    "notebooks/PREPARATION_HANDOFF_REVIEW.ipynb",
    "src/features/phase4_data_loader.py",
    "src/features/phase4_feature_engineering.py",
    "src/features/phase4_dataset_preparation.py",
    "src/features/phase4_handoff.py",
    "tests/test_phase4_loading_validation.py",
    "tests/test_phase4_feature_engineering.py",
    "tests/test_phase4_dataset_preparation.py",
    "data/processed/features/dnh_total_monthly_features_v1.csv",
    "data/processed/features/dnh_total_monthly_target_v1.csv",
    "data/processed/splits/train_features_v1.csv",
    "data/processed/splits/train_target_v1.csv",
    "data/processed/splits/train_row_ids_v1.json",
    "data/processed/splits/validation_features_v1.csv",
    "data/processed/splits/validation_target_v1.csv",
    "data/processed/splits/validation_row_ids_v1.json",
    "data/processed/splits/test_features_v1.csv",
    "data/processed/splits/test_target_v1.csv",
    "data/processed/splits/test_row_ids_v1.json",
)


def build_phase4_handoff_manifest(project_root: Path | str = Path(".")) -> Dict:
    """Return an existence and purpose manifest for the handoff package."""
    root = Path(project_root)
    artifacts = []
    for relative_path in REQUIRED_HANDOFF_ARTIFACTS:
        path = root / relative_path
        artifacts.append({
            "path": relative_path,
            "exists": path.exists(),
            "kind": path.suffix.lstrip(".") or "directory",
        })

    missing = [item["path"] for item in artifacts if not item["exists"]]
    return {
        "package": "DNH_total monthly forecasting preparation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_NEXT_PHASE" if not missing else "INCOMPLETE",
        "missing_artifacts": missing,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "scope": [
            "validated source data",
            "EDA and realism findings",
            "feature availability decisions",
            "shared feature matrix and target",
            "chronological train validation test partitions",
            "reproducibility metadata and row IDs",
        ],
        "limitations": [
            "The data is synthetic and does not represent observed DNH demand.",
            "Model fitting and final test evaluation have not started.",
        ],
    }


def write_phase4_handoff_manifest(
    project_root: Path | str = Path("."),
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Write JSON and Markdown handoff summaries and return the JSON path."""
    root = Path(project_root)
    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(output_path) if output_path else report_dir / "phase4_handoff_manifest.json"
    manifest = build_phase4_handoff_manifest(root)
    json_path.write_text(json.dumps(manifest, indent=2))

    markdown_path = json_path.with_suffix(".md")
    lines = [
        "# DNH Total Forecasting Preparation Handoff",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "This package contains the validated source data, analysis findings, feature decisions, shared matrix, chronological partitions, and reproducibility metadata needed to begin model work.",
        "",
        "## Artifact status",
        "",
    ]
    lines.extend(
        f"- [{'x' if item['exists'] else ' '}] `{item['path']}`"
        for item in manifest["artifacts"]
    )
    lines.extend([
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in manifest["limitations"]],
    ])
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path
