"""Shared model-ready feature matrix and chronological split preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd

from .phase4_feature_engineering import build_leakage_safe_features


DEFAULT_SPLIT_SIZES = {
    "train": 108,
    "validation": 24,
    "test": 36,
}


def prepare_shared_feature_matrix(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the complete shared matrix and keep the target separate."""
    features, target = build_leakage_safe_features(data, drop_incomplete=True)
    if features.empty:
        raise ValueError("Feature matrix cannot be empty")
    if not features.index.equals(target.index):
        raise ValueError("Feature and target indexes must be identical")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature matrix must remain chronological")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Shared feature matrix and target must not contain missing values")
    return features, target


def create_chronological_splits(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    split_sizes: Mapping[str, int] = DEFAULT_SPLIT_SIZES,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Create contiguous train, validation, and test views without shuffling."""
    if not features.index.equals(target.index):
        raise ValueError("Feature and target indexes must be identical")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Features must be chronologically ordered")

    expected_names = ("train", "validation", "test")
    if tuple(split_sizes.keys()) != expected_names:
        raise ValueError("split_sizes must contain train, validation, test in that order")
    if any(int(size) <= 0 for size in split_sizes.values()):
        raise ValueError("All split sizes must be positive")
    if sum(split_sizes.values()) != len(features):
        raise ValueError("Split sizes must sum to the number of feature rows")

    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    start = 0
    for name in expected_names:
        stop = start + int(split_sizes[name])
        splits[name] = (features.iloc[start:stop].copy(), target.iloc[start:stop].copy())
        start = stop
    return splits


def describe_chronological_splits(
    splits: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    """Return counts and date boundaries for each chronological split."""
    rows = []
    for name in ("train", "validation", "test"):
        if name not in splits:
            raise ValueError(f"Missing split: {name}")
        features, target = splits[name]
        if not features.index.equals(target.index):
            raise ValueError(f"Feature and target indexes differ in {name} split")
        rows.append({
            "split": name,
            "rows": len(features),
            "start_date": features.index.min().strftime("%Y-%m-%d"),
            "end_date": features.index.max().strftime("%Y-%m-%d"),
            "feature_count": len(features.columns),
        })
    return pd.DataFrame(rows)


def save_prepared_artifacts(
    features: pd.DataFrame,
    target: pd.Series,
    splits: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    output_dir: Path | str = Path("data/processed"),
) -> Dict[str, Path]:
    """Persist the shared matrix, target, split views, and boundary metadata."""
    output_root = Path(output_dir)
    feature_dir = output_root / "features"
    split_dir = output_root / "splits"
    feature_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    feature_path = feature_dir / "dnh_total_monthly_features_v1.csv"
    target_path = feature_dir / "dnh_total_monthly_target_v1.csv"
    features.to_csv(feature_path, index_label="date")
    target.to_frame().to_csv(target_path, index_label="date")

    split_summary = describe_chronological_splits(splits)
    split_paths: Dict[str, Path] = {}
    split_row_ids: Dict[str, list[str]] = {}
    for name, (split_features, split_target) in splits.items():
        split_feature_path = split_dir / f"{name}_features_v1.csv"
        split_target_path = split_dir / f"{name}_target_v1.csv"
        row_ids_path = split_dir / f"{name}_row_ids_v1.json"
        split_features.to_csv(split_feature_path, index_label="date")
        split_target.to_frame().to_csv(split_target_path, index_label="date")
        row_ids = [date.strftime("%Y-%m-%d") for date in split_features.index]
        row_ids_path.write_text(json.dumps(row_ids, indent=2))
        split_row_ids[name] = row_ids
        split_paths[f"{name}_features"] = split_feature_path
        split_paths[f"{name}_target"] = split_target_path
        split_paths[f"{name}_row_ids"] = row_ids_path

    metadata = {
        "dataset": "DNH_total",
        "feature_count": len(features.columns),
        "feature_columns": list(features.columns),
        "target_column": target.name,
        "source_rows": len(features) + 12,
        "warm_up_rows_removed": 12,
        "complete_rows": len(features),
        "split_summary": split_summary.to_dict(orient="records"),
        "split_row_ids": split_row_ids,
        "split_rationale": "The first 12 months are removed for complete 12-month history; the final 36 complete months remain untouched for testing.",
        "source_feature_specification": "docs/FEATURE_MATRIX_AND_SPLIT_SPECIFICATION.md",
        "chronological": True,
        "random_shuffle": False,
        "future_value_fill": False,
    }
    metadata_path = split_dir / "dnh_total_split_metadata_v1.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "features": feature_path,
        "target": target_path,
        "metadata": metadata_path,
        **split_paths,
    }
