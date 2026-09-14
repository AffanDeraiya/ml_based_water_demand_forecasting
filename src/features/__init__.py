"""Feature engineering utilities for Phase 4 and later forecasting work."""

from .phase4_data_loader import (
    canonical_dataset_path,
    get_configured_dataset_paths,
    load_dnh_total_dataset,
    validate_dnh_total_dataframe,
)
from .phase4_feature_engineering import (
    build_leakage_safe_features,
    feature_availability,
)
from .phase4_dataset_preparation import (
    create_chronological_splits,
    describe_chronological_splits,
    prepare_shared_feature_matrix,
    save_prepared_artifacts,
)
from .phase4_handoff import (
    build_phase4_handoff_manifest,
    write_phase4_handoff_manifest,
)

__all__ = [
    "canonical_dataset_path",
    "get_configured_dataset_paths",
    "load_dnh_total_dataset",
    "validate_dnh_total_dataframe",
    "build_leakage_safe_features",
    "feature_availability",
    "prepare_shared_feature_matrix",
    "create_chronological_splits",
    "describe_chronological_splits",
    "save_prepared_artifacts",
    "build_phase4_handoff_manifest",
    "write_phase4_handoff_manifest",
]
