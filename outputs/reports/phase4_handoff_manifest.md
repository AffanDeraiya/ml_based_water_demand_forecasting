# DNH Total Forecasting Preparation Handoff

Status: `READY_FOR_NEXT_PHASE`

This package contains the validated source data, analysis findings, feature decisions, shared matrix, chronological partitions, and reproducibility metadata needed to begin model work.

## Artifact status

- [x] `data/synthetic/raw/synthetic_dnh_total_monthly_v2.csv`
- [x] `data/synthetic/metadata/synthetic_dnh_total_monthly_v2_metadata.json`
- [x] `outputs/reports/dnh_total_eda_summary.json`
- [x] `outputs/reports/dnh_total_data_quality_and_realism.md`
- [x] `outputs/reports/feature_availability_audit.json`
- [x] `outputs/figures/eda/01_target_demand_trend.png`
- [x] `outputs/figures/eda/02_weather_monthly_profiles.png`
- [x] `outputs/figures/eda/03_monsoon_dry_comparison.png`
- [x] `outputs/figures/eda/04_demographics_and_system_state.png`
- [x] `outputs/figures/eda/05_driver_correlations.png`
- [x] `outputs/figures/eda/06_lagged_driver_relationships.png`
- [x] `outputs/figures/feature_audit/01_features_by_group.png`
- [x] `outputs/figures/feature_audit/02_warmup_missingness.png`
- [x] `outputs/figures/feature_audit/03_approved_feature_relationships.png`
- [x] `outputs/figures/matrix_splits/01_chronological_split_timeline.png`
- [x] `data/processed/splits/dnh_total_split_metadata_v1.json`
- [x] `docs/DATA_CONTRACT.md`
- [x] `docs/FEATURE_ENGINEERING_SPECIFICATION.md`
- [x] `docs/FEATURE_MATRIX_AND_SPLIT_SPECIFICATION.md`
- [x] `notebooks/PHASE_4_TASK_4_3_EDA.ipynb`
- [x] `notebooks/FEATURE_AVAILABILITY_AND_INCLUSION_REVIEW.ipynb`
- [x] `notebooks/SHARED_FEATURE_MATRIX_AND_TIME_SPLITS.ipynb`
- [x] `notebooks/PREPARATION_HANDOFF_REVIEW.ipynb`
- [x] `src/features/phase4_data_loader.py`
- [x] `src/features/phase4_feature_engineering.py`
- [x] `src/features/phase4_dataset_preparation.py`
- [x] `src/features/phase4_handoff.py`
- [x] `tests/test_phase4_loading_validation.py`
- [x] `tests/test_phase4_feature_engineering.py`
- [x] `tests/test_phase4_dataset_preparation.py`
- [x] `data/processed/features/dnh_total_monthly_features_v1.csv`
- [x] `data/processed/features/dnh_total_monthly_target_v1.csv`
- [x] `data/processed/splits/train_features_v1.csv`
- [x] `data/processed/splits/train_target_v1.csv`
- [x] `data/processed/splits/train_row_ids_v1.json`
- [x] `data/processed/splits/validation_features_v1.csv`
- [x] `data/processed/splits/validation_target_v1.csv`
- [x] `data/processed/splits/validation_row_ids_v1.json`
- [x] `data/processed/splits/test_features_v1.csv`
- [x] `data/processed/splits/test_target_v1.csv`
- [x] `data/processed/splits/test_row_ids_v1.json`

## Limitations

- The data is synthetic and does not represent observed DNH demand.
- Model fitting and final test evaluation have not started.
