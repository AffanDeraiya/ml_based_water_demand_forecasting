import json
from pathlib import Path

from src.evaluation.phase5_handoff import (
    build_phase5_artifact_manifest,
    build_phase5_handoff,
    write_phase5_handoff,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase5_artifact_manifest_checksums_required_package():
    manifest = build_phase5_artifact_manifest(PROJECT_ROOT)

    assert manifest["status"] == "READY_FOR_HANDOFF"
    assert manifest["missing_artifacts"] == []
    assert manifest["artifact_count"] >= 30
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_phase5_handoff_reports_all_models_and_next_phase():
    handoff = build_phase5_handoff(PROJECT_ROOT)

    assert handoff["status"] == "READY_FOR_PHASE_6"
    assert set(handoff["included_models"]) == {"Seasonal naive", "Random Forest", "ANN", "LSTM"}
    assert handoff["excluded_models"] == []
    assert len(handoff["best_models_by_rmse"]) == 2
    assert handoff["next_phase"].startswith("Phase 6")


def test_phase5_handoff_reports_are_written():
    paths = write_phase5_handoff(PROJECT_ROOT)

    assert all(path.exists() for path in paths.values())
    assert json.loads(paths["artifact_manifest"].read_text())["status"] == "READY_FOR_HANDOFF"
    assert json.loads(paths["handoff_json"].read_text())["status"] == "READY_FOR_PHASE_6"