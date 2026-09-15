from pathlib import Path
import json

from src.evaluation.phase6_evaluation import (
    build_phase6_baseline_comparison,
    build_phase6_horizon_review,
    build_phase6_regime_analysis,
    build_phase6_regime_error_summary,
    build_phase6_residual_analysis,
    build_phase6_scorecard,
    build_phase6_selection_review,
    save_phase6_package,
    save_phase6_evaluation,
    validate_phase5_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase5_inputs_are_ready_for_phase6():
    inputs = validate_phase5_inputs(PROJECT_ROOT)

    assert inputs["models"] == ["ANN", "LSTM", "Random Forest", "Seasonal naive"]
    assert len(inputs["horizon"]) == 12
    assert inputs["integrity_issues"] == []
    assert all(inputs["integrity_checks"].values())


def test_phase6_scorecard_recomputes_all_models_and_splits():
    scorecard = build_phase6_scorecard(PROJECT_ROOT)

    assert len(scorecard) == 8
    assert set(scorecard["split"]) == {"validation", "test"}
    assert scorecard.groupby("split").size().to_dict() == {"validation": 4, "test": 4}
    assert scorecard.groupby("split").first().index.tolist() == ["test", "validation"]


def test_phase6_residual_and_horizon_outputs_are_persisted():
    analysis = build_phase6_residual_analysis(PROJECT_ROOT)
    paths = save_phase6_evaluation(PROJECT_ROOT)

    assert len(analysis["diagnostics"]) == 8
    assert len(analysis["horizon_errors"]) == 12
    assert not analysis["seasonal_errors"].empty
    assert len(analysis["large_errors"]) == 24
    assert set(analysis["stability"]["model"]) == {"ANN", "LSTM", "Random Forest", "Seasonal naive"}
    assert all(path.exists() for path in paths.values())


def test_phase6_selection_and_baseline_reviews_are_recomputed():
    selection = build_phase6_selection_review(PROJECT_ROOT)
    baseline = build_phase6_baseline_comparison(PROJECT_ROOT)

    assert set(selection["split"]) == {"validation", "test"}
    assert selection["rmse_leader"].notna().all()
    assert len(baseline) == 6
    assert set(baseline["model"]) == {"ANN", "LSTM", "Random Forest"}
    assert baseline[["mae_delta_vs_baseline", "rmse_delta_vs_baseline"]].notna().all().all()


def test_phase6_final_package_contains_visuals_and_phase7_handoff():
    paths = save_phase6_package(PROJECT_ROOT)
    manifest = json.loads(paths["artifact_manifest"].read_text())
    handoff = json.loads(paths["handoff_json"].read_text())

    assert manifest["status"] == "READY_FOR_PHASE_7"
    assert manifest["artifact_count"] >= 20
    assert handoff["status"] == "READY_FOR_PHASE_7"
    assert handoff["phase7_ready"] is True
    assert paths["handoff_markdown"].exists()
    assert (PROJECT_ROOT / "outputs" / "figures" / "phase6" / "08_regime_timeline.png").exists()
    assert (PROJECT_ROOT / "outputs" / "reports" / "phase6_regime_centroids.csv").exists()


def test_phase6_horizon_review_summarizes_short_horizon_performance():
    summary = build_phase6_horizon_review(PROJECT_ROOT)

    assert set(summary["model"]) == {"ANN", "LSTM", "Random Forest", "Seasonal naive"}
    assert set(summary["horizon_step"]) == {1, 3}
    assert summary["one_month_absolute_error"].notna().all()
    assert summary["three_month_mae"].notna().all()


def test_phase6_regime_analysis_uses_training_only_scaling_and_assigns_all_periods():
    regimes = build_phase6_regime_analysis(PROJECT_ROOT)

    assert regimes["selected_k"] in {2, 3, 4, 5, 6}
    assert len(regimes["assignments"]) == 168
    assert set(regimes["assignments"]["split"]) == {"train", "validation", "test"}
    assert set(regimes["assignments"]["cluster"]) == set(range(regimes["selected_k"]))


def test_phase6_regime_error_summary_is_available_by_model_and_cluster():
    summary = build_phase6_regime_error_summary(PROJECT_ROOT)

    assert set(summary["model"]) == {"ANN", "LSTM", "Random Forest", "Seasonal naive"}
    assert summary["mae"].notna().all()
    assert summary["rmse"].notna().all()