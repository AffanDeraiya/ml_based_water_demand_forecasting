# Phase 6 - Evaluation and Demand-Regime Analysis

## Purpose

Phase 6 turns the completed Phase 5 model package into a rigorous evaluation of forecasting behaviour and an exploratory analysis of monthly demand regimes. It uses the saved predictions, metrics, diagnostics, horizon forecasts, and chronological data artifacts from Phase 5.

This phase is concerned with interpretation and evidence, not new model training. It must distinguish model performance on synthetic data from claims that could be made after real-data replacement.

The work in this phase includes:

- validating the complete Phase 5 package before interpretation
- reviewing model performance on the untouched test period
- quantifying error, residual behaviour, and forecast stability
- reviewing one-month and three-month sequential forecasts
- exploring monthly demand regimes with leakage-safe K-Means
- relating regime membership to model errors without turning clustering into a predictive feature by default
- producing a consolidated evaluation package and handoff for Phase 7 reporting

It does not include final narrative reporting or real-data replacement. Those belong to Phase 7 and Phase 8.

---

## Governing rules

The following requirements are non-negotiable:

- Use the Phase 5 artifact manifest and saved predictions as the source of truth.
- Preserve the fixed chronological structure: train 108 months, validation 24 months, test 36 months.
- Do not alter model selection or hyperparameters after inspecting test results.
- Do not use test outcomes to refit models or revise the selected model.
- Report MAE, RMSE, sMAPE, and R-squared where appropriate; do not reduce results to a generic accuracy percentage.
- Treat uncertainty and limitations explicitly, especially because the data is synthetic.
- Use K-Means only for exploratory monthly demand-regime analysis, never geographic clustering.
- Fit scaling and clustering on the training period when regime analysis feeds any downstream interpretation or predictive workflow.
- Keep quarterly interpretation as the sum of three sequential monthly forecasts; do not create a separate quarterly model.
- Preserve all source artifacts, seeds, configurations, and analysis outputs for reproducibility.

---

## Phase 6 objectives

1. Confirm that the saved Phase 5 package is complete, internally consistent, and reloadable.
2. Produce a final model scorecard using the untouched test period.
3. Quantify residual bias, spread, outliers, seasonal error patterns, and horizon degradation.
4. Review the one-month and three-month sequential forecasts for all completed models.
5. Explore monthly demand regimes using leakage-safe K-Means.
6. Examine whether model errors differ across exploratory demand regimes.
7. Produce a consolidated evaluation notebook, report artifacts, and a Phase 7 handoff.

## Expected current status

Validated end-to-end: Tasks 6.1-6.8 are implemented, packaged, and pass the relevant evaluation checks. Phase 6 is ready for the next reporting phase.

- [x] Task 6.1 - Validate the Phase 5 package and evaluation inputs
- [x] Task 6.2 - Produce the final model scorecard and test-period review
- [x] Task 6.3 - Analyse residuals, uncertainty, and forecast stability
- [x] Task 6.4 - Review one-month and three-month sequential forecasts
- [x] Task 6.5 - Build a leakage-safe monthly demand-regime analysis
- [x] Task 6.6 - Relate model errors to demand regimes
- [x] Task 6.7 - Produce evaluation visuals and the consolidated evaluation package
- [x] Task 6.8 - Prepare the Phase 6 handoff for Phase 7

---

## Task-by-task checklist

### Task 6.1 - Validate the Phase 5 package and evaluation inputs

- [x] Load `outputs/reports/phase5_artifact_manifest.json` and verify `READY_FOR_HANDOFF` status.
- [x] Load `outputs/reports/phase5_handoff_manifest.json` and verify all four models are included.
- [x] Verify that every model has saved validation/test predictions, metrics, configuration, and diagnostic artifacts.
- [x] Verify that prediction indexes align with the accepted validation and test target indexes.
- [x] Verify the shared feature matrix, target, split metadata, and row counts remain unchanged.
- [x] Recompute selected scorecard metrics from persisted predictions.
- [x] Record any missing, stale, or inconsistent artifacts before interpretation begins.

Definition of done:

- the Phase 5 package can be loaded without retraining
- all model predictions and target windows align exactly
- evaluation begins from a verified and reproducible input package

---

### Task 6.2 - Produce the final model scorecard and test-period review

- [x] Rebuild the four-model validation and test scorecard from persisted predictions.
- [x] Report MAE, RMSE, sMAPE, MAPE, and R-squared where meaningful.
- [x] Rank models separately on validation and test RMSE and MAE.
- [x] Identify the selected test-period leader without changing model selection after test inspection.
- [x] Compare every advanced model against the seasonal-naive baseline.
- [x] Report absolute and relative differences against the baseline with clear directionality.
- [x] Record the synthetic-data status beside every interpretation.

Definition of done:

- one authoritative scorecard exists for validation and test performance
- the best completed model is identifiable by objective metrics
- no unsupported operational or real-world accuracy claim is made

---

### Task 6.3 - Analyse residuals, uncertainty, and forecast stability

- [x] Load per-model residual files and residual-diagnostic JSON artifacts.
- [x] Quantify residual mean, standard deviation, MAE, RMSE, and maximum absolute error.
- [x] Check for systematic overprediction or underprediction by split.
- [x] Examine error spread across calendar months and seasons.
- [x] Identify large-error months and compare their behaviour across models.
- [x] Summarize uncertainty using error distributions, quantiles, and limitations rather than unsupported confidence claims.
- [x] Compare validation residual behaviour with test residual behaviour to identify possible instability.
- [x] Record whether the test-period ranking is consistent with validation selection.

Definition of done:

- residual behaviour is described quantitatively and visually
- uncertainty language is appropriate for a synthetic benchmark
- model instability or error concentration is explicitly documented

Implementation note: the reusable evaluator writes `outputs/metrics/phase6_residual_diagnostics.csv`, `outputs/reports/phase6_monthly_errors.csv`, `outputs/reports/phase6_seasonal_errors.csv`, `outputs/reports/phase6_large_errors.csv`, `outputs/reports/phase6_horizon_errors.csv`, and `outputs/metrics/phase6_stability_review.csv`. Selection and baseline comparisons are written to `outputs/metrics/phase6_selection_review.csv` and `outputs/metrics/phase6_baseline_comparison.csv`.

---

### Task 6.4 - Review one-month and three-month sequential forecasts

- [x] Load `outputs/reports/phase5_horizon_forecasts.csv` and `outputs/metrics/phase5_horizon_metrics.json`.
- [x] Verify that all four models have one-step and three-step forecasts from the same test origin.
- [x] Compare one-month absolute error with three-month MAE and RMSE.
- [x] Check whether forecast ranking changes as the horizon increases.
- [x] Inspect recursive error accumulation and divergence from observed demand.
- [x] Confirm that horizon features use recursive predicted demand lags and rolling values.
- [x] Produce a clear visual showing actual demand against each short-horizon forecast.

Definition of done:

- the short-horizon workflow is validated independently of the long test scorecard
- horizon degradation and recursive uncertainty are documented
- quarterly interpretation remains the sum of sequential monthly forecasts

---

### Task 6.5 - Build a leakage-safe monthly demand-regime analysis

- [x] Define regime inputs using monthly demand behaviour and approved explanatory variables.
- [x] Decide whether the clustering input is demand level, seasonal demand profile, growth, variability, or a documented combination.
- [x] Fit feature scaling on the training period only.
- [x] Select a small documented range of candidate cluster counts.
- [x] Use an objective selection aid such as silhouette score, inertia, and interpretability.
- [x] Fit the chosen K-Means model on training-period regime inputs.
- [x] Assign validation and test months to the fitted regime space without refitting on those periods.
- [x] Confirm that clusters describe monthly demand regimes, not zones or geography.
- [x] Save cluster assignments, centroids, scaling metadata, selected cluster count, and random seed.

Definition of done:

- a reproducible exploratory regime model exists
- the K-Means workflow respects the chronological and scaling rules
- regime names are descriptive and based on observed demand characteristics

---

### Task 6.6 - Relate model errors to demand regimes

- [x] Join model residuals with regime assignments by month.
- [x] Summarize MAE, RMSE, bias, and error spread by model and regime.
- [x] Identify regimes where the seasonal baseline is strong or weak.
- [x] Identify regimes where ANN, LSTM, or Random Forest provide meaningful improvement.
- [x] Check whether regime-specific conclusions are stable between validation and test periods.
- [x] Treat regime relationships as exploratory unless independently supported by real data.
- [x] Avoid using regime labels as model features unless a later approved experiment defines that workflow explicitly.

Definition of done:

- regime-specific model behaviour is summarized without causal overclaiming
- the analysis explains where models succeed or fail
- the exploratory role of K-Means remains clear

---

### Task 6.7 - Produce evaluation visuals and the consolidated evaluation package

- [x] Produce the final forecast-versus-actual comparison plot.
- [x] Produce residual time-series and residual-distribution plots.
- [x] Produce metric comparison plots for validation and test periods.
- [x] Produce one-month and three-month horizon plots.
- [x] Produce regime-size, regime-centroid, and regime-timeline plots.
- [x] Save consolidated scorecards, residual summaries, horizon summaries, and regime summaries.
- [x] Create a presentation-ready evaluation notebook without internal implementation-plan wording.
- [x] Record the exact source artifacts and commands used to regenerate the evaluation package.

Definition of done:

- Phase 6 outputs are easy to inspect without reading implementation code
- all plots and tables trace back to saved Phase 5 artifacts
- the evaluation package is ready for reporting work

---

### Task 6.8 - Prepare the Phase 6 handoff for Phase 7

- [x] Confirm all Phase 6 outputs are present and checksummed.
- [x] Confirm that no model was refit or retuned using test results.
- [x] Document final model ranking, residual caveats, regime findings, and horizon findings.
- [x] Separate synthetic-data evidence from claims that require real data.
- [x] Record unresolved questions for real-data replacement.
- [x] Write a Phase 6 JSON and Markdown handoff manifest.
- [x] Mark the package ready for Phase 7 reporting only after all required artifacts are present.

Definition of done:

- Phase 6 produces a clean, reviewable evaluation package
- Phase 7 can begin without repeating model evaluation
- limitations and unresolved real-data questions are visible in the handoff

---

## Deliverables expected at the end of Phase 6

- final validation and test scorecard for all four models
- baseline-relative improvement summary
- residual diagnostics and uncertainty/error summaries
- one-month and three-month sequential forecast review
- leakage-safe K-Means regime model and assignments
- regime-specific model error summaries
- forecast, residual, horizon, and regime visualizations
- presentation-ready evaluation notebook
- checksummed Phase 6 artifact manifest
- Phase 6 handoff package for Phase 7 reporting

---

## Exit criteria for Phase 6

Phase 6 is complete only when all of the following are true:

- [x] the Phase 5 package is verified and used as the evaluation source of truth
- [x] all four models are compared on the same validation and test windows
- [x] baseline-relative performance is reported with appropriate limitations
- [x] residual and uncertainty behaviour is quantified
- [x] one-month and three-month sequential forecasts are reviewed
- [x] K-Means regimes are fitted and interpreted under the stated leakage controls
- [x] regime-specific model errors are summarized without causal overclaiming
- [x] all evaluation visuals and tables are saved reproducibly
- [x] the package is ready for Phase 7 reporting

---

## Phase 7 handoff

Once the checks above are complete, Phase 7 can begin with:

- final narrative reporting
- methodology and results documentation
- synthetic-data limitations and real-data replacement guidance
- reproducibility runbook finalization
- preparation of the project’s final reader-facing outputs

---

## Notes for implementation discipline

- Keep evaluation separate from model training and model selection.
- Do not silently refit models after inspecting test results.
- Do not treat K-Means regimes as geographic clusters.
- Do not describe synthetic benchmark performance as operational accuracy.
- Prefer saved tables, plots, and manifests over notebook-only conclusions.
- Preserve the distinction between monthly forecasts and quarterly sums of sequential monthly forecasts.
