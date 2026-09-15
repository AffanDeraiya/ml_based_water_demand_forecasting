# Phase 5 — Forecasting Models and Baseline Evaluation

## Purpose

Phase 5 turns the prepared DNH-total feature matrix and chronological splits into a reproducible model-building and evaluation workflow. This phase focuses on forecasting performance under a strict time-series protocol, not on ad hoc experimentation.

The work in this phase is intentionally limited to:
- loading the shared feature matrix and target from the saved Phase 4 artifacts
- establishing a common evaluation harness and metrics framework
- building a seasonal-naive baseline
- fitting and tuning a Random Forest model
- fitting and tuning an ANN model
- fitting and tuning an LSTM model
- comparing models on the untouched test period
- saving model outputs, metrics, and diagnostics for the next phase

It does not include final reporting or real-data replacement. Those belong to Phase 6 and Phase 7.

---

## Governing rules

The following requirements are non-negotiable:

- Use the shared feature matrix and target created in Phase 4, not newly re-derived features.
- Use the fixed chronological split defined in Phase 4: train 108 months, validation 24 months, test 36 months.
- No random train/test split for time-series forecasting.
- Fit all preprocessing, scaling, and selection steps only on the training data.
- Keep the test period untouched until final model comparison.
- Do not build a separate quarterly model; quarterly demand remains the sum of sequential monthly forecasts.
- Report metrics in a way that is appropriate for continuous time-series demand forecasting.
- Save all model settings, metrics, and artifact paths for reproducibility.

---

## Phase 5 objectives

1. Build a single reusable evaluation harness for all models.
2. Establish a seasonal-naive baseline against which all advanced models are compared.
3. Train and tune a Random Forest forecaster on the shared feature matrix.
4. Train and tune an ANN forecaster on the same shared matrix and training-only scaling.
5. Train and tune an LSTM forecaster using the correct sequence structure and training-only scaling.
6. Compare model performance on the untouched test period using MAE, RMSE, sMAPE, and R² where appropriate.
7. Save model artifacts, scorecards, and residual/forecast diagnostic outputs for the next phase.

## Current implementation status

- [x] Task 5.1 — Build the model-evaluation framework and data-loading contract
- [x] Task 5.2 — Implement the seasonal-naive baseline
- [x] Task 5.3 — Train and tune the Random Forest baseline model
- [x] Task 5.4 — Train and tune the ANN model
- [x] Task 5.5 — Train and tune the LSTM model
- [x] Task 5.6 — Compare model performance and generate diagnostic plots
- [x] Task 5.7 — Save reproducible model outputs and summary artifacts
- [x] Task 5.8 — Prepare the Phase 5 handoff for Phase 6

### Horizon forecast deliverable

- [x] Generate one-month-ahead forecasts from the first held-out test origin.
- [x] Generate three-month sequential forecasts using recursive predicted demand lags and rolling features.
- [x] Persist four-model horizon forecasts and horizon error summaries.

Implementation note: horizon outputs are written to `outputs/reports/phase5_horizon_forecasts.csv` and `outputs/metrics/phase5_horizon_metrics.json` by the reusable horizon forecasting module.

---

## Task-by-task checklist

### Task 5.1 — Build the model-evaluation framework and data-loading contract
- [x] Define the standard model-loading path for the Phase 4 feature matrix and target.
- [x] Confirm the exact train / validation / test row sets and date ranges before any modelling begins.
- [x] Build a reusable metrics helper for MAE, RMSE, sMAPE, and R² where appropriate.
- [x] Define a single function that fits a model, predicts on a split, and reports metrics.
- [x] Save model configuration metadata alongside outputs.
- [x] Ensure the framework accepts a model object, train set, validation set, test set, and metric names consistently.

Definition of done:
- one reusable training/evaluation pipeline exists for all forecast models
- all models consume the same feature matrix and the same split definitions
- metrics and predictions can be reproduced from saved artifacts

---

### Task 5.2 — Implement the seasonal-naive baseline
- [x] Define the baseline as a naive seasonal forecast using historical month-of-year behaviour.
- [x] Use the validation period to confirm that the baseline is sensible and stable before test-time comparison.
- [x] Compute baseline predictions for the validation and test windows.
- [x] Record baseline metrics and residual diagnostics.
- [x] Store the baseline model configuration and its predictions in a reproducible format.

Implementation note: the seasonal-naive model, validation/test predictions, metrics, residual diagnostics, model configuration, and presentation notebook are complete.

Definition of done:
- a seasonal-naive baseline exists and is reproducible
- baseline predictions are generated only from historical recurring seasonal structure
- the baseline is documented as the comparator for every advanced model

---

### Task 5.3 — Train and tune the Random Forest model
- [x] Load the shared feature matrix and target from Phase 4 outputs.
- [x] Define the train/validation split in the same order used for all other models.
- [x] Train a Random Forest regressor on the training set only.
- [x] Tune at least the key hyperparameters using the validation period, with a documented selection rule.
- [x] Check feature importance and rank-order stability for interpretation.
- [x] Generate validation and test predictions and compute summary metrics.
- [x] Save trained model file(s), feature names, and configuration metadata.

Definition of done:
- a Random Forest model has been trained with training-only fitting discipline
- validation-driven tuning is complete and documented
- the final model predicts on the untouched test period and the results are saved

---

### Task 5.4 — Train and tune the ANN model
- [x] Prepare the training inputs for an ANN using the shared feature matrix.
- [x] Apply any required scaling or normalization using training statistics only.
- [x] Define a simple network architecture appropriate for tabular monthly demand forecasting.
- [x] Tune hyperparameters on the validation set, including at least architecture and learning-rate choices.
- [x] Train the network with an explicit validation-monitoring loop.
- [x] Save model weights/configuration and evaluation outputs.
- [x] Ensure that the ANN is not exposed to future leakage through feature construction or scaling.

Implementation note: the ANN uses training-only scaling, chronological validation monitoring with patience-based stopping, and preserves the best validation state before held-out test evaluation.

Definition of done:
- an ANN model has been trained and validated under chronological rules
- all transformations are fit only on training data
- test-period performance is captured and stored for comparison

---

### Task 5.5 — Train and tune the LSTM model
- [x] Define the sequence structure required by the LSTM using the monthly time index and lagged features.
- [x] Prepare a supervised sequence dataset that respects the chronology and avoids target leakage.
- [x] Apply any sequence scaling using training-only statistics.
- [x] Define the base architecture and tune the main hyperparameters on the validation period.
- [x] Train and monitor the model for overfitting or instability.
- [x] Save the trained sequence model, preprocessing metadata, and epoch diagnostics.
- [x] Confirm that the LSTM is operating on the same underlying task and split definition as the other models.

Definition of done:
- an LSTM model has been trained on the correct monthly sequence representation
- no leakage is introduced through the sequence-building logic
- the model is evaluated on the untouched test set and saved for reporting

---

### Task 5.6 — Compare model performance and generate diagnostic plots
- [x] Compare all completed models on the same validation set before finalizing interpretation.
- [x] Compare all completed models on the untouched test set using the agreed metrics.
- [x] Produce forecast-vs-actual plots for each completed model.
- [x] Produce residual plots and error summaries where useful.
- [x] Plot the seasonal benchmark against the stronger models to show relative improvement.
- [x] Save a summary table of metrics by model and split.
- [x] Record any observed caveats, such as synthetic-data limitations or model instability.

Implementation note: the comparison now includes the seasonal naive, Random Forest, ANN, and LSTM packages on the same validation and test windows.

Definition of done:
- model comparison is reproducible and easy to inspect
- the final test-period results are summarized in a single comparison artifact
- the strongest model is identifiable from objective performance metrics

---

### Task 5.7 — Save reproducible model outputs and summary artifacts
- [x] Save trained model objects or model weights for each model.
- [x] Save per-model predictions for validation and test periods.
- [x] Save metrics tables explicitly including model name, split, MAE, RMSE, sMAPE, and R² when applicable.
- [x] Save configuration metadata, including feature set version, split metadata version, and model hyperparameters.
- [x] Save an artifact manifest listing all generated outputs.
- [x] Record the exact command or script sequence used to regenerate the model package.

Implementation note: the package is generated by `scripts/package_phase5.py` and includes checksummed source, model, prediction, metric, diagnostic, figure, notebook, and test artifacts.

Definition of done:
- all Phase 5 outputs are captured in a reproducible, discoverable form
- models and predictions can be reloaded without re-running the entire pipeline
- a model package is ready for Phase 6 review and reporting

---

### Task 5.8 — Prepare the Phase 5 handoff for Phase 6
- [x] Confirm the baseline and advanced-model results are consistent with the accepted feature and split definitions.
- [x] Confirm all artifacts and results are saved in the expected outputs folders.
- [x] Check that the final test set remains untouched and is used only for final evaluation.
- [x] Document any modelling surprises, failure modes, or synthetic-data caveats.
- [x] Prepare the model package for Phase 6 evaluation and demand-regime analysis.

Implementation note: the Phase 5 handoff is written to `outputs/reports/phase5_handoff_manifest.json` and `.md` with status `READY_FOR_PHASE_6`.

Definition of done:
- Phase 5 produces a clean, reviewable model package
- Phase 6 can begin without reworking the model definitions or the split logic
- the project can clearly distinguish model comparisons from final reporting

---

## Deliverables expected at the end of Phase 5

The following outputs are expected before moving to Phase 6:

- reusable model-training and evaluation framework
- seasonal-naive baseline code and metrics
- Random Forest training code and saved model artifact
- ANN training code and saved model artifact
- LSTM training code and saved model artifact
- validation and test predictions for each model
- metric comparison table and summary plots
- saved model metadata and configuration provenance
- clear documentation of what was fitted and what remained untouched

---

## Exit criteria for Phase 5

Phase 5 is complete only when all of the following are true:

- [x] the saved shared feature matrix and target from Phase 4 are used as the source of truth
- [x] the seasonal-naive baseline is implemented and scored
- [x] Random Forest, ANN, and LSTM have each been trained and evaluated under the same chronology
- [x] all transformations are fit on training data only
- [x] the test period remains untouched until the final comparison step
- [x] model outputs and diagnostic artifacts are saved reproducibly
- [x] the package is ready for Phase 6 evaluation and regime analysis

---

## Phase 6 handoff

Once the checks above are complete, Phase 6 can begin with:
- final model comparison
- test-set performance review
- residual and forecast diagnostics
- demand-regime analysis (if included)
- reporting of model strengths and limitations

---

## Notes for implementation discipline

- Keep Phase 5 modular: evaluation, forecasting models, and diagnostics should be separated where possible.
- Prefer reproducible scripts and saved metrics over notebook-only experimentation.
- Do not tune on the test set; use validation for tuning and the test set only for final comparison.
- If a model fails to train or behaves erratically, record the root cause and keep the pipeline reproducible.
- Treat the synthetic dataset as a pipeline proof-of-concept, not as an evidence base for real-world operational claims.

This plan should be reviewed before implementation, and only after approval should Phase 5 coding begin.
