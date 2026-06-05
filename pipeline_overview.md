# Pipeline Overview

This book is organized around the maintained ENKF notebook pipeline. The
workflow is sequential: each notebook writes explicit artifacts under
`output_refactored/`, and later notebooks load those files through paths defined
in `shared_config.py`.

The maintained pipeline replaces the original monolithic
`LSTM+EnKF_v2.ipynb`. The legacy notebook remains in the repository for
provenance, but it is not part of the primary book navigation.

## Pipeline Map

| Stage | Notebook                                   | Role                                                                                                                            | Main downstream dependency                                                      |
|-------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| 1     | `01_data_pipeline.ipynb`                   | Build chronological train, validation, and test splits; engineer features; create raw and EnKF-preprocessed sequence artifacts. | Produces the raw and EnKF branch datasets used by all model notebooks.          |
| 2     | `02_tuning.ipynb`                          | Tune the standalone LSTM and the EnKF-preprocessed LSTM with the same Optuna search harness.                                    | Writes branch-specific hyperparameters and tuned sequence bundles for training. |
| 3     | `03_train_and_baseline_eval.ipynb`         | Train the tuned standalone LSTM, train the tuned LSTM-EnKF branch, and evaluate simple baselines.                               | Produces checkpoints, probability streams, and primary metric files.            |
| 4     | `04_enkf_uncertainty.ipynb`                | Run MC dropout on the EnKF branch and write uncertainty-aware predictions.                                                      | Supplies uncertainty streams for statistics, comparison, and diagnostics.       |
| 5     | `04a_enkf_innovation_decomposition.ipynb`  | Decompose EnKF innovations and Kalman gains.                                                                                    | Writes EnKF diagnostic tensors for variance decomposition.                      |
| 6     | `04b_enkf_variance_decomposition.ipynb`    | Split uncertainty into driver-state and model-weight components.                                                                | Refreshes the 3-D MC prediction tensor and decomposition outputs.               |
| 7     | `05_statistical_analysis.ipynb`            | Run bootstrap/statistical analyses and sensitivity summaries.                                                                   | Feeds the final comparison and block-bootstrap audit.                           |
| 8     | `06_alternative_models.ipynb`              | Train and evaluate comparison models on the same chronological split.                                                           | Writes alternative-model streams for final comparison and validator analysis.   |
| 9     | `07_final_comparison.ipynb`                | Assemble final model comparisons, summary tables, figures, and Brier decomposition.                                             | Supplies comparison outputs used by diagnostics and manuscript reporting.       |
| 10    | `08_diagnostics.ipynb`                     | Convert uncertainty and prediction outputs into diagnostic tables and publication-ready figures.                                | Frames the hybrid branch as a triage and validation signal.                     |
| 11    | `09_lstm_validators.ipynb`                 | Compare standalone-LSTM and LSTM-EnKF validator streams against alternative models.                                             | Writes validator tables, figures, and paired-bootstrap summaries.               |
| 12    | `09a_block_bootstrap_power_analysis.ipynb` | Audit IID bootstrap claims with moving-block bootstrap and power analysis.                                                      | Produces temporal-dependence and power-analysis reporting outputs.              |

## Branches

The primary model comparison uses two separate branches.

| Branch                 | Training inputs                             | Model                 | Output stream               |
|------------------------|---------------------------------------------|-----------------------|-----------------------------|
| Standalone LSTM        | Raw engineered and scaled sequences         | Tuned LSTM            | `LSTM_TEST_PROBS_PATH`      |
| LSTM-EnKF preprocessor | Causally EnKF-preprocessed driver sequences | Separately tuned LSTM | `LSTM_ENKF_TEST_PROBS_PATH` |

The EnKF branch is not a test-only correction. Notebook 01 applies the filter
chronologically before scaling and sequence construction, then notebooks 02 and
03 tune and train a separate LSTM on that transformed feature space.

## Interpretation

The current workflow should be read as a forecasting and validation pipeline,
not only as a leaderboard. The standalone LSTM and LSTM-EnKF branch are compared
directly, but the durable interpretation of the EnKF branch is its value as a
diagnostic and validator stream. Notebook 08 evaluates uncertainty and triage
signals, while notebook 09 tests whether the EnKF-derived validator helps
adjudicate alternative-model predictions on matched test weeks.

## Reproducibility Contract

- Run notebooks from the project root in numeric order.
- Do not rely on hidden notebook kernel state between chapters.
- Load cross-notebook dependencies through paths in `shared_config.py`.
- Use `pipeline_utils.py` helpers for sequence artifacts, JSON-safe writing,
  threshold selection, and probability-stream evaluation.
- Re-run notebook 08 after notebook 04b when the driver-state versus
  model-weight variance decomposition is needed.
- Keep notebook 06 comparison models on comparable tuning and training budgets
  to the LSTM branches.

