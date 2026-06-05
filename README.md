# LSTM + EnKF Red-Tide Forecasting Pipeline

This repository contains a sequential notebook pipeline for weekly red-tide
bloom forecasting. The main comparison is between a standalone tuned LSTM and a
separate LSTM trained on driver sequences that have first been causally
preprocessed with an Ensemble Kalman Filter (EnKF).

The notebooks communicate through explicit files under `output_refactored/`.
Downstream notebooks should load those artifacts through paths defined in
`shared_config.py`; they should not rely on hidden state from a previously run
notebook kernel.

## Book Contents

:::{toc}
:context: project
:::

## Run Order

Run notebooks from the project root in this order:

```text
01_data_pipeline.ipynb
02_tuning.ipynb
03_train_and_baseline_eval.ipynb
04_enkf_uncertainty.ipynb
04a_enkf_innovation_decomposition.ipynb
04b_enkf_variance_decomposition.ipynb
05_statistical_analysis.ipynb
06_alternative_models.ipynb
07_final_comparison.ipynb
08_diagnostics.ipynb
09_lstm_validators.ipynb
09a_block_bootstrap_power_analysis.ipynb
```

Notebook 06 can run once notebooks 03 and 04 have produced the LSTM, LSTM-EnKF,
and baseline probability streams. Notebook 07 expects both the uncertainty
stream from notebook 04 and the alternative-model streams from notebook 06.

Notebook 04a consumes the EnKF-preprocessor artifacts from notebook 01 and
writes `output_refactored/intermediate/enkf_diagnostics.npz`. Notebook 04b
consumes that diagnostics file plus the trained EnKF-branch LSTM from notebook
03, then overwrites `MC_PREDS_PATH` with a 3-D tensor shaped
`(n_test, n_enkf, n_mc)`. Re-run notebook 08 after notebook 04b when you want
the driver-state vs model-weight variance decomposition tables and figures.

The legacy `LSTM+EnKF_v2.ipynb` is retained as the original monolithic notebook;
the numbered notebooks are the maintained pipeline.

## Primary Comparison

- **Standalone LSTM**: trained on raw engineered and scaled sequence artifacts.
- **LSTM-EnKF preprocessor**: applies the EnKF causally to the driver series
  before scaling and sequence creation, then tunes and trains a separate LSTM on
  those EnKF-preprocessed sequences.

The EnKF branch is not a test-only correction to the last timestep. The filter
is applied consistently to train, validation, and test windows before model
training so the LSTM sees the same transformed feature space during training
and inference.

## Custom Python Support Files

| File | Role |
|---|---|
| `shared_config.py` | Canonical workflow flags, model constants, state-variable lists, and artifact paths. Every notebook imports this file. |
| `lstm_utils.py` | Sequence construction, default/tunable LSTM builders, and powered class-weight calculation. |
| `pipeline_utils.py` | Lightweight artifact helpers: JSON writing, required-file checks, hyperparameter loading, sequence artifact persistence, threshold selection, and probability-stream metrics. |
| `enkf_utils.py` | Shared EnKF implementation, legacy MC-dropout forecast loop, causal EnKF preprocessing helpers, and propagator-sensitivity experiment harness. |
| `enkf_preprocessor.py` | Standalone copy of the EnKF preprocessor functions. The active pipeline imports the integrated versions from `enkf_utils.py`. |
| `propagators.py` | Random-walk, AR(1), VAR(1), and seasonal AR(1) state propagators used by the legacy propagator-sensitivity analysis. |
| `brier_decomposition.py` | Murphy/Ferro-Fricker Brier-score decomposition used by the final comparison notebook. |

## Artifact Contract

| Producer | Key artifacts | Consumers |
|---|---|---|
| 01 | raw train/val/test parquet, raw scaler/features, raw default sequences | 02, 03, 04a, 04b, 06 |
| 01 | EnKF-preprocessed train/val/test parquet, EnKF scaler/features, EnKF default sequences, EnKF Q/R metadata | 02, 03, 04, 04a, 04b |
| 02 | tuned raw and EnKF hyperparameter JSON files | 03 |
| 02 | tuned raw and EnKF sequence NPZ files | 03, 04, 06 |
| 03 | standalone LSTM checkpoint, metrics, probabilities | 04, 05, 06, 07, 09, 09a |
| 03 | LSTM-EnKF checkpoint, metrics, probabilities | 04, 04b, 05, 07, 09, 09a |
| 03 | simple-baseline metrics and probability streams | 07, 08 |
| 04 | MC-dropout predictions, EnKF-LSTM metrics, prediction CSV | 05, 07, 08 |
| 04a | EnKF innovation/gain diagnostics and `enkf_diagnostics.npz` | 04b |
| 04b | 3-D MC tensor, variance-decomposition table/figure, refreshed EnKF probability stream | 08, 09, 09a |
| 05 | bootstrap/statistical analysis JSON and sensitivity outputs | 07, 09a |
| 06 | alternative-model probability streams and tuning summary | 07, 09, 09a |
| 07 | final model-comparison CSV/tables/figures and Brier decomposition | 08, 09a |
| 08 | uncertainty diagnostics, triage tables, coverage/calibration figures | manuscript outputs |
| 09 | validator tables/figures for alternative-model stream adjudication | 09a |
| 09a | block-bootstrap audit and power-analysis tables/figures | manuscript outputs |

Common artifact locations:

```text
output_refactored/
output_refactored/intermediate/
output_refactored/enkf_decomposition/
output_refactored/diagnostics/
output_refactored/validation_tools/
output_refactored/block_bootstrap/
```

## Configuration

Edit `shared_config.py` to change global behavior:

- `FAST_TEST = True` runs a small smoke-test tuning budget.
- `FAST_TEST = False` is the production setting.
- `OUTPUT_DIR = 'output_refactored/'` controls canonical artifact output.
- `ENKF_STATE_VARS`, `MANUAL_Q_DIAG`, `MANUAL_R_DIAG`, and related EnKF
  constants define the state assimilation setup.
- `RAW_*` and `ENKF_*` paths separate standalone-LSTM artifacts from
  EnKF-preprocessor artifacts.

## Environment

Install the Python dependencies from `requirements.txt` in a virtual
environment before running the notebooks:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The pipeline expects `data_weekly_intepolated.csv` in the project root and uses
`external_data/` for supporting inputs.

## Fairness Notes

- Notebook 02 runs the same Optuna search space and budget for the standalone
  LSTM and the EnKF-preprocessed LSTM.
- Notebook 01 estimates EnKF preprocessor noise from the training slice and
  then applies the causal filter over the chronological data.
- Alternative models in notebook 06 use the same chronological train,
  validation, and test split. Model-specific validation grids are written to
  `alt_model_tuning_summary.json`.
- The test set is held out until final evaluation.
- Notebook 09a audits IID bootstrap claims with moving-block bootstrap because
  bloom weeks are temporally clustered.
