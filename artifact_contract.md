# Artifact Contract

The ENKF pipeline is file-driven. Every notebook should generate the files
needed by downstream notebooks, and downstream notebooks should read those files
from the canonical paths in `shared_config.py`.

## Producer And Consumer Map

| Producer | Key artifacts                                                                                                                               | Consumers                |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| 01       | Raw train, validation, and test parquet files; raw scaler and feature list; raw default sequence bundle.                                    | 02, 03, 04a, 04b, 06     |
| 01       | EnKF-preprocessed train, validation, and test parquet files; EnKF scaler and feature list; EnKF default sequence bundle; EnKF Q/R metadata. | 02, 03, 04, 04a, 04b     |
| 02       | `best_hps_lstm.json`, `best_hps_lstm_enkf.json`, and `tuning_manifest.json`.                                                                | 03                       |
| 02       | `sequences_lstm_best.npz` and `sequences_lstm_enkf_best.npz`.                                                                               | 03, 04, 06               |
| 03       | Standalone LSTM checkpoint, metrics, and test probabilities.                                                                                | 04, 05, 06, 07, 09, 09a  |
| 03       | LSTM-EnKF checkpoint, metrics, and test probabilities.                                                                                      | 04, 04b, 05, 07, 09, 09a |
| 03       | Simple-baseline metrics and probability streams.                                                                                            | 07, 08                   |
| 04       | MC-dropout predictions, EnKF-LSTM uncertainty metrics, and prediction CSV.                                                                  | 05, 07, 08               |
| 04a      | EnKF innovation, gain, and state diagnostics in `enkf_diagnostics.npz`.                                                                     | 04b                      |
| 04b      | 3-D MC tensor, variance-decomposition table and figure, and refreshed EnKF probability stream.                                              | 08, 09, 09a              |
| 05       | Bootstrap/statistical analysis JSON and sensitivity outputs.                                                                                | 07, 09a                  |
| 06       | Alternative-model probability streams and tuning summary.                                                                                   | 07, 09, 09a              |
| 07       | Final model-comparison tables, figures, and Brier decomposition.                                                                            | 08, 09a                  |
| 08       | Uncertainty diagnostics, triage tables, coverage figures, and calibration figures.                                                          | Manuscript outputs       |
| 09       | Validator tables and figures for alternative-model adjudication.                                                                            | 09a                      |
| 09a      | Block-bootstrap audit and power-analysis tables and figures.                                                                                | Manuscript outputs       |

## Canonical Output Locations

```text
output_refactored/
output_refactored/intermediate/
output_refactored/enkf_decomposition/
output_refactored/diagnostics/
output_refactored/validation_tools/
output_refactored/block_bootstrap/
```

## Shared Path Surface

`shared_config.py` is the pipeline contract. The most important path groups are:

| Path group                                               | Purpose                                                                                                         |
|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `RAW_*`                                                  | Standalone-LSTM branch data, scaler, feature-list, sequence, hyperparameter, checkpoint, and probability paths. |
| `ENKF_*`                                                 | EnKF-preprocessor branch data, scaler, feature-list, sequence, noise, checkpoint, and probability paths.        |
| `*_METRICS_PATH`                                         | JSON metric summaries consumed by comparison and reporting notebooks.                                           |
| `MC_PREDS_PATH`                                          | MC-dropout tensor used by uncertainty and decomposition notebooks.                                              |
| `ALT_MODEL_STREAMS_PATH`                                 | Alternative-model probability stream bundle used by notebooks 07, 09, and 09a.                                  |
| `SENSITIVITY_RESULTS_PATH` and `PROPAGATOR_RESULTS_PATH` | Optional sensitivity outputs used by final comparison and reporting.                                            |

## Empty Or Optional Outputs

Optional outputs should degrade gracefully. If a sensitivity or diagnostics table
has no rows, it should still be written as a headered CSV rather than as a
zero-byte file. Downstream notebooks should use optional-table readers where a
missing or empty optional artifact is acceptable.

