# Project Support Files

The numbered notebooks delegate shared behavior to Python modules in the
repository root. These modules keep the book chapters focused on analysis while
preserving a single contract for paths, model utilities, and ENKF logic.

| File | Role |
|---|---|
| `shared_config.py` | Canonical workflow flags, model constants, state-variable lists, and artifact paths. Every maintained notebook imports this file. |
| `pipeline_utils.py` | File-contract helpers for required artifacts, JSON writing, hyperparameter loading, sequence persistence, threshold selection, and probability-stream evaluation. |
| `lstm_utils.py` | Sequence construction, LSTM builders, and class-weight calculation. |
| `enkf_utils.py` | Shared EnKF implementation, causal EnKF preprocessing, MC-dropout forecast helpers, and propagator-sensitivity experiment harness. |
| `enkf_preprocessor.py` | Standalone copy of EnKF preprocessor functions retained for reference; the active pipeline imports integrated helpers from `enkf_utils.py`. |
| `propagators.py` | Random-walk, AR(1), VAR(1), and seasonal AR(1) state propagators used by the propagator-sensitivity analysis. |
| `brier_decomposition.py` | Murphy/Ferro-Fricker Brier-score decomposition used in the final comparison notebook. |

## Configuration Points

The main switches live in `shared_config.py`.

| Setting | Meaning |
|---|---|
| `FAST_TEST` | When `True`, runs a small smoke-test tuning budget. Keep `False` for production analyses. |
| `OUTPUT_DIR` | Root directory for generated artifacts. The maintained pipeline uses `output_refactored/`. |
| `INPUT_DATA_PATH` | Weekly red-tide input table. The project keeps the original misspelled path `data_weekly_intepolated.csv` for compatibility. |
| `ENKF_STATE_VARS` | Driver variables included in the EnKF state. |
| `MANUAL_Q_DIAG` and `MANUAL_R_DIAG` | Manual process and observation noise diagonals for the EnKF preprocessor. |
| `RAW_*` and `ENKF_*` | Separate artifact paths for the standalone and EnKF-preprocessed branches. |

## Local Environment

Install notebook dependencies into a virtual environment before running the
pipeline.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The GitHub Pages workflow builds the MyST book separately from the modeling
environment, so generated notebook outputs should be committed or produced in
the notebook environment before the book is published.

