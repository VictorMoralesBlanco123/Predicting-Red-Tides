"""
shared_config.py
================

Shared configuration for the LSTM+EnKF red-tide forecasting pipeline.

This module is imported by every notebook in the split pipeline:

    01_data_pipeline.ipynb
    02_tuning.ipynb
    03_train_and_baseline_eval.ipynb
    04_enkf_uncertainty.ipynb
    05_statistical_analysis.ipynb
    06_alternative_models.ipynb
    07_final_comparison.ipynb
    08_diagnostics.ipynb

Usage at the top of each notebook:

    from shared_config import *
"""

import os

# ---------------------------------------------------------------------------
# Workflow flags  (was Part 1, cell 4)
# ---------------------------------------------------------------------------
# === Feature Engineering ===
USE_ENHANCED_FEATURES = True

# === Preprocessing ===
SCALER_TYPE = 'Robust'           # 'Robust' or 'Standard'

# === Training ===
USE_CLASS_WEIGHT = True
PERFORM_TUNING   = True          # Optuna hyperparameter search

# === Advanced Steps ===
PERFORM_ENKF     = True

# === Test mode (papermill can override with -p FAST_TEST True) ===
# When True, Optuna runs a tiny budget so the whole notebook can smoke-test
# end-to-end in a few minutes. Production runs should keep this False.
FAST_TEST        = False
SEED             = 42

# ---------------------------------------------------------------------------
# File paths  (was Part 1, cell 5)
# ---------------------------------------------------------------------------
INPUT_DATA_PATH = 'data_weekly_intepolated.csv'
EXTERNAL_DATA_DIR = 'external_data/'
OUTPUT_DIR = 'output_refactored/'

scaler_suffix = SCALER_TYPE.lower()
feature_suffix = 'enhanced' if USE_ENHANCED_FEATURES else 'basic'
SCALER_FILENAME = os.path.join(OUTPUT_DIR, f'red_tide_scaler_{scaler_suffix}.joblib')
FEATURE_LIST_FILENAME = os.path.join(OUTPUT_DIR, f'red_tide_feature_list_{feature_suffix}.joblib')
SEQUENCES_FILENAME_TEMPLATE = os.path.join(OUTPUT_DIR, 'sequences_horizon{}wk_{}.npz')
MODEL_CHECKPOINT_TEMPLATE = os.path.join(OUTPUT_DIR, 'best_lstm_model_{}.keras')
MODEL_WEIGHTS_CHECKPOINT_TEMPLATE = os.path.join(OUTPUT_DIR, 'best_lstm_model_{}.weights.h5')

TUNER_PROJECT_DIR = 'keras_tuner_dir_refactored'
TUNER_PROJECT_NAME = 'red_tide_lstm_tuning'

# Artifacts produced by the split pipeline (new — not in original notebook)
INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, 'intermediate')
TRAIN_DF_PATH      = os.path.join(INTERMEDIATE_DIR, 'train_df_unscaled.parquet')
VAL_DF_PATH        = os.path.join(INTERMEDIATE_DIR, 'val_df_unscaled.parquet')
TEST_DF_PATH       = os.path.join(INTERMEDIATE_DIR, 'test_df_unscaled.parquet')
TRAIN_SCALED_PATH  = os.path.join(INTERMEDIATE_DIR, 'train_scaled.parquet')
VAL_SCALED_PATH    = os.path.join(INTERMEDIATE_DIR, 'val_scaled.parquet')
TEST_SCALED_PATH   = os.path.join(INTERMEDIATE_DIR, 'test_scaled.parquet')
BEST_HPS_PATH      = os.path.join(INTERMEDIATE_DIR, 'best_hps.json')
BASELINE_METRICS_PATH = os.path.join(INTERMEDIATE_DIR, 'baseline_metrics.json')
LSTM_TEST_PROBS_PATH  = os.path.join(INTERMEDIATE_DIR, 'lstm_test_probs.npy')
MC_PREDS_PATH         = os.path.join(INTERMEDIATE_DIR, 'mc_predictions.npy')
ENKF_METRICS_PATH     = os.path.join(INTERMEDIATE_DIR, 'enkf_metrics.json')
ALT_MODEL_STREAMS_PATH = os.path.join(INTERMEDIATE_DIR, 'alt_model_streams.npz')
SENSITIVITY_RESULTS_PATH = os.path.join(INTERMEDIATE_DIR, 'sensitivity_results.json')
PROPAGATOR_RESULTS_PATH  = os.path.join(INTERMEDIATE_DIR, 'propagator_sensitivity_results.json')


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Core parameters  (was Part 1, cell 6)
# ---------------------------------------------------------------------------
DATETIME_COL = 'time'
TARGET_COL = 'kb'
TARGET_BINARY_COL = 'bloom_target'
BLOOM_THRESHOLD = 1e5             # Cells/L
SEQUENCE_LENGTH = 26
FORECAST_HORIZON = 1

# Explicit raw vs EnKF-preprocessor branches.
RAW_PIPELINE_TAG = feature_suffix
ENKF_PIPELINE_TAG = f'{feature_suffix}_enkf'
RAW_SEQUENCES_PATH = SEQUENCES_FILENAME_TEMPLATE.format(FORECAST_HORIZON, RAW_PIPELINE_TAG)
ENKF_SEQUENCES_PATH = SEQUENCES_FILENAME_TEMPLATE.format(FORECAST_HORIZON, ENKF_PIPELINE_TAG)
RAW_TUNED_SEQUENCES_PATH = os.path.join(INTERMEDIATE_DIR, 'sequences_lstm_best.npz')
ENKF_TUNED_SEQUENCES_PATH = os.path.join(INTERMEDIATE_DIR, 'sequences_lstm_enkf_best.npz')

RAW_BEST_HPS_PATH = os.path.join(INTERMEDIATE_DIR, 'best_hps_lstm.json')
ENKF_BEST_HPS_PATH = os.path.join(INTERMEDIATE_DIR, 'best_hps_lstm_enkf.json')

ENKF_TRAIN_DF_PATH = os.path.join(INTERMEDIATE_DIR, 'train_df_enkf_unscaled.parquet')
ENKF_VAL_DF_PATH = os.path.join(INTERMEDIATE_DIR, 'val_df_enkf_unscaled.parquet')
ENKF_TEST_DF_PATH = os.path.join(INTERMEDIATE_DIR, 'test_df_enkf_unscaled.parquet')
ENKF_TRAIN_SCALED_PATH = os.path.join(INTERMEDIATE_DIR, 'train_enkf_scaled.parquet')
ENKF_VAL_SCALED_PATH = os.path.join(INTERMEDIATE_DIR, 'val_enkf_scaled.parquet')
ENKF_TEST_SCALED_PATH = os.path.join(INTERMEDIATE_DIR, 'test_enkf_scaled.parquet')
ENKF_SCALER_FILENAME = os.path.join(OUTPUT_DIR, f'red_tide_scaler_{scaler_suffix}_enkf.joblib')
ENKF_FEATURE_LIST_FILENAME = os.path.join(OUTPUT_DIR, f'red_tide_feature_list_{feature_suffix}_enkf.joblib')
ENKF_PREPROCESSOR_NOISE_PATH = os.path.join(INTERMEDIATE_DIR, 'enkf_preprocessor_noise.json')

STANDALONE_LSTM_MODEL_NAME = 'standalone_lstm_tuned_weighted'
ENKF_LSTM_MODEL_NAME = 'lstm_enkf_preprocessor_tuned_weighted'
STANDALONE_LSTM_MODEL_PATH = MODEL_CHECKPOINT_TEMPLATE.format(STANDALONE_LSTM_MODEL_NAME)
ENKF_LSTM_MODEL_PATH = MODEL_CHECKPOINT_TEMPLATE.format(ENKF_LSTM_MODEL_NAME)
LSTM_ENKF_TEST_PROBS_PATH = os.path.join(INTERMEDIATE_DIR, 'lstm_enkf_test_probs.npy')
LSTM_ENKF_METRICS_PATH = os.path.join(INTERMEDIATE_DIR, 'lstm_enkf_metrics.json')
ENKF_MC_PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, 'predictions_enkf_mc.csv')
ENKF_MC_METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, 'metrics_enkf_mc.json')
ALT_MODEL_TUNING_SUMMARY_PATH = os.path.join(INTERMEDIATE_DIR, 'alt_model_tuning_summary.json')


# ---------------------------------------------------------------------------
# Feature engineering config  (was Part 1, cell 7)
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    'zos', 'water_temp',
    'peace_discharge', 'peace_TN', 'peace_TP',
    'wind_u', 'wind_v'
]
BASIC_LAG_CONFIG = {
    'kb_lags':  [1, 2],
    'env_lags': list(range(1, 7))
}
ENHANCED_LAG_CONFIG = {
    'kb_rolling_windows':       [(1, 4), (5, 8), (9, 12)],
    'discharge_rolling_window': 4,
    'env_lags':                 [1]
}

# --- Data Splitting ---
TRAIN_SPLIT_RATIO      = 0.70
VALIDATION_SPLIT_RATIO = 0.15

# --- LSTM defaults (used when PERFORM_TUNING=False or as fallbacks) ---
DEFAULT_HPS_DICT = {
    'seq_length':        12,
    'n_lstm_layers':     2,
    'units_1':           64,
    'units_2':           32,
    'units_3':           16,
    'dropout':           0.3,
    'recurrent_dropout': 0.0,
    'l2_reg':            1e-6,
    'optimizer':         'adam',
    'learning_rate':     1e-3,
    'weight_decay':      1e-6,
    'batch_size':        32,
    'cw_power':          1.0,
}
DEFAULT_LSTM_UNITS    = DEFAULT_HPS_DICT['units_1']
DEFAULT_DROPOUT_RATE  = DEFAULT_HPS_DICT['dropout']
DEFAULT_LEARNING_RATE = DEFAULT_HPS_DICT['learning_rate']

# --- Final Training Parameters ---
TRAIN_EPOCHS    = 50
TRAIN_BATCH_SIZE = DEFAULT_HPS_DICT['batch_size']
TRAIN_PATIENCE  = 10

# --- EnKF Configuration ---
ENKF_STATE_VARS = [
    'peace_discharge', 'peace_TN', 'peace_TP', 'kb', 'wind_u', 'wind_v'
]
N_ENKF = 50
ENKF_PREPROCESS_STATE_VARS = ENKF_STATE_VARS
ENKF_OBS_NOISE_FRAC = 0.25
ENKF_PROC_NOISE_FRAC = 0.02
ENKF_PROTECT_VARS = ['kb']
ENKF_PROTECT_OBS_FRAC = 0.02
ENKF_APPEND_STD_FEATURES = True
ENKF_NOISE_ESTIMATION = 'manual'
MANUAL_R_DIAG = [
    (100.0 * 0.01)**2,
    (0.2  * 0.01)**2,
    (0.02 * 0.01)**2,
    (5.0  * 0.05)**2,
    (5.0  * 0.05)**2,
    (0.1  * 0.01)**2
]
MANUAL_Q_DIAG = [
    (150.0 * 2.0)**2,
    (0.3   * 2.0)**2,
    (0.03  * 2.0)**2,
    (10.0  * 1.5)**2,
    (10.0  * 1.5)**2,
    (0.1   * 1.0)**2
]

# --- MC Dropout ---
N_MC_SAMPLES = 50


def print_config_summary():
    """Print the workflow flags. Call this from the setup cell of any notebook."""
    print("--- Workflow Configuration Summary ---")
    print(f"Enhanced Features Enabled:     {USE_ENHANCED_FEATURES}")
    print(f"Scaler Type Selected:          {SCALER_TYPE}")
    print(f"Class Weighting Enabled:       {USE_CLASS_WEIGHT}")
    print(f"Hyperparameter Tuning Enabled: {PERFORM_TUNING}")
    print(f"EnKF Enabled:                  {PERFORM_ENKF}")
    print(f"FAST_TEST Mode:                {FAST_TEST}")
    print(f"Sequence Length (default):     {SEQUENCE_LENGTH}")
    print(f"Forecast Horizon:              {FORECAST_HORIZON} week(s)")
    print(f"Output directory:              {OUTPUT_DIR}")
    print("--------------------------------------")