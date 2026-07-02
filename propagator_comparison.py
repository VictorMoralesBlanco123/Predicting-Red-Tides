"""
Fair propagation-method comparison for the LSTM-EnKF preprocessor branch.

The legacy notebook-05 propagator sensitivity harness swaps the EnKF forecast
step at test time after the LSTM is trained.  The functions here implement the
fair comparison instead: rebuild the EnKF-preprocessed data branch, retrain the
LSTM with the tuned EnKF hyperparameters, and evaluate held-out test metrics
for each fitted propagator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import onset_utils as ou
from data_pipeline_utils import (
    add_onset_feature_columns,
    create_features,
    load_and_prepare_data,
    preprocess_data,
    split_data_chronological,
)
from enkf_utils import estimate_enkf_noise, enkf_preprocess_drivers_with_propagator
from lstm_utils import build_tunable_lstm, class_weights_powered
from pipeline_utils import (
    best_f1_threshold,
    build_sequence_artifact,
    evaluate_probability_stream,
    load_hps,
    write_json,
)
from propagators import fit_all_propagators, summarize_propagators
from shared_config import (
    BASE_FEATURES,
    BASIC_LAG_CONFIG,
    BLOOM_THRESHOLD,
    DATETIME_COL,
    DEFAULT_HPS_DICT,
    ENHANCED_LAG_CONFIG,
    ENKF_BEST_HPS_PATH,
    ENKF_OBS_NOISE_FRAC,
    ENKF_PREPROCESS_STATE_VARS,
    ENKF_PROC_NOISE_FRAC,
    ENKF_SEASONAL_N_HARMONICS,
    ENKF_PROTECT_OBS_FRAC,
    ENKF_PROTECT_VARS,
    ENKF_APPEND_STD_FEATURES,
    FAST_TEST,
    FORECAST_HORIZON,
    INPUT_DATA_PATH,
    N_ENKF,
    OUTPUT_DIR,
    PROPAGATOR_COMPARISON_DIR,
    PROPAGATOR_COMPARISON_TABLE_PATH,
    PROPAGATOR_RESULTS_PATH,
    SCALER_TYPE,
    SEED,
    SENSITIVITY_RESULTS_PATH,
    TARGET_BINARY_COL,
    TARGET_COL,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_SPLIT_RATIO,
    USE_CLASS_WEIGHT,
    USE_ENHANCED_FEATURES,
    VALIDATION_SPLIT_RATIO,
)


PROPAGATOR_ORDER = ["RandomWalk", "AR1", "VAR1", "SeasonalAR1"]


def _safe_label(label: str) -> str:
    return (
        str(label)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _onset_counts(y_true, y_pred):
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    onsets = np.where((y_true[:-1] == 0) & (y_true[1:] == 1))[0] + 1
    n_onsets = int(len(onsets))
    n_det = int(sum(y_pred[idx] == 1 for idx in onsets if idx < len(y_pred)))
    return n_onsets, n_det, float(n_det / n_onsets) if n_onsets else float("nan")


def _build_table_s4(results: dict, propagator_order: Iterable[str] = PROPAGATOR_ORDER) -> pd.DataFrame:
    ordered = [p for p in propagator_order if p in results]
    ordered += [p for p in results if p not in ordered]
    rows = []

    for label in ordered:
        res = results[label]
        n_onsets = int(res["n_onsets"])
        n_det = int(res["n_det"])
        onset_text = "N/A" if n_onsets == 0 else f"{n_det}/{n_onsets} ({100.0 * n_det / n_onsets:.0f}%)"
        row = {
            "Propagator": f"{label}" + (" *" if label == "RandomWalk" else ""),
            "Accuracy (%)": f"{res['acc'] * 100:.2f}",
            "Precision (%)": f"{res['prec'] * 100:.2f}",
            "Recall (%)": f"{res['rec'] * 100:.2f}",
            "F1 (%)": f"{res['f1'] * 100:.2f}",
            "Brier Score": f"{res['brier']:.4f}",
            "Onset Hit Rate": onset_text,
            "Threshold": f"{res['threshold']:.3f}",
            "Q trace": f"{res.get('Q_trace', float('nan')):.3e}",
        }
        if res.get("spectral_radius") is not None:
            row["VAR(1) rho(A)"] = f"{res['spectral_radius']:.3f}"
        rows.append(row)

    return pd.DataFrame(rows)


def _train_lstm_for_propagator(
    label: str,
    seq: dict,
    hps: dict,
    branch_dir: Path,
    epochs: int,
    patience: int,
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    h = dict(DEFAULT_HPS_DICT)
    h.update(hps)
    batch_size = int(h["batch_size"])
    input_shape = (seq["X_train"].shape[1], seq["X_train"].shape[2])
    model = build_tunable_lstm(input_shape, h)

    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = class_weights_powered(
            seq["y_train"].flatten().astype(int),
            power=h.get("cw_power", 1.0),
        )

    sample_weight = ou.onset_sample_weights(
        seq["y_train"],
        onset_multiplier=5.0,
        pre_onset_multiplier=3.0,
        pre_onset_window=4,
        buffer_weeks=4,
        class_weight=class_weights,
    )

    model_path = branch_dir / f"lstm_enkf_propagator_{_safe_label(label)}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        seq["X_train"],
        seq["y_train"],
        validation_data=(seq["X_val"], seq["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=sample_weight,
        callbacks=callbacks,
        verbose=2,
    )

    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    return model, history, model_path


def _evaluate_model(label: str, model, seq: dict, prop, branch_dir: Path) -> dict:
    val_prob = model.predict(seq["X_val"], verbose=0).flatten().astype(np.float32)
    test_prob = model.predict(seq["X_test"], verbose=0).flatten().astype(np.float32)
    y_val = np.asarray(seq["y_val"]).flatten().astype(int)
    y_test = np.asarray(seq["y_test"]).flatten().astype(int)

    threshold = best_f1_threshold(y_val, val_prob)
    metrics = evaluate_probability_stream(y_test, test_prob, threshold=threshold)
    y_pred = (test_prob >= threshold).astype(int)
    n_onsets, n_det, ohr = _onset_counts(y_test, y_pred)

    prob_path = branch_dir / f"test_probs_{_safe_label(label)}.npy"
    np.save(prob_path, test_prob)

    row = {
        "acc": metrics["accuracy"],
        "prec": metrics["precision"],
        "rec": metrics["recall"],
        "f1": metrics["F1@threshold"],
        "brier": metrics["Brier"],
        "auc": metrics["AUC"],
        "auprc": metrics["AUPRC"],
        "n_onsets": n_onsets,
        "n_det": n_det,
        "ohr": ohr,
        "threshold": float(threshold),
        "threshold_source": "validation_F1",
        "Q_trace": float(np.trace(prop.Q)) if getattr(prop, "Q", None) is not None else float("nan"),
        "n_train": int(len(seq["y_train"])),
        "n_val": int(len(seq["y_val"])),
        "n_test": int(len(seq["y_test"])),
        "seq_length": int(seq["X_train"].shape[1]),
        "n_features": int(seq["X_train"].shape[2]),
        "prob_path": str(prob_path),
    }
    if hasattr(prop, "spectral_radius") and prop.spectral_radius is not None:
        row["spectral_radius"] = float(prop.spectral_radius)
    return row


def _load_existing_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def run_fair_propagator_comparison(
    methods: Optional[Iterable[str]] = None,
    fast_test: Optional[bool] = None,
):
    """Run the fair retrained propagator comparison.

    Parameters
    ----------
    methods:
        Optional subset of propagator labels. Defaults to RandomWalk, AR1,
        VAR1, and SeasonalAR1.
    fast_test:
        When true, reduces EnKF members and training epochs for a smoke test.
        Defaults to ``shared_config.FAST_TEST``.

    Returns
    -------
    (results, table_s4, ranked)
        ``results`` is the JSON-compatible dict written to
        ``PROPAGATOR_RESULTS_PATH``.
    """
    use_fast = FAST_TEST if fast_test is None else bool(fast_test)
    methods = list(methods or PROPAGATOR_ORDER)
    n_enkf_compare = min(N_ENKF, 10) if use_fast else N_ENKF
    epochs = min(TRAIN_EPOCHS, 3) if use_fast else TRAIN_EPOCHS
    patience = min(TRAIN_PATIENCE, 2) if use_fast else TRAIN_PATIENCE

    if not os.path.exists(ENKF_BEST_HPS_PATH):
        raise FileNotFoundError(f"Missing {ENKF_BEST_HPS_PATH}. Run notebook 02 before notebook 03a.")

    output_dir = Path(PROPAGATOR_COMPARISON_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)

    hps = load_hps(ENKF_BEST_HPS_PATH, DEFAULT_HPS_DICT)
    seq_length = int(hps["seq_length"])

    df_raw_base = load_and_prepare_data(
        INPUT_DATA_PATH,
        DATETIME_COL,
        TARGET_COL,
        BLOOM_THRESHOLD,
        TARGET_BINARY_COL,
    )
    df_raw_features = add_onset_feature_columns(df_raw_base, TARGET_BINARY_COL)
    df_raw_processed = create_features(
        df_raw_features,
        TARGET_COL,
        TARGET_BINARY_COL,
        BASE_FEATURES,
        use_enhanced=USE_ENHANCED_FEATURES,
        basic_cfg=BASIC_LAG_CONFIG,
        enhanced_cfg=ENHANCED_LAG_CONFIG,
    )
    raw_train_df, _, _ = split_data_chronological(
        df_raw_processed,
        TRAIN_SPLIT_RATIO,
        VALIDATION_SPLIT_RATIO,
    )

    state_vars_present = [
        v for v in ENKF_PREPROCESS_STATE_VARS
        if v in df_raw_base.columns and v in raw_train_df.columns
    ]
    if not state_vars_present:
        raise ValueError("No EnKF state variables are available for propagator comparison.")

    q_reference_diag, r_diag = estimate_enkf_noise(
        raw_train_df,
        state_vars_present,
        obs_noise_frac=ENKF_OBS_NOISE_FRAC,
        proc_noise_frac=ENKF_PROC_NOISE_FRAC,
        protect_vars=ENKF_PROTECT_VARS,
        protect_obs_frac=ENKF_PROTECT_OBS_FRAC,
    )

    all_propagators = fit_all_propagators(
        raw_train_df,
        state_vars_present,
        manual_q_diag=None,
        seasonal_n_harmonics=ENKF_SEASONAL_N_HARMONICS,
    )
    missing = [m for m in methods if m not in all_propagators]
    if missing:
        raise KeyError(f"Unknown fitted propagator(s): {missing}")

    prop_summary = summarize_propagators(all_propagators)
    prop_summary.to_csv(output_dir / "propagator_parameter_summary.csv", index=False)

    results = {}
    artifacts = {}

    for label in methods:
        prop = all_propagators[label]
        safe = _safe_label(label)
        branch_dir = output_dir / safe
        branch_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"FAIR PROPAGATOR BRANCH: {label}")
        print("=" * 100)

        df_enkf_base = enkf_preprocess_drivers_with_propagator(
            df_raw_base,
            state_vars=state_vars_present,
            propagator=prop,
            R_diag=r_diag,
            Q_matrix=prop.Q,
            n_enkf=n_enkf_compare,
            seed=SEED,
            append_std=ENKF_APPEND_STD_FEATURES,
        )
        df_enkf_features = add_onset_feature_columns(df_enkf_base, TARGET_BINARY_COL)
        df_processed = create_features(
            df_enkf_features,
            TARGET_COL,
            TARGET_BINARY_COL,
            BASE_FEATURES,
            use_enhanced=USE_ENHANCED_FEATURES,
            basic_cfg=BASIC_LAG_CONFIG,
            enhanced_cfg=ENHANCED_LAG_CONFIG,
        )
        train_df, val_df, test_df = split_data_chronological(
            df_processed,
            TRAIN_SPLIT_RATIO,
            VALIDATION_SPLIT_RATIO,
        )

        train_scaled, val_scaled, test_scaled, _, feature_columns = preprocess_data(
            train_df,
            val_df,
            test_df,
            TARGET_BINARY_COL,
            scaler_type=SCALER_TYPE,
            scaler_path=str(branch_dir / f"scaler_{safe}.joblib"),
            feature_list_path=str(branch_dir / f"feature_columns_{safe}.joblib"),
            imputer_path=str(branch_dir / f"imputer_{safe}.joblib"),
        )

        train_df.to_parquet(branch_dir / f"train_df_{safe}.parquet")
        val_df.to_parquet(branch_dir / f"val_df_{safe}.parquet")
        test_df.to_parquet(branch_dir / f"test_df_{safe}.parquet")
        train_scaled.to_parquet(branch_dir / f"train_scaled_{safe}.parquet")
        val_scaled.to_parquet(branch_dir / f"val_scaled_{safe}.parquet")
        test_scaled.to_parquet(branch_dir / f"test_scaled_{safe}.parquet")

        seq_path = branch_dir / f"sequences_{safe}.npz"
        seq_arrays = build_sequence_artifact(
            train_scaled,
            val_scaled,
            test_scaled,
            feature_columns,
            TARGET_BINARY_COL,
            seq_length,
            FORECAST_HORIZON,
            str(seq_path),
        )
        seq = {
            **seq_arrays,
            "feature_columns": list(feature_columns)
            + ([] if TARGET_BINARY_COL in feature_columns else [TARGET_BINARY_COL]),
        }
        print(
            f"Sequences: X_train={seq['X_train'].shape}, "
            f"X_val={seq['X_val'].shape}, X_test={seq['X_test'].shape}"
        )

        model, _, model_path = _train_lstm_for_propagator(
            label,
            seq,
            hps,
            branch_dir,
            epochs=epochs,
            patience=patience,
        )
        row = _evaluate_model(label, model, seq, prop, branch_dir)
        row["model_path"] = str(model_path)
        results[label] = row

        artifacts[label] = {
            "branch_dir": str(branch_dir),
            "sequence_path": str(seq_path),
            "model_path": str(model_path),
            "feature_columns_path": str(branch_dir / f"feature_columns_{safe}.joblib"),
            "scaler_path": str(branch_dir / f"scaler_{safe}.joblib"),
        }
        write_json(branch_dir / f"metrics_{safe}.json", row)
        print(
            f"Metrics for {label}: acc={row['acc']:.3f}, "
            f"f1={row['f1']:.3f}, brier={row['brier']:.4f}, ohr={row['ohr']:.3f}"
        )

    metadata = {
        "comparison_type": "fair_retrained_preprocessor_branch",
        "threshold_source": "validation_F1",
        "hps_policy": "fixed ENKF-branch hyperparameters from notebook 02 to isolate propagation method",
        "q_policy": "each propagator uses process-noise covariance fitted on train residuals",
        "r_policy": "R estimated once from the raw training split with estimate_enkf_noise",
        "state_vars": state_vars_present,
        "n_enkf": int(n_enkf_compare),
        "train_epochs": int(epochs),
        "train_patience": int(patience),
        "hps_path": ENKF_BEST_HPS_PATH,
        "hps": hps,
        "artifacts": artifacts,
        "Q_reference_diag_from_notebook01_policy": q_reference_diag,
        "R_diag": r_diag,
    }

    write_json(PROPAGATOR_RESULTS_PATH, results)
    write_json(output_dir / "propagator_comparison_metadata.json", metadata)
    write_json(output_dir / "propagator_comparison_results.json", results)

    sensitivity_payload = _load_existing_json(SENSITIVITY_RESULTS_PATH)
    sensitivity_payload["propagator"] = results
    sensitivity_payload["propagator_note"] = metadata
    write_json(SENSITIVITY_RESULTS_PATH, sensitivity_payload)

    table_s4 = _build_table_s4(results)
    table_s4.to_csv(PROPAGATOR_COMPARISON_TABLE_PATH, index=False)
    table_s4.to_csv("output/TableS4_propagator_sensitivity.csv", index=False)

    ranked = (
        pd.DataFrame.from_dict(results, orient="index")
        .rename_axis("propagator")
        .reset_index()
        .sort_values(["brier", "f1"], ascending=[True, False])
    )
    ranked.to_csv(output_dir / "propagator_ranked_metrics.csv", index=False)

    print(f"\nWrote {PROPAGATOR_RESULTS_PATH}")
    print(f"Wrote {SENSITIVITY_RESULTS_PATH}")
    print(f"Wrote {PROPAGATOR_COMPARISON_TABLE_PATH}")
    print("Wrote output/TableS4_propagator_sensitivity.csv")

    return results, table_s4, ranked


if __name__ == "__main__":
    run_fair_propagator_comparison()
