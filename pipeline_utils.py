"""
pipeline_utils.py
=================

Small, dependency-light helpers used by the split notebook pipeline.  The
functions here deliberately avoid TensorFlow imports so they can be used by
data, tuning, comparison, and diagnostics notebooks without initializing the
deep-learning stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lstm_utils import create_sequences_from_df


class HPsShim:
    """Dict wrapper with a KerasTuner-like ``get`` API."""

    def __init__(self, params: Mapping):
        self._params = dict(params)

    def get(self, key, default=None):
        return self._params.get(key, default)

    def __getitem__(self, key):
        return self._params[key]

    def __repr__(self):
        return f"HPsShim({self._params})"

    def to_dict(self):
        return dict(self._params)


def jsonable(value):
    """Convert numpy/pandas values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def write_json(path, payload, indent=2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=indent)


def require_files(paths: Iterable[str]) -> None:
    missing = [str(p) for p in paths if not os.path.exists(p)]
    if missing:
        formatted = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(f"Missing required upstream artifact(s):\n{formatted}")


def hps_with_defaults(hps, defaults):
    out = dict(defaults)
    if hps is None:
        return out
    if hasattr(hps, "to_dict"):
        out.update(hps.to_dict())
    else:
        out.update(dict(hps))
    return out


def load_hps(path, defaults):
    if not os.path.exists(path):
        return dict(defaults)
    with open(path, encoding="utf-8") as f:
        return hps_with_defaults(json.load(f), defaults)


def assert_hps_sequence_match(label, hps, seq_len, x_array=None):
    """Validate that tuned HPs and sequence artifacts use the same lookback."""
    seq_len = int(seq_len)
    h = hps_with_defaults(hps, {}) if hps is not None else {}
    if "seq_length" in h and int(h["seq_length"]) != seq_len:
        raise ValueError(
            f"{label} seq_length mismatch: HP file has {int(h['seq_length'])}, "
            f"but sequence artifact has {seq_len}. Re-run notebook 02 and then "
            "rerun downstream notebooks from a fresh kernel."
        )
    if x_array is not None:
        x_seq_len = int(np.asarray(x_array).shape[1])
        if x_seq_len != seq_len:
            raise ValueError(
                f"{label} sequence artifact metadata says seq_length={seq_len}, "
                f"but X has shape[1]={x_seq_len}."
            )
    return seq_len


def build_sequence_artifact(
    train_scaled_df,
    val_scaled_df,
    test_scaled_df,
    feature_columns,
    target_binary_col,
    seq_length,
    forecast_horizon,
    output_path,
):
    """Create and persist aligned LSTM arrays from already-scaled splits.

    The target column is used ONLY to extract ``y``. It is removed from the X
    feature channels (see ``create_sequences_from_df(drop_target_from_x=True)``)
    so the binary bloom label never enters the model input. The persisted
    ``feature_columns`` therefore lists the X channels only (target excluded).
    """
    feature_only = [c for c in feature_columns if c != target_binary_col]
    cols_to_use = list(feature_only)
    cols_to_use.append(target_binary_col)            # target last, used only for y
    target_idx = cols_to_use.index(target_binary_col)

    sequences = {}
    for split_name, df in [
        ("train", train_scaled_df),
        ("val", val_scaled_df),
        ("test", test_scaled_df),
    ]:
        X, y = create_sequences_from_df(
            df[cols_to_use],
            seq_length=seq_length,
            target_col_idx=target_idx,
            pred_step=forecast_horizon,
            return_targets=True,
            drop_target_from_x=True,
        )
        sequences[f"X_{split_name}"] = np.asarray(X, dtype=np.float32)
        sequences[f"y_{split_name}"] = np.asarray(y, dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **sequences,
        feature_columns=np.asarray(feature_only),
        seq_length=np.asarray(seq_length),
        forecast_horizon=np.asarray(forecast_horizon),
    )
    return sequences


def load_sequence_artifact(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "feature_columns": list(data["feature_columns"]),
            "seq_length": int(data["seq_length"]) if "seq_length" in data.files else data["X_train"].shape[1],
            "forecast_horizon": int(data["forecast_horizon"]) if "forecast_horizon" in data.files else None,
        }


def best_f1_threshold(y_true, y_prob, fallback=0.5):
    y_true = np.asarray(y_true).flatten().astype(int)
    y_prob = np.asarray(y_prob).flatten()
    if len(y_true) != len(y_prob):
        n = min(len(y_true), len(y_prob))
        y_true = y_true[-n:]
        y_prob = y_prob[-n:]
    if len(y_true) == 0:
        return float(fallback)
    if len(np.unique(y_true)) < 2:
        return float(fallback)
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    fscore = np.nan_to_num(2 * prec * rec / (prec + rec + 1e-10))
    best_ix = int(np.argmax(fscore))
    return float(thresh[best_ix]) if best_ix < len(thresh) else float(fallback)


def evaluate_probability_stream(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).flatten().astype(int)
    y_prob = np.asarray(y_prob).flatten()
    if len(y_true) != len(y_prob):
        n = min(len(y_true), len(y_prob))
        y_true = y_true[-n:]
        y_prob = y_prob[-n:]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "F1@threshold": float(f1_score(y_true, y_pred, zero_division=0)),
        "F1@0.5": float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold_used": float(threshold),
        "n_test": int(len(y_true)),
        "n_positive_test": int(y_true.sum()),
    }
