"""
lstm_utils.py
=============

Shared model-builder and data-preparation helpers used across notebooks
01–06. Extracted verbatim from the original LSTM_EnKF_v2.ipynb cells
(Parts 6, 8, 9). Reusing this module from every notebook means changes
flow to every consumer in one place, instead of by copy-paste.

Function origins:
    create_sequences_from_df  ← cell 17 (Part 6)
    build_lstm_model          ← cell 21 (Part 8)
    class_weights_powered     ← cell 27 (Part 9 tuning prep)
    build_tunable_lstm        ← cell 28 (Part 9 tuning prep)
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, Nadam
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    # Older TF versions expose AdamW under keras.optimizers.experimental
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.regularizers import l2 as l2_reg_fn
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall

# build_lstm_model() reads DEFAULT_HPS_DICT from shared_config; we import
# lazily to avoid a circular dependency if shared_config grows.
from shared_config import DEFAULT_HPS_DICT


# ---------------------------------------------------------------------------
# Sequence creation  (Part 6)
# ---------------------------------------------------------------------------
def create_sequences_from_df(df, seq_length, target_col_idx=0, pred_step=1,
                             return_targets=True, drop_target_from_x=True):
    """
    Creates sequences for an LSTM model from a DataFrame.

    The column at ``target_col_idx`` is the prediction target. When
    ``drop_target_from_x=True`` (default) that column is used ONLY to build
    ``y`` and is removed from the ``X`` feature channels, so the label never
    enters the model input. The remaining columns keep their original order.

    Set ``drop_target_from_x=False`` to recover the legacy behaviour in which
    every column — including the target — is part of ``X`` (this was the
    source of the bloom_target leakage; keep it False only for non-target
    autoregressive windowing).

    Note: the target column is only dropped when ``return_targets=True`` (i.e.
    when ``target_col_idx`` actually denotes a target). With
    ``return_targets=False`` all columns are retained.
    """
    xs = []
    ys = []

    # Convert dataframe to numpy if needed
    data = df.values if hasattr(df, 'values') else np.asarray(df)
    n_cols = data.shape[1] if data.ndim == 2 else 1

    # Decide which columns become X feature channels.
    if drop_target_from_x and return_targets and data.ndim == 2:
        feature_idx = [c for c in range(n_cols) if c != target_col_idx]
    else:
        feature_idx = None  # keep all columns

    # Loop through data
    # We stop earlier to account for the prediction step ahead (pred_step)
    for i in range(len(data) - seq_length - pred_step + 1):
        window = data[i:(i + seq_length)]
        xs.append(window[:, feature_idx] if feature_idx is not None else window)

        if return_targets:
            # Target is 'pred_step' steps after the sequence ends
            y = data[i + seq_length + pred_step - 1, target_col_idx]
            ys.append(y)

    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Default LSTM builder  (Part 8)
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape, hps=None):
    """Build a default LSTM. For tuned models use build_tunable_lstm directly.

    Args:
        input_shape: (seq_len, n_features)
        hps: optional dict-like; if provided uses those values, else defaults.
    """
    h = dict(DEFAULT_HPS_DICT)
    if hps is not None:
        # Accept HPsShim (.to_dict()), plain dict, or KerasTuner-style .get()
        if hasattr(hps, 'to_dict'):
            for k, v in hps.to_dict().items():
                h[k] = v
        elif isinstance(hps, dict):
            h.update(hps)
        else:
            for k in list(h.keys()):
                try:
                    v = hps.get(k)
                except Exception:
                    v = None
                if v is not None:
                    h[k] = v

    model = Sequential()
    model.add(Input(shape=input_shape))
    layer_sizes = [h['units_1'], h['units_2'], h.get('units_3', 16)]
    n_layers = int(h['n_lstm_layers'])
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        model.add(LSTM(units=layer_sizes[i], return_sequences=return_seq))
        model.add(Dropout(h['dropout']))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=h['learning_rate']),
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(name='accuracy'), AUC(name='auc'),
                 Recall(name='recall'), Precision(name='precision')]
    )
    return model


# --- Test Block ---


# ---------------------------------------------------------------------------
# Class-weighting helper  (Part 9 tuning prep)
# ---------------------------------------------------------------------------
def class_weights_powered(y, power=1.0):
    """
    Generalized class weighting. power=0 -> uniform; power=1 -> balanced;
    power>1 -> aggressively upweight minority class.
    """
    y = np.asarray(y).flatten().astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    n_total = len(y)
    n_classes = len(classes)
    # Standard balanced weight raised to the power
    base = n_total / (n_classes * counts)
    weights = base ** power
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Tunable LSTM builder  (Part 9 tuning prep)
# ---------------------------------------------------------------------------
def build_tunable_lstm(input_shape, hps):
    """
    Build LSTM with all tunable knobs exposed via the hps dict.

    hps keys used: n_lstm_layers, units_1, units_2, units_3,
                   dropout, recurrent_dropout, l2_reg,
                   optimizer, learning_rate, weight_decay
    """
    reg = l2_reg_fn(hps['l2_reg']) if hps['l2_reg'] > 1e-7 else None

    inp = Input(shape=input_shape)
    x = inp
    layer_sizes = [hps['units_1'], hps['units_2'], hps.get('units_3', 16)]
    for i in range(hps['n_lstm_layers']):
        return_seq = (i < hps['n_lstm_layers'] - 1)
        x = LSTM(
            layer_sizes[i],
            return_sequences=return_seq,
            dropout=0.0,                     # apply explicit Dropout layers instead
            recurrent_dropout=hps['recurrent_dropout'],
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
        )(x)
        x = Dropout(hps['dropout'])(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)

    # Optimizer
    if hps['optimizer'] == 'adamw':
        opt = AdamW(learning_rate=hps['learning_rate'], weight_decay=hps['weight_decay'])
    elif hps['optimizer'] == 'nadam':
        opt = Nadam(learning_rate=hps['learning_rate'])
    else:
        opt = Adam(learning_rate=hps['learning_rate'])

    model.compile(
        optimizer=opt,
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(name='accuracy'), AUC(name='auc'),
                 AUC(name='auprc', curve='PR'),
                 Precision(name='precision'), Recall(name='recall')],
    )
    return model

