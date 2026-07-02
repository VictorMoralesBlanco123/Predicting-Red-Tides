"""
onset_utils.py
==============
Onset-focused targets, metrics, losses, and feature engineering for the
LSTM+EnKF red-tide forecasting pipeline.

This module is additive: it depends on the existing shared_config and
pipeline_utils helpers but does not modify them. All functions are designed
to be called from any notebook in the pipeline after the standard artifacts
have been written, or from the model-fitting cells of notebooks 02/03 when
retraining is desired.

Conventions
-----------
- ``y_binary`` arrays are 1-D per-timestep bloom indicators (bloom_target).
- ``y_prob`` arrays are 1-D probability streams in [0, 1] aligned with the
  test sequences produced by ``pipeline_utils.build_sequence_artifact``.
- Tolerance windows are expressed in weeks (one timestep == one week).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Tuple


# ---------------------------------------------------------------------------
# Onset target construction
# ---------------------------------------------------------------------------

def make_onset_target(y_binary, buffer_weeks: int = 4) -> np.ndarray:
    """Define an onset as a positive timestep preceded by ``buffer_weeks`` of
    consecutive non-bloom weeks.

    Parameters
    ----------
    y_binary : array-like of {0, 1}
        Per-timestep bloom indicator.
    buffer_weeks : int
        Number of preceding bloom-free weeks required to count as an onset.

    Returns
    -------
    np.ndarray of {0, 1}
        1 only on onset weeks; 0 on no-bloom and on continuation weeks.
    """
    y = np.asarray(y_binary).astype(int).flatten()
    onset = np.zeros_like(y)
    for t in range(buffer_weeks, len(y)):
        if y[t] == 1 and y[t - buffer_weeks:t].sum() == 0:
            onset[t] = 1
    return onset


def onset_event_indices(onset_target) -> np.ndarray:
    """Integer indices of onset events (where onset_target == 1)."""
    return np.where(np.asarray(onset_target) == 1)[0]


# ---------------------------------------------------------------------------
# Event-based skill metrics with a tolerance window
# ---------------------------------------------------------------------------

def _windowed_hits_and_alarms(
    true_events: Iterable[int],
    pred_events: Iterable[int],
    window: int,
) -> Tuple[int, int, int]:
    """Greedy nearest-match assignment of predicted events to true events
    within +/- ``window`` timesteps. Each predicted event can match at most
    one true event and vice versa.

    Returns
    -------
    (hits, misses, false_alarms)
    """
    true_events = sorted(int(e) for e in true_events)
    pred_events = sorted(int(p) for p in pred_events)

    used_preds: set = set()
    hits = 0
    for t in true_events:
        best = None
        best_dist = window + 1
        for i, p in enumerate(pred_events):
            if i in used_preds:
                continue
            d = abs(p - t)
            if d <= window and d < best_dist:
                best = i
                best_dist = d
        if best is not None:
            used_preds.add(best)
            hits += 1

    misses = len(true_events) - hits
    false_alarms = len(pred_events) - len(used_preds)
    return hits, misses, false_alarms


def event_based_metrics(y_true_onset, y_pred_binary, window: int = 2) -> Dict:
    """Event-based POD, FAR, CSI, HSS, F1, and frequency bias on onsets.

    A predicted positive within +/- ``window`` timesteps of a true onset
    counts as a hit. Predictions and onset timesteps are matched greedily
    to the nearest available counterpart.

    Parameters
    ----------
    y_true_onset : array-like of {0, 1}
        Onset indicator from ``make_onset_target``.
    y_pred_binary : array-like of {0, 1}
        Hard predictions from the model after thresholding.
    window : int
        Tolerance window in weeks.
    """
    y_true = np.asarray(y_true_onset).astype(int).flatten()
    y_pred = np.asarray(y_pred_binary).astype(int).flatten()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]

    true_events = onset_event_indices(y_true)
    pred_events = np.where(y_pred == 1)[0]

    hits, misses, fa = _windowed_hits_and_alarms(true_events, pred_events, window)

    pod = hits / max(hits + misses, 1)
    far = fa / max(hits + fa, 1)
    csi = hits / max(hits + misses + fa, 1)

    # Heidke skill score against random forecast with same marginals
    n_total = max(len(y_true), 1)
    a = hits
    b = fa
    c = misses
    d = n_total - a - b - c
    numerator = 2.0 * (a * d - b * c)
    denominator = (a + c) * (c + d) + (a + b) * (b + d)
    hss = numerator / denominator if denominator > 0 else float("nan")

    f1 = 2 * hits / max(2 * hits + fa + misses, 1)
    bias = (hits + fa) / max(hits + misses, 1)

    return {
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(fa),
        "n_true_events": int(len(true_events)),
        "n_pred_events": int(len(pred_events)),
        "POD": float(pod),
        "FAR": float(far),
        "CSI": float(csi),
        "HSS": float(hss),
        "event_F1": float(f1),
        "frequency_bias": float(bias),
        "tolerance_window": int(window),
    }


# ---------------------------------------------------------------------------
# Lead-time analysis
# ---------------------------------------------------------------------------

def onset_lead_times(y_true_onset, y_pred_binary, window: int = 4) -> np.ndarray:
    """For each true onset, return the lead time (weeks) of the earliest
    positive prediction within +/- ``window`` timesteps. Positive values
    mean the model predicted *before* the true onset (the desired case);
    negative values mean late detection; NaN means missed.
    """
    y_true = np.asarray(y_true_onset).astype(int).flatten()
    y_pred = np.asarray(y_pred_binary).astype(int).flatten()
    n = min(len(y_true), len(y_pred))
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]

    leads = []
    for t in onset_event_indices(y_true):
        lo = max(0, t - window)
        hi = min(n, t + window + 1)
        local_preds = np.where(y_pred[lo:hi] == 1)[0]
        if len(local_preds) == 0:
            leads.append(np.nan)
        else:
            earliest_abs = lo + local_preds[0]
            leads.append(t - earliest_abs)
    return np.array(leads, dtype=float)


# ---------------------------------------------------------------------------
# Onset-aware threshold selection
# ---------------------------------------------------------------------------

def best_onset_csi_threshold(
    y_prob,
    y_binary,
    buffer_weeks: int = 4,
    window: int = 2,
    fallback: float = 0.5,
    n_steps: int = 200,
) -> Tuple[float, Dict]:
    """Sweep probability thresholds and return the value that maximizes
    event-based CSI on onset events. Returns (threshold, metrics_dict).
    """
    y_prob = np.asarray(y_prob).flatten()
    y_binary = np.asarray(y_binary).astype(int).flatten()
    n = min(len(y_prob), len(y_binary))
    y_prob = y_prob[-n:]
    y_binary = y_binary[-n:]

    y_onset = make_onset_target(y_binary, buffer_weeks=buffer_weeks)
    if y_onset.sum() == 0:
        return float(fallback), {"CSI": float("nan"), "n_true_events": 0}

    lo = max(float(np.min(y_prob)), 1e-4)
    hi = min(float(np.max(y_prob)), 1.0 - 1e-4)
    if hi <= lo:
        return float(fallback), {"CSI": float("nan"), "n_true_events": int(y_onset.sum())}
    thresholds = np.linspace(lo, hi, n_steps)

    best_t = float(fallback)
    best_csi = -1.0
    best_metrics: Dict = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = event_based_metrics(y_onset, y_pred, window=window)
        if m["CSI"] > best_csi:
            best_csi = m["CSI"]
            best_t = float(t)
            best_metrics = m

    best_metrics["threshold"] = best_t
    return best_t, best_metrics


def best_onset_hss_threshold(
    y_prob,
    y_binary,
    buffer_weeks: int = 4,
    window: int = 2,
    fallback: float = 0.5,
    n_steps: int = 200,
) -> Tuple[float, Dict]:
    """Same as ``best_onset_csi_threshold`` but optimizes HSS instead of
    CSI. Useful as a sensitivity check on the operating-point choice.
    """
    y_prob = np.asarray(y_prob).flatten()
    y_binary = np.asarray(y_binary).astype(int).flatten()
    n = min(len(y_prob), len(y_binary))
    y_prob = y_prob[-n:]
    y_binary = y_binary[-n:]

    y_onset = make_onset_target(y_binary, buffer_weeks=buffer_weeks)
    if y_onset.sum() == 0:
        return float(fallback), {"HSS": float("nan"), "n_true_events": 0}

    thresholds = np.linspace(
        max(float(np.min(y_prob)), 1e-4),
        min(float(np.max(y_prob)), 1.0 - 1e-4),
        n_steps,
    )
    best_t = float(fallback)
    best_hss = -np.inf
    best_metrics: Dict = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        m = event_based_metrics(y_onset, y_pred, window=window)
        if np.isfinite(m["HSS"]) and m["HSS"] > best_hss:
            best_hss = m["HSS"]
            best_t = float(t)
            best_metrics = m

    best_metrics["threshold"] = best_t
    return best_t, best_metrics


# ---------------------------------------------------------------------------
# Sample weighting for training (use with model.fit(sample_weight=...))
# ---------------------------------------------------------------------------

def onset_sample_weights(
    y_binary,
    onset_multiplier: float = 5.0,
    pre_onset_multiplier: float = 3.0,
    pre_onset_window: int = 4,
    buffer_weeks: int = 4,
    class_weight: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Per-timestep sample weights that upweight onset and pre-onset samples
    on top of class weights.

    Parameters
    ----------
    y_binary : array-like
        Training-time bloom indicator aligned with sequence targets.
    onset_multiplier : float
        Multiplier applied to onset timesteps.
    pre_onset_multiplier : float
        Multiplier applied to the ``pre_onset_window`` timesteps immediately
        preceding an onset (these are the high-value lead-time samples).
    pre_onset_window : int
    buffer_weeks : int
        Passed through to ``make_onset_target``.
    class_weight : dict or None
        Optional sklearn-style {0: w0, 1: w1} dict. If provided, weights are
        initialised from class weights before applying onset multipliers.

    Returns
    -------
    np.ndarray of float32 with shape (len(y_binary),)
    """
    y = np.asarray(y_binary).astype(int).flatten()
    if class_weight is None:
        weights = np.ones_like(y, dtype=float)
    else:
        w0 = float(class_weight.get(0, 1.0))
        w1 = float(class_weight.get(1, 1.0))
        weights = np.where(y == 1, w1, w0).astype(float)

    onset = make_onset_target(y, buffer_weeks=buffer_weeks)
    onset_idx = onset_event_indices(onset)

    weights[onset_idx] *= onset_multiplier
    for t in onset_idx:
        lo = max(0, t - pre_onset_window)
        # pre-onset weeks are by construction non-bloom, so this lifts the
        # negative-class samples that are most informative about onset.
        weights[lo:t] *= pre_onset_multiplier

    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Focal loss (TF/Keras-compatible)
# ---------------------------------------------------------------------------

def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.75, from_logits: bool = False):
    """Keras-compatible binary focal loss callable.

    Use as ``model.compile(loss=binary_focal_loss(gamma=2.0, alpha=0.75), ...)``.

    ``alpha`` balances classes (alpha > 0.5 favors the positive class);
    ``gamma`` downweights easy examples. With persistent bloom weeks
    dominating the gradient under standard BCE, gamma=2.0 and alpha=0.75
    are a sensible starting point for onset-sensitive training.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required for binary_focal_loss.") from exc

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if from_logits:
            y_pred = tf.sigmoid(y_pred)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return -tf.reduce_mean(alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

    loss.__name__ = f"binary_focal_loss_g{gamma}_a{alpha}"
    return loss


# ---------------------------------------------------------------------------
# Onset-relevant feature engineering
# ---------------------------------------------------------------------------

def alongshore_wind(wind_u, wind_v, shelf_angle_deg: float = 340.0) -> np.ndarray:
    """Project (u, v) wind components onto the alongshore axis of the West
    Florida Shelf. Default ``shelf_angle_deg`` = 340 deg (NNW), broadly
    consistent with the orientation used in Weisberg et al. for the WFS.

    With the meteorological (u, v) = (east-component, north-component)
    convention, this returns the wind component projected onto the
    along-shelf direction defined by the chosen angle. Positive values
    indicate poleward (upcoast) flow, which is associated with upwelling
    favorable to bloom initiation in SW Florida.
    """
    theta = np.deg2rad(float(shelf_angle_deg))
    u = np.asarray(wind_u, dtype=float)
    v = np.asarray(wind_v, dtype=float)
    return u * np.sin(theta) + v * np.cos(theta)


def add_rate_of_change_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    windows: Iterable[int] = (2, 4, 8),
    suffix: str = "d",
) -> pd.DataFrame:
    """Append (current - lagged) delta features for each column / window
    combination. Returns a new DataFrame; the input is not modified.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for w in windows:
            out[f"{c}_{suffix}{w}"] = out[c] - out[c].shift(w)
    return out


def add_cumulative_anomaly_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    climatology_period: int = 52,
    accumulation_windows: Iterable[int] = (4, 8, 12),
    causal: bool = True,
) -> pd.DataFrame:
    """Append running-sum anomaly features computed against a seasonal
    climatology estimated by week-of-year. ``climatology_period`` is the
    expected seasonal length in timesteps (52 weeks for weekly data).

    When ``causal=True`` (default) the per-week-of-year climatology at row t is
    the expanding mean over that bucket using only current-and-earlier rows.
    This makes the feature strictly causal, so estimating it on the full
    contiguous series before the train/val/test split introduces no test-into-
    train leakage. (The first occurrence of each bucket has anomaly 0, which is
    the same kind of leading-window edge effect the lag features already incur.)

    Set ``causal=False`` to recover the legacy whole-series bucket mean, which
    uses future rows to estimate the climatology and therefore leaks across the
    split — kept only for reproducing the old behaviour.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    woy = np.arange(n) % climatology_period
    for c in cols:
        if c not in out.columns:
            continue
        if causal:
            # Expanding (causal) bucket mean: uses only past-and-current rows.
            clim = out[c].groupby(woy).transform(lambda s: s.expanding().mean())
        else:
            clim = out.groupby(woy)[c].transform("mean")
        anom = out[c] - clim
        for w in accumulation_windows:
            out[f"{c}_anom_sum{w}"] = anom.rolling(window=w, min_periods=1).sum()
    return out


def weeks_since_last_bloom(y_binary) -> np.ndarray:
    """Counter of weeks since the most recent bloom timestep. Returns NaN
    for the leading window before any bloom has been observed.
    """
    y = np.asarray(y_binary).astype(int).flatten()
    out = np.zeros_like(y, dtype=float)
    counter = 0
    seen = False
    for i, v in enumerate(y):
        if v == 1:
            counter = 0
            seen = True
            out[i] = 0
        else:
            if seen:
                counter += 1
                out[i] = counter
            else:
                out[i] = np.nan
    return out


# ---------------------------------------------------------------------------
# Probabilistic onset alerts using ensemble outputs
# ---------------------------------------------------------------------------

def ensemble_onset_alert(
    mc_predictions: np.ndarray,
    threshold: float = 0.5,
    alert_fraction: float = 0.3,
    horizon: int = 4,
) -> np.ndarray:
    """Convert a 3-D ensemble of MC-dropout predictions into a binary onset
    alert at each timestep.

    Parameters
    ----------
    mc_predictions : ndarray with shape (n_test, n_enkf, n_mc)
        The 3-D tensor produced by notebook 04b. If a 2-D (n_test, n_samples)
        array is passed it is treated as a single ensemble axis.
    threshold : float
        Per-member probability threshold for ``bloom`` declaration.
    alert_fraction : float
        Fraction of ensemble members required to declare an alert at a given
        timestep. Lower values are more aggressive (higher POD, higher FAR).
    horizon : int
        Number of forward timesteps over which to aggregate via a forward
        rolling OR. Set to 0 for a per-timestep alert.

    Returns
    -------
    np.ndarray of {0, 1} with shape (n_test,)
    """
    preds = np.asarray(mc_predictions)
    if preds.ndim == 3:
        flat = preds.reshape(preds.shape[0], -1)
    elif preds.ndim == 2:
        flat = preds
    else:
        raise ValueError("mc_predictions must be 2-D or 3-D.")

    member_pos = (flat >= float(threshold)).astype(int)
    fraction = member_pos.mean(axis=1)
    per_step_alert = (fraction >= float(alert_fraction)).astype(int)

    if horizon <= 0:
        return per_step_alert

    n = len(per_step_alert)
    out = np.zeros_like(per_step_alert)
    for i in range(n):
        out[i] = int(per_step_alert[i:i + horizon + 1].any())
    return out
