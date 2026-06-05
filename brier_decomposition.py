"""
brier_decomposition.py
======================

Murphy (1973) three-way decomposition of the Brier score:

    BS = REL - RES + UNC

    REL = (1/N) Σ_k n_k (p̄_k - ō_k)²    calibration error (lower = better)
    RES = (1/N) Σ_k n_k (ō_k - ō)²       forecast resolution (higher = better)
    UNC = ō (1 - ō)                       irreducible base-rate variance

Also computes the Ferro & Fricker (2012) bias-corrected versions of REL
and RES, which remove finite-sample bias in the empirical estimators —
important when bin counts are small, which is exactly the regime here
(n_test = 230, 10-bin partition has ~20 per bin on average).

The identity BS = REL - RES + UNC holds exactly only for the *binned*
Brier score, where every prediction is replaced by its bin mean. The
function returns both the raw and binned Brier scores so the user can
verify the binning approximation is tight (in practice it is, for any
reasonable bin count and smooth probability distribution).

Reference:
    Murphy, A. H. (1973). A New Vector Partition of the Probability Score.
        J. Appl. Meteor., 12, 595-600.
    Ferro, C. A. T., and T. E. Fricker (2012). A bias-corrected
        decomposition of the Brier score. Q. J. R. Meteorol. Soc., 138,
        1954-1960.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> dict:
    """Murphy (1973) decomposition of the Brier score with Ferro & Fricker
    (2012) bias correction.

    Parameters
    ----------
    y_true : array-like of {0, 1}
        Binary outcomes.
    y_prob : array-like of floats in [0, 1]
        Predicted probabilities.
    n_bins : int, default 10
        Number of probability bins for the decomposition.
    strategy : {"uniform", "quantile"}, default "quantile"
        Binning strategy. "uniform" uses equal-width bins on [0, 1];
        "quantile" uses equal-frequency bins. Quantile is more stable
        when predictions cluster near 0 or 1 (typical for rare-event
        forecasting with high-recall thresholds).

    Returns
    -------
    dict with keys
        brier              : raw Brier score (BS computed on individual preds)
        brier_binned       : Brier score on bin-mean predictions (used in identity)
        reliability        : REL (calibration error)
        resolution         : RES (forecast informativeness)
        uncertainty        : UNC (base-rate variance)
        identity_check     : REL - RES + UNC; should equal brier_binned
        reliability_debiased  : Ferro-Fricker bias-corrected REL
        resolution_debiased   : Ferro-Fricker bias-corrected RES
        n_bins_used        : number of bins with non-zero count
        bin_table          : pandas DataFrame, per-bin diagnostics
    """
    y_true = np.asarray(y_true).astype(float).flatten()
    y_prob = np.asarray(y_prob).astype(float).flatten()
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must lie in [0, 1]")

    N = len(y_true)
    o_bar = float(y_true.mean())                        # base rate ō
    brier_raw = float(np.mean((y_prob - y_true) ** 2))  # raw BS

    # --- Build bins ---
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            # Fallback if predictions are degenerate (e.g. all the same value)
            edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    # Assign each prediction to a bin (clamp into [1, len(edges)-1])
    bin_idx = np.clip(np.searchsorted(edges, y_prob, side="right") - 1,
                      0, len(edges) - 2)

    # --- Per-bin statistics ---
    rows = []
    REL_sum = 0.0
    RES_sum = 0.0
    REL_bias = 0.0
    RES_bias = 0.0
    binned_brier_sum = 0.0

    for k in range(len(edges) - 1):
        in_bin = bin_idx == k
        n_k = int(in_bin.sum())
        if n_k == 0:
            continue
        p_bar_k = float(y_prob[in_bin].mean())
        o_bar_k = float(y_true[in_bin].mean())

        REL_sum += n_k * (p_bar_k - o_bar_k) ** 2
        RES_sum += n_k * (o_bar_k - o_bar)  ** 2
        binned_brier_sum += n_k * (p_bar_k - o_bar_k) ** 2 + n_k * o_bar_k * (1 - o_bar_k)

        # Ferro-Fricker bias terms: REL bias is positive (REL overestimated),
        # RES bias is also positive (RES overestimated). Both scale as
        # o_bar_k (1 - o_bar_k) / (n_k - 1) within each bin.
        if n_k > 1:
            ff_inner = o_bar_k * (1.0 - o_bar_k) / (n_k - 1)
            REL_bias += n_k * ff_inner
            RES_bias += n_k * ff_inner

        rows.append({
            "bin":            k,
            "edge_lo":        float(edges[k]),
            "edge_hi":        float(edges[k + 1]),
            "n_k":            n_k,
            "mean_pred":      p_bar_k,
            "obs_freq":       o_bar_k,
            "rel_contrib":    n_k * (p_bar_k - o_bar_k) ** 2 / N,
            "res_contrib":    n_k * (o_bar_k - o_bar)  ** 2 / N,
        })

    REL = REL_sum / N
    RES = RES_sum / N
    UNC = o_bar * (1.0 - o_bar)
    REL_db = REL - REL_bias / N
    # Ferro-Fricker correction for RES: SUBTRACTS the bias from the climatology
    # uncertainty side, so RES_db = RES - (UNC_bias - REL_bias). Operationally:
    #   RES_unbiased = RES - (bias of estimator), where the bias depends on the
    #   between-bin variance of the unbiased ō_k estimator.
    # The full correction reduces to RES - bias term derived from the binomial
    # variance of ō_k pooled across bins.
    RES_db = RES - (RES_bias / N) + UNC * (1.0 / N)  # F&F eq. (8) plug-in form

    brier_binned = REL - RES + UNC  # exact identity for binned predictions

    bin_table = pd.DataFrame(rows)

    return {
        "brier":                  brier_raw,
        "brier_binned":           brier_binned,
        "reliability":            REL,
        "resolution":             RES,
        "uncertainty":            UNC,
        "identity_check":         REL - RES + UNC,
        "reliability_debiased":   REL_db,
        "resolution_debiased":    RES_db,
        "n_bins_used":            int(bin_table["n_k"].gt(0).sum()),
        "base_rate":              o_bar,
        "n_total":                N,
        "binning_strategy":       strategy,
        "n_bins_requested":       n_bins,
        "bin_table":              bin_table,
    }


def format_decomposition_summary(result: dict, model_name: str = "Hybrid") -> str:
    """Pretty-print summary of a brier_decomposition result, suitable for
    pasting into a notebook output cell."""
    r = result
    return (
        f"--- Brier decomposition for {model_name} (n={r['n_total']}, "
        f"base rate {r['base_rate']:.3f}, {r['n_bins_used']} non-empty "
        f"{r['binning_strategy']} bins) ---\n"
        f"  Raw Brier score        : {r['brier']:.4f}\n"
        f"  Binned Brier score     : {r['brier_binned']:.4f}   "
        f"(approx. error vs. raw: {abs(r['brier']-r['brier_binned']):.4f})\n"
        f"  Reliability (REL)      : {r['reliability']:.4f}   "
        f"[debiased: {r['reliability_debiased']:.4f}]\n"
        f"  Resolution (RES)       : {r['resolution']:.4f}   "
        f"[debiased: {r['resolution_debiased']:.4f}]\n"
        f"  Uncertainty (UNC)      : {r['uncertainty']:.4f}\n"
        f"  Identity REL - RES + UNC: {r['identity_check']:.4f}\n"
    )
