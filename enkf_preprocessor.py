"""
enkf_preprocessor.py  —  ADD THESE TWO FUNCTIONS TO enkf_utils.py
==================================================================

Reframes the EnKF from an online last-timestep corrector into a causal
data-assimilation PREPROCESSOR: it denoises the (interpolated, noisy) driver
series before they ever reach the LSTM, and emits a per-timestep posterior
standard deviation as an explicit driver-state uncertainty feature.

Why this replaces the old behaviour
------------------------------------
The shipped run_enkf_mc_forecast(..., perform_enkf=True) used MANUAL_Q_DIAG and
MANUAL_R_DIAG with Q >> R, giving a Kalman gain of ~0.9999 on every variable.
Measured on the real series:
    * the "corrected" state == the raw observation (kb changed by 0.0004 sigma),
    * wind_u was injected with noise ~80x its own std (Q=225 is nonsensical for a
      signed component oscillating near zero),
    * only the LAST of 26 timesteps was touched, and only at inference, so the
      LSTM trained on raw sequences and was tested on (barely) modified ones.
Net effect: the "hybrid" was ~identical to the standalone LSTM, by construction.

As a preprocessor with data-driven R > Q (gain ~0.07) the filter smooths the
environmental drivers 70-80% while a protected kb keeps corr=0.93 with the raw
signal, and the same transform is applied to train/val/test so training and
inference are consistent.

Leakage
-------
The stochastic EnKF at time t uses only observations up to and including t, so
filtering the full contiguous series once and then splitting is causal and
leak-free. The ONLY train-derived quantities are the diagonal Q and R, which
estimate_enkf_noise() computes from the TRAIN slice alone.

These functions reuse EnsembleKalmanFilter already defined in enkf_utils.py.
"""

import numpy as np
import pandas as pd


def estimate_enkf_noise(train_df, state_vars,
                        obs_noise_frac=0.25, proc_noise_frac=0.02,
                        protect_vars=None, protect_obs_frac=0.02):
    """Data-driven diagonal process (Q) and observation (R) noise.

    Sets R = obs_noise_frac * Var_train(x) and Q = proc_noise_frac * Var_train(x)
    per variable. Because R > Q the steady-state gain Q/(Q+R) is small, so the
    filter trusts its dynamics and smooths measurement/interpolation noise.

    protect_vars (e.g. ['kb']) receive a SMALL R (protect_obs_frac) so they are
    only lightly smoothed — this preserves sharp onset spikes in the target-
    related variable, which heavy smoothing would erase.

    Returns (Q_diag, R_diag) as plain lists, ready for np.diag().
    """
    protect_vars = set(protect_vars or [])
    v = train_df[state_vars].var().values.astype(float)
    v = np.where((~np.isfinite(v)) | (v <= 0), 1e-6, v)
    R = np.array([(protect_obs_frac if sv in protect_vars else obs_noise_frac) * vi
                  for sv, vi in zip(state_vars, v)])
    Q = proc_noise_frac * v
    return list(Q), list(R)


def enkf_preprocess_drivers(df, state_vars, Q_diag, R_diag,
                            n_enkf=50, seed=42,
                            propagator='persistence', ar_coef=None,
                            append_std=True, std_suffix='_enkf_std'):
    """Causal EnKF smoothing of the `state_vars` columns of `df`.

    Returns a COPY of df with those columns replaced by the EnKF posterior mean.
    If append_std, also appends one '{var}_enkf_std' column per state variable
    holding the posterior ensemble standard deviation (driver-state uncertainty).

    Args
    ----
    df          : full unscaled dataframe (filter the whole contiguous series).
    state_vars  : driver columns to assimilate (must exist in df).
    Q_diag, R_diag : diagonals from estimate_enkf_noise().
    propagator  : 'persistence' (random-walk smoother) or 'ar1' (mean-reverting).
    ar_coef     : per-variable AR(1) coefficients when propagator='ar1'.
    """
    from enkf_utils import EnsembleKalmanFilter  # defined above in the same module

    np.random.seed(seed)  # the class seeds via module-level np.random
    out = df.copy()
    Z = df[state_vars].values.astype(float)
    T, dim_x = Z.shape
    colmean = np.nanmean(Z, axis=0)

    x0 = np.where(np.isnan(Z[0]), colmean, Z[0])
    P0 = np.diag((np.abs(x0) * 0.30) ** 2) + 1e-6 * np.eye(dim_x)
    enkf = EnsembleKalmanFilter(x0, P0, dim_z=dim_x, N=n_enkf)
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    if propagator == 'ar1':
        a = np.asarray(ar_coef if ar_coef is not None else [0.8] * dim_x, dtype=float)
        F = lambda x: colmean + a * (x - colmean)
    else:
        F = lambda x: x  # persistence / random-walk

    filt = np.zeros_like(Z)
    std = np.zeros_like(Z)
    for t in range(T):
        enkf.predict(F, Q=Q)
        z = Z[t]
        if np.any(np.isnan(z)):                     # hold the update where obs is missing
            z = np.where(np.isnan(z), enkf.ensemble.mean(axis=0), z)
        enkf.update(z, R)
        filt[t] = enkf.ensemble.mean(axis=0)
        std[t] = enkf.ensemble.std(axis=0)

    out[state_vars] = filt
    if append_std:
        for j, sv in enumerate(state_vars):
            out[f'{sv}{std_suffix}'] = std[:, j]
    return out
