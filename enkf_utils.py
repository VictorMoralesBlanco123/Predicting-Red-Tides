"""
enkf_utils.py
=============

Shared Ensemble Kalman Filter and MC-dropout forecasting code, used by
notebook 04 (forecasting) and notebook 05 (Part 19 sensitivity sweep).
Extracted verbatim from the original LSTM_EnKF_v2.ipynb cells (Parts 13–14).

Function/class origins:
    EnsembleKalmanFilter      ← cell 49 (Part 13)
    setup_enkf                ← cell 49 (Part 13)
    run_enkf_mc_forecast      ← cell 51 (Part 14)
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.metrics import precision_recall_curve

from shared_config import ENKF_STATE_VARS

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


# ---------------------------------------------------------------------------
# Ensemble Kalman Filter + setup helper  (Part 13)
# ---------------------------------------------------------------------------
class EnsembleKalmanFilter:
    """
    Stochastic Ensemble Kalman Filter (Burgers et al., 1998).
    Includes observation perturbation to maintain ensemble variance.
    """
    def __init__(self, x_init, P_init, dim_z, N):
        """
        Args:
            x_init: Initial state mean (dim_x,)
            P_init: Initial state covariance (dim_x, dim_x)
            dim_z:  Dimension of observations
            N:      Number of ensemble members
        """
        self.dim_x = len(x_init)
        self.dim_z = dim_z
        self.N = N
        self.x = None # Current state mean
        self.P = None # Current state covariance

        # Initialize Ensemble
        # We enforce symmetry and positive-definiteness on P_init
        P_init = (P_init + P_init.T) / 2
        self.ensemble = self._multivariate_normal(x_init, P_init, N)

        # Calculate initial stats
        self.x = np.mean(self.ensemble, axis=0)
        self.P = np.cov(self.ensemble.T)

        print(f"EnKF Initialized. Members: {N}, State Dim: {self.dim_x}")

    def predict(self, F_func, Q, dt=1):
        """
        Propagate state forward: x = F(x) + Noise
        """
        # 1. Apply Model Dynamics (The "Physics" or "AI")
        # We apply F_func to every member of the ensemble
        self.ensemble = np.apply_along_axis(F_func, 1, self.ensemble)

        # 2. Add Process Noise (System Error)
        if Q is not None:
            Q = (Q + Q.T) / 2
            noise = self._multivariate_normal(np.zeros(self.dim_x), Q, self.N)
            self.ensemble += noise

        # Update stats
        self.x = np.mean(self.ensemble, axis=0)
        self.P = np.cov(self.ensemble.T)

    def update(self, z, R, H=None):
        """
        Assimilate observation: x = x + K(z - Hx)
        """
        if H is None: H = np.eye(self.dim_z, self.dim_x)

        # 1. Project Ensemble to Observation Space
        # Shape: (N, dim_z)
        Hx = self.ensemble @ H.T

        # 2. Perturb Observations (CRITICAL for Stochastic EnKF)
        # We treat the observation as a random variable, not a single truth
        R = (R + R.T) / 2
        obs_noise = self._multivariate_normal(np.zeros(self.dim_z), R, self.N)
        z_perturbed = z + obs_noise

        # 3. Calculate Innovation (Residual)
        # Difference between "noisy measurement" and "predicted measurement"
        D = z_perturbed - Hx

        # 4. Calculate Kalman Gain (K)
        # P_zz = H P H' + R
        # K = P H' (P_zz)^-1
        P_prior = np.cov(self.ensemble.T)
        P_zz = H @ P_prior @ H.T + R

        # Use Pseudo-Inverse for stability
        K = P_prior @ H.T @ np.linalg.pinv(P_zz)

        # 5. Update Ensemble
        # x_new = x_old + K * Innovation
        self.ensemble = self.ensemble + (D @ K.T)

        # Update stats
        self.x = np.mean(self.ensemble, axis=0)
        self.P = np.cov(self.ensemble.T)

        return self.x, self.P

    def _multivariate_normal(self, mean, cov, size):
        """Helper to sample safely, adding jitter if matrix is singular."""
        try:
            return np.random.multivariate_normal(mean, cov, size)
        except np.linalg.LinAlgError:
            # Add small "jitter" to diagonal to fix numerical instability
            print("Warning: Matrix not positive definite. Adding jitter.")
            epsilon = 1e-6 * np.eye(len(mean))
            return np.random.multivariate_normal(mean, cov + epsilon, size)


# --- 2. Setup Function (Robust with Manual Overrides) ---
def setup_enkf(enkf_state_vars, n_enkf, train_df, test_df_unscaled,
               seq_len, feature_columns, manual_r_diag=None, manual_q_diag=None):
    """
    Configures the EnKF.
    Allows manual override of Q (Process) and R (Observation) noise diagonals.
    """
    print("\n--- Setting up EnKF ---")

    # Validation
    if not all(col in test_df_unscaled.columns for col in enkf_state_vars):
        print(f"Error: Missing state variables in test data.")
        return None, None, None, None

    dim_x = len(enkf_state_vars)

    # A. Initial State (x0)
    start_idx = seq_len - 1
    x_init = test_df_unscaled[enkf_state_vars].iloc[start_idx].values.astype(float)
    print(f"Initial State Vector (t=0): {x_init}")

    # B. Initial Covariance (P0)
    P_init = np.diag((np.abs(x_init) * 0.30) ** 2) + 1e-6 * np.eye(dim_x)

    # C. Noise Matrices (Q and R)

    # --- R MATRIX (Observation Noise) ---
    if manual_r_diag is not None:
        if len(manual_r_diag) != dim_x:
            print(f"Error: Manual R diag length ({len(manual_r_diag)}) does not match state vars ({dim_x})")
            return None, None, None, None
        print("-> Using MANUAL R Matrix values.")
        R_matrix = np.diag(manual_r_diag)
    else:
        print("-> Calculating Automatic R Matrix (1% of mean).")
        r_diags = []
        for col in enkf_state_vars:
            mean_val = train_df[col].mean()
            r_diags.append((abs(mean_val) * 0.01) ** 2)
        R_matrix = np.diag(r_diags)

    # --- Q MATRIX (Process Noise) ---
    if manual_q_diag is not None:
        if len(manual_q_diag) != dim_x:
            print(f"Error: Manual Q diag length ({len(manual_q_diag)}) does not match state vars ({dim_x})")
            return None, None, None, None
        print("-> Using MANUAL Q Matrix values.")
        Q_matrix = np.diag(manual_q_diag)
    else:
        print("-> Calculating Automatic Q Matrix (based on volatility).")
        q_diags = []
        for col in enkf_state_vars:
            volatility = train_df[col].diff().std()
            if pd.isna(volatility) or volatility == 0: volatility = 1e-3
            q_diags.append((volatility * 2.0) ** 2)
        Q_matrix = np.diag(q_diags)

    print("Noise Matrices Finalized:")
    print(f"  R (Obs Noise) diag: {np.diag(R_matrix)}")
    print(f"  Q (Proc Noise) diag: {np.diag(Q_matrix)}")

    # D. Feature Indices
    try:
        enkf_indices = [feature_columns.index(var) for var in enkf_state_vars]
    except ValueError as e:
        print(f"Error mapping variables to features: {e}")
        return None, None, None, None

    # E. Create Instance
    enkf = EnsembleKalmanFilter(x_init, P_init, dim_z=dim_x, N=n_enkf)
    enkf.R = R_matrix
    enkf.Q = Q_matrix

    return enkf, enkf_indices, R_matrix, Q_matrix

# --- 3. Execute ---


# ---------------------------------------------------------------------------
# EnKF + MC dropout forecasting loop  (Part 14)
# ---------------------------------------------------------------------------
def run_enkf_mc_forecast(model, enkf_instance, enkf_indices, R_matrix, Q_matrix,
                         x_test, y_test, test_df_unscaled, scaler,
                         seq_len, perform_enkf=True, perform_mc=True, n_mc_samples=50):
    """
    Optimized Forecasting Loop with Vectorized MC Dropout.
    """
    # 1. Validation & Setup
    if model is None or x_test is None: return None, None
    print(f"\n--- Starting Forecast (EnKF={perform_enkf}, MC={perform_mc}) ---")

    num_steps = len(y_test)
    n_features = x_test.shape[2]

    # Pre-allocate output array
    # If MC is off, we still use shape (N, 1) for consistency
    samples_col = n_mc_samples if perform_mc else 1
    predictions = np.zeros((num_steps, samples_col))

    # 2. Pre-fetch Scaler Parameters (Speed Optimization)
    # Accessing scaler attributes inside the loop is slow. Do it once here.
    if perform_enkf:
        try:
            # Handle StandardScaler vs RobustScaler
            if hasattr(scaler, 'mean_'):
                mu, sigma = scaler.mean_, scaler.scale_
            elif hasattr(scaler, 'center_'):
                mu, sigma = scaler.center_, scaler.scale_
            else:
                print("Error: Scaler not fitted."); return None, None

            # Keep only the stats for the EnKF variables to avoid indexing in loop
            enkf_mu = mu[enkf_indices]
            enkf_sigma = sigma[enkf_indices]
        except Exception as e:
            print(f"Scaler Error: {e}"); return None, None

    # 3. Main Loop
    # EnKF must be sequential (step t depends on step t-1)
    for i in tqdm(range(num_steps), desc="Forecasting"):

        # A. Current Timestamp Index
        # The test_df index corresponding to the *end* of the current sequence
        df_idx = i + seq_len - 1
        if df_idx >= len(test_df_unscaled): break

        # B. EnKF Update Step
        updated_state = None
        if perform_enkf:
            # Get Observation (z)
            z = test_df_unscaled[ENKF_STATE_VARS].iloc[df_idx].values

            # 1. Predict (Move Ensemble Forward)
            # Note: In a full Hybrid EnKF, the 'F' function would run the LSTM here.
            # Here we use the persistence assumption for the state transition.
            enkf_instance.predict(lambda x: x, Q=Q_matrix)

            # 2. Update (Correct with Observation)
            updated_state, _ = enkf_instance.update(z, R_matrix)

        # C. Update LSTM Input
        # We copy the sequence so we don't overwrite the original data
        current_seq = x_test[i].copy() # Shape: (Seq_Len, Features)

        if perform_enkf and updated_state is not None:
            # Scale the updated state (Vectorized Math)
            # (Raw - Mean) / Scale
            scaled_state = (updated_state - enkf_mu) / enkf_sigma

            # Inject into the LAST time step of the sequence
            # We replace only the columns corresponding to EnKF variables
            current_seq[-1, enkf_indices] = scaled_state

        # D. LSTM Prediction (Vectorized MC Dropout)
        # Prepare Tensor
        # Shape: (1, Seq_Len, Features)
        input_tensor = tf.convert_to_tensor([current_seq], dtype=tf.float32)

        if perform_mc:
            # OPTIMIZATION: Tile the input N times to create a batch
            # New Shape: (n_mc_samples, Seq_Len, Features)
            batch = tf.tile(input_tensor, [n_mc_samples, 1, 1])

            # ONE single call to the model for all samples
            # training=True enables Dropout
            preds = model(batch, training=True)

            # Store results (flatten to 1D array of probabilities)
            predictions[i, :] = preds[:, 0].numpy()
        else:
            # Standard Inference (No Dropout)
            pred = model(input_tensor, training=False)
            predictions[i, 0] = pred[0, 0].numpy()

    return predictions, y_test[:len(predictions)]


def estimate_enkf_noise(train_df, state_vars,
                        obs_noise_frac=0.25, proc_noise_frac=0.02,
                        protect_vars=None, protect_obs_frac=0.02,
                        return_matrices=False):
    """Estimate diagonal process/observation noise from the training slice.

    ``R > Q`` makes the filter act as a causal smoother for noisy/interpolated
    drivers. Variables listed in ``protect_vars`` (usually ``kb``) get a smaller
    observation-noise fraction so sharp bloom onsets are not over-smoothed.
    """
    protect_vars = set(protect_vars or [])
    variances = train_df[state_vars].astype(float).var().values
    variances = np.where((~np.isfinite(variances)) | (variances <= 0), 1e-6, variances)

    q_diag = proc_noise_frac * variances
    r_diag = np.array([
        (protect_obs_frac if var in protect_vars else obs_noise_frac) * var_variance
        for var, var_variance in zip(state_vars, variances)
    ])

    if return_matrices:
        return np.diag(q_diag), np.diag(r_diag)
    return q_diag.tolist(), r_diag.tolist()


def enkf_preprocess_drivers(df, state_vars, Q_diag, R_diag,
                            n_enkf=50, seed=42,
                            propagator='persistence', ar_coef=None,
                            append_std=True, std_suffix='_enkf_std'):
    """Causally smooth selected driver columns with an EnKF preprocessor.

    The returned dataframe is a copy of ``df``. The selected state variables are
    replaced by posterior ensemble means. If ``append_std`` is true, posterior
    standard-deviation columns are appended as explicit uncertainty features.
    """
    np.random.seed(seed)
    out = df.copy()
    state_vars = list(state_vars)
    Z = df[state_vars].astype(float).values
    n_steps, dim_x = Z.shape
    col_mean = np.nanmean(Z, axis=0)

    x0 = np.where(np.isnan(Z[0]), col_mean, Z[0])
    P0 = np.diag((np.abs(x0) * 0.30) ** 2) + 1e-6 * np.eye(dim_x)
    enkf = EnsembleKalmanFilter(x0, P0, dim_z=dim_x, N=n_enkf)
    Q = np.diag(np.asarray(Q_diag, dtype=float)) if np.ndim(Q_diag) == 1 else np.asarray(Q_diag, dtype=float)
    R = np.diag(np.asarray(R_diag, dtype=float)) if np.ndim(R_diag) == 1 else np.asarray(R_diag, dtype=float)

    if propagator == 'ar1':
        a = np.asarray(ar_coef if ar_coef is not None else [0.8] * dim_x, dtype=float)
        F = lambda x: col_mean + a * (x - col_mean)
    else:
        F = lambda x: x

    filtered = np.zeros_like(Z, dtype=float)
    std = np.zeros_like(Z, dtype=float)
    for t in range(n_steps):
        enkf.predict(F, Q=Q)
        z = Z[t]
        if np.any(np.isnan(z)):
            z = np.where(np.isnan(z), enkf.ensemble.mean(axis=0), z)
        enkf.update(z, R)
        filtered[t] = enkf.ensemble.mean(axis=0)
        std[t] = enkf.ensemble.std(axis=0)

    out[state_vars] = filtered
    if append_std:
        for j, var in enumerate(state_vars):
            out[f'{var}{std_suffix}'] = std[:, j]
    return out



def _best_f1_threshold_local(y_true, y_prob, fallback=0.5):
    """Local threshold helper to avoid importing pipeline_utils from this module."""
    from sklearn.metrics import precision_recall_curve

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
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fscore = np.nan_to_num(2 * precision * recall / (precision + recall + 1e-10))
    best_idx = int(np.argmax(fscore))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else float(fallback)


def _regularize_covariance(cov, dim):
    """Return a finite symmetric covariance matrix with a small jitter."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape != (dim, dim) or not np.isfinite(cov).all():
        diag = np.nan_to_num(np.diag(cov) if cov.ndim == 2 else cov, nan=1e-6, posinf=1e-6, neginf=1e-6)
        if len(diag) != dim:
            diag = np.full(dim, 1e-6)
        cov = np.diag(np.maximum(np.abs(diag), 1e-9))
    cov = (cov + cov.T) / 2.0
    cov = cov + 1e-9 * np.eye(dim)
    return cov


def _sample_zero_mean_mvn(rng, cov, size):
    """Draw ``size`` samples from N(0, cov), robust to ill-conditioned / near-PSD
    covariances.

    The fitted propagator Q spans a huge dynamic range across the state vars
    (kb variance ~1e12 vs. wind ~1e1). ``np.cov`` returns a matrix that is PSD
    in exact arithmetic but, at that condition number, carries tiny negative
    eigenvalues from floating-point roundoff. ``rng.multivariate_normal`` then
    emits ``RuntimeWarning: covariance is not symmetric positive-semidefinite``
    (and the surrounding ``try/except LinAlgError`` never fires, because that
    path warns rather than raising).

    We factor the symmetrised covariance via eigendecomposition and clip
    negative roundoff eigenvalues to zero, which is mathematically equivalent
    to sampling from the nearest PSD matrix and produces no warning.
    """
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    factor = vecs * np.sqrt(vals)                      # cov ~= factor @ factor.T
    z = rng.standard_normal(size=(size, cov.shape[0]))
    return z @ factor.T


def enkf_preprocess_drivers_with_propagator(
    df,
    state_vars,
    propagator,
    R_diag,
    n_enkf=50,
    seed=42,
    Q_matrix=None,
    append_std=True,
    std_suffix="_enkf_std",
):
    """Causally smooth driver columns with a fitted propagator in the forecast step.

    Unlike ``run_propagator_sensitivity_experiment()``, this is a true
    preprocessing path: the selected propagator is used to filter the whole
    chronological driver series before feature engineering, sequence creation,
    and model training.  The propagator must already be fit on the training
    slice only.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    state_vars = list(state_vars)
    observations = df[state_vars].astype(float).values
    n_steps, dim = observations.shape

    valid_mask = ~np.any(np.isnan(observations), axis=1)
    if not valid_mask.any():
        raise ValueError("No complete observations available to seed the EnKF.")
    first_valid_idx = int(np.argmax(valid_mask))
    x0 = observations[first_valid_idx]

    base_q = Q_matrix if Q_matrix is not None else getattr(propagator, "Q", None)
    if base_q is None:
        base_q = np.eye(dim) * 1e-6
    Q = _regularize_covariance(base_q, dim)
    R = _regularize_covariance(R_diag, dim)

    p0_diag = np.maximum((np.abs(x0) * 0.30) ** 2, np.diag(Q))
    P0 = _regularize_covariance(np.diag(p0_diag), dim)
    enkf = EnsembleKalmanFilter(x0, P0, dim_z=dim, N=n_enkf)

    start_week = None
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > first_valid_idx:
        start_week = int(df.index[first_valid_idx].isocalendar().week)
    if hasattr(propagator, "reset"):
        propagator.reset(start_week_of_year=start_week)

    filtered = np.zeros_like(observations, dtype=float)
    std = np.zeros_like(observations, dtype=float)

    for t in range(n_steps):
        if t > first_valid_idx:
            forecast_ensemble = propagator.propagate(enkf.ensemble.T).T
            noise = _sample_zero_mean_mvn(rng, Q, enkf.N)
            enkf.ensemble = forecast_ensemble + noise
            enkf.x = np.mean(enkf.ensemble, axis=0)
            enkf.P = np.cov(enkf.ensemble.T)

        z = observations[t]
        if np.any(np.isnan(z)):
            z = np.where(np.isnan(z), enkf.ensemble.mean(axis=0), z)
        enkf.update(z, R)
        filtered[t] = enkf.ensemble.mean(axis=0)
        std[t] = enkf.ensemble.std(axis=0)

    out[state_vars] = filtered
    if append_std:
        for j, var in enumerate(state_vars):
            out[f"{var}{std_suffix}"] = std[:, j]
    return out


def enkf_with_propagator_diagnostics(
    observations,
    R_diag,
    propagator,
    Q_matrix=None,
    n_enkf=50,
    seed=42,
    index=None,
):
    """Run a stochastic EnKF with a fitted propagator and return diagnostics.

    This mirrors ``enkf_preprocess_drivers_with_propagator`` but preserves the
    per-step ensemble members, innovations, Kalman-gain diagonal, and forecast
    / analysis covariance diagonals that notebook 04a and 04b need.
    """
    rng = np.random.default_rng(seed)
    observations = np.asarray(observations, dtype=float)
    n_steps, dim = observations.shape

    base_q = Q_matrix if Q_matrix is not None else getattr(propagator, "Q", None)
    if base_q is None:
        base_q = np.eye(dim) * 1e-6
    Q = _regularize_covariance(base_q, dim)
    R = _regularize_covariance(R_diag, dim)

    valid_mask = ~np.any(np.isnan(observations), axis=1)
    if not valid_mask.any():
        raise ValueError("No valid observations.")
    first_valid_idx = int(np.argmax(valid_mask))
    x0 = observations[first_valid_idx]

    p0_diag = np.maximum((np.abs(x0) * 0.30) ** 2, np.diag(Q))
    P0 = _regularize_covariance(np.diag(p0_diag), dim)
    ensemble = rng.multivariate_normal(x0, P0, size=n_enkf)

    start_week = None
    if index is not None and len(index) > first_valid_idx:
        try:
            start_week = int(pd.DatetimeIndex(index)[first_valid_idx].isocalendar().week)
        except Exception:
            start_week = None
    if hasattr(propagator, "reset"):
        propagator.reset(start_week_of_year=start_week)

    members = np.zeros((n_enkf, n_steps, dim), dtype=np.float64)
    x_f_mean = np.zeros((n_steps, dim), dtype=np.float64)
    x_a_mean = np.zeros((n_steps, dim), dtype=np.float64)
    d_b = np.full((n_steps, dim), np.nan, dtype=np.float64)
    d_a = np.full((n_steps, dim), np.nan, dtype=np.float64)
    K_diag = np.zeros((n_steps, dim), dtype=np.float64)
    P_f_diag = np.zeros((n_steps, dim), dtype=np.float64)
    P_a_diag = np.zeros((n_steps, dim), dtype=np.float64)
    obs_used = np.zeros(n_steps, dtype=bool)

    for t in range(n_steps):
        if t > first_valid_idx:
            ens_f = propagator.propagate(ensemble.T).T
            noise = _sample_zero_mean_mvn(rng, Q, n_enkf)
            ens_f = ens_f + noise
        else:
            ens_f = ensemble.copy()

        xf = ens_f.mean(axis=0)
        Pf = np.cov(ens_f.T) if n_enkf > 1 else np.zeros((dim, dim))
        x_f_mean[t] = xf
        P_f_diag[t] = np.diag(Pf)

        y = observations[t]
        if np.any(np.isnan(y)):
            ens_a = ens_f
            xa = xf
            Pa = Pf
        else:
            innov = y - xf
            d_b[t] = innov
            obs_used[t] = True

            S = Pf + R
            try:
                K_gain = np.linalg.solve(S.T, Pf.T).T
            except np.linalg.LinAlgError:
                K_gain = Pf @ np.linalg.pinv(S)
            K_diag[t] = np.diag(K_gain)

            v = rng.multivariate_normal(np.zeros(dim), R, size=n_enkf)
            y_pert = y[None, :] + v
            ens_a = ens_f + (y_pert - ens_f) @ K_gain.T
            xa = ens_a.mean(axis=0)
            Pa = np.cov(ens_a.T) if n_enkf > 1 else np.zeros((dim, dim))
            d_a[t] = y - xa

        x_a_mean[t] = xa
        P_a_diag[t] = np.diag(Pa)
        members[:, t, :] = ens_a
        ensemble = ens_a

    return {
        "members": members,
        "x_forecast_mean": x_f_mean,
        "x_analysis_mean": x_a_mean,
        "innovations_background": d_b,
        "innovations_analysis": d_a,
        "kalman_gain_diag": K_diag,
        "P_forecast_diag": P_f_diag,
        "P_analysis_diag": P_a_diag,
        "obs_used_mask": obs_used,
    }


def run_propagator_sensitivity_experiment(
    trained_model,
    X_test,
    y_test,
    test_df,
    train_df,
    scaler,
    feature_columns,
    enkf_state_vars,
    seq_len,
    n_ensemble,
    n_mc,
    Q_baseline_diag,
    R_baseline_diag,
):
    """Compare EnKF forecast propagators under the legacy test-time harness.

    This is retained for notebook 05 / Table S4 compatibility. In the reworked
    pipeline the main LSTM-EnKF comparison uses a causal EnKF preprocessor, so
    changing propagators fairly would require regenerating notebook 01 artifacts
    and retraining notebook 03 models. This function still runs the historical
    test-time EnKF sensitivity experiment so older cells do not fail.
    """
    from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                                 precision_score, recall_score)
    from propagators import fit_all_propagators

    print("\n" + "=" * 100)
    print("EXPERIMENT C: Legacy test-time EnKF propagator sensitivity")
    print("=" * 100)
    print("Note: the primary pipeline now uses EnKF as a preprocessor. Treat these")
    print("      results as a legacy diagnostic unless you retrain per propagator.")

    state_vars = list(enkf_state_vars)
    dim = len(state_vars)
    propagators = fit_all_propagators(train_df, state_vars, manual_q_diag=Q_baseline_diag)

    # Scaler parameters for replacing the last timestep state in scaled sequences.
    if hasattr(scaler, 'mean_'):
        mu, sigma = scaler.mean_, scaler.scale_
    elif hasattr(scaler, 'center_'):
        mu, sigma = scaler.center_, scaler.scale_
    else:
        raise ValueError("Scaler must expose mean_/scale_ or center_/scale_.")

    enkf_indices = [list(feature_columns).index(v) for v in state_vars]
    enkf_mu = mu[enkf_indices]
    enkf_sigma = np.where(np.asarray(sigma[enkf_indices]) == 0, 1.0, sigma[enkf_indices])

    start_week = None
    if isinstance(test_df.index, pd.DatetimeIndex) and len(test_df.index) > seq_len:
        start_week = int(test_df.index[min(seq_len - 1, len(test_df.index) - 1)].isocalendar().week)

    results = {}
    for label, prop in propagators.items():
        print(f"\n--- Propagator: {label} ---")
        try:
            enkf, _, R_matrix, _ = setup_enkf(
                enkf_state_vars=state_vars,
                n_enkf=n_ensemble,
                train_df=train_df,
                test_df_unscaled=test_df,
                seq_len=seq_len,
                feature_columns=list(feature_columns),
                manual_r_diag=R_baseline_diag,
                manual_q_diag=Q_baseline_diag,
            )
            if enkf is None:
                print(f"  skipped: setup_enkf returned None")
                continue

            if hasattr(prop, 'reset'):
                prop.reset(start_week_of_year=start_week)
            Q_matrix = _regularize_covariance(prop.Q if prop.Q is not None else np.diag(Q_baseline_diag), dim)
            R_matrix = _regularize_covariance(R_matrix, dim)

            n_steps = len(y_test)
            n_samples = int(n_mc) if int(n_mc) > 0 else 1
            preds = np.zeros((n_steps, n_samples), dtype=np.float32)
            n_done = 0

            for i in tqdm(range(n_steps), desc=f"{label}"):
                df_idx = i + seq_len - 1
                if df_idx >= len(test_df):
                    break

                forecast_ensemble = prop.propagate(enkf.ensemble.T).T
                noise = _sample_zero_mean_mvn(np.random, Q_matrix, enkf.N)
                enkf.ensemble = forecast_ensemble + noise
                enkf.x = np.mean(enkf.ensemble, axis=0)
                enkf.P = np.cov(enkf.ensemble.T)

                z = test_df[state_vars].iloc[df_idx].values.astype(float)
                updated_state, _ = enkf.update(z, R_matrix)

                current_seq = X_test[i].copy()
                current_seq[-1, enkf_indices] = (updated_state - enkf_mu) / enkf_sigma
                input_tensor = tf.convert_to_tensor([current_seq], dtype=tf.float32)

                if n_samples > 1:
                    batch = tf.tile(input_tensor, [n_samples, 1, 1])
                    out = trained_model(batch, training=True)[:, 0].numpy()
                else:
                    out = trained_model(input_tensor, training=False)[:, 0].numpy()
                preds[i, :] = out
                n_done += 1

            preds = preds[:n_done]
            y_true = np.asarray(y_test).flatten().astype(int)[:n_done]
            y_prob = preds.mean(axis=1)
            threshold = _best_f1_threshold_local(y_true, y_prob)
            y_pred = (y_prob >= threshold).astype(int)

            onsets = np.where((y_true[:-1] == 0) & (y_true[1:] == 1))[0] + 1
            n_onsets = int(len(onsets))
            n_det = int(sum(y_pred[idx] == 1 for idx in onsets if idx < len(y_pred)))

            row = {
                'acc': float(accuracy_score(y_true, y_pred)),
                'prec': float(precision_score(y_true, y_pred, zero_division=0)),
                'rec': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'brier': float(brier_score_loss(y_true, y_prob)),
                'n_onsets': n_onsets,
                'n_det': n_det,
                'ohr': float(n_det / n_onsets) if n_onsets else float('nan'),
                'threshold': float(threshold),
                'Q_trace': float(np.trace(Q_matrix)),
            }
            if hasattr(prop, 'spectral_radius') and prop.spectral_radius is not None:
                row['spectral_radius'] = float(prop.spectral_radius)
            results[label] = row
            print(
                f"  acc={row['acc']:.3f} prec={row['prec']:.3f} rec={row['rec']:.3f} "
                f"f1={row['f1']:.3f} brier={row['brier']:.4f} threshold={row['threshold']:.3f}"
            )

        except Exception as exc:
            print(f"  ERROR in {label}: {exc}")
            continue

    return results
