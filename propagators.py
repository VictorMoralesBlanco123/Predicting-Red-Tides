"""
Alternative driver-propagation models for the LSTM-EnKF hybrid framework.

Replaces the random-walk forecast step `x_t^f = x_{t-1}^a + q_t` with one of
three trained propagators. All propagators expose the same interface so they
can be swapped into the EnKF forecast step without changing the analysis-step
or LSTM-coupling code.

State variable ordering must match ENKF_STATE_VARS in shared_config.py:
    ['peace_discharge', 'peace_TN', 'peace_TP', 'kb', 'wind_u', 'wind_v']

Usage pattern (intended integration with EnsembleKalmanFilter):

    propagator = AR1Propagator(state_vars=ENKF_STATE_VARS).fit(train_df)
    Q_matrix   = propagator.Q                 # plug into setup_enkf
    # In the EnKF forecast step, replace:
    #     x_forecast = x_analysis + noise
    # with:
    #     x_forecast = propagator.propagate(x_analysis) + noise
    # where noise is drawn from N(0, Q_matrix) as before.

The propagators are deterministic (return the conditional mean). Stochastic
process noise is left to the EnKF's existing noise-injection code, using the
Q matrix that this module fits from training residuals. This keeps the
analysis step and noise machinery in enkf_utils.py untouched.

Author: Victor M. Blanco (with Claude review-cycle assistance)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePropagator(ABC):
    """Base class. Subclasses implement `fit()` and `_propagate_impl()`."""

    def __init__(self, state_vars: Sequence[str]):
        self.state_vars = list(state_vars)
        self.n_state = len(self.state_vars)
        self.Q: Optional[np.ndarray] = None      # innovation covariance fit from training
        self._fitted: bool = False
        self._t: int = 0                          # internal forecast-step counter
        self._start_woy: Optional[int] = None     # week-of-year at forecast start

    # --- subclass hooks ---
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BasePropagator":
        """Fit parameters from training data only. Returns self."""

    @abstractmethod
    def _propagate_impl(self, ensemble: np.ndarray) -> np.ndarray:
        """Deterministic one-step forecast.

        Parameters
        ----------
        ensemble : np.ndarray, shape (n_state, n_ensemble)
            Current analysis ensemble.

        Returns
        -------
        np.ndarray, shape (n_state, n_ensemble)
            Forecast mean for each ensemble member (no stochastic noise added).
        """

    # --- public API used by the EnKF integration ---
    def propagate(self, ensemble: np.ndarray) -> np.ndarray:
        """Advance the ensemble by one step and increment internal time counter."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.propagate() called before fit()")
        if ensemble.shape[0] != self.n_state:
            raise ValueError(
                f"ensemble has {ensemble.shape[0]} rows, expected {self.n_state} "
                f"({self.state_vars})"
            )
        out = self._propagate_impl(ensemble)
        self._t += 1
        return out

    def reset(self, start_week_of_year: Optional[int] = None) -> None:
        """Reset the internal step counter. Required for seasonal propagator."""
        self._t = 0
        self._start_woy = start_week_of_year

    @property
    def current_week_of_year(self) -> Optional[int]:
        if self._start_woy is None:
            return None
        return ((self._start_woy - 1 + self._t) % 52) + 1

    def name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# 1. Random walk (reference baseline, matches the current implementation)
# ---------------------------------------------------------------------------

class RandomWalkPropagator(BasePropagator):
    """x_t = x_{t-1} + q_t. The current EnKF baseline, re-implemented here
    so all four configurations share one comparison harness."""

    def fit(self, train_df: pd.DataFrame, manual_q_diag: Optional[Sequence[float]] = None):
        if manual_q_diag is not None:
            self.Q = np.diag(np.asarray(manual_q_diag, dtype=float))
        else:
            diffs = train_df[self.state_vars].diff().dropna().values
            self.Q = np.cov(diffs.T)
        self._fitted = True
        return self

    def _propagate_impl(self, ensemble):
        return ensemble.copy()


# ---------------------------------------------------------------------------
# 2. AR(1) per variable with mean reversion
# ---------------------------------------------------------------------------

class AR1Propagator(BasePropagator):
    r"""Per-variable AR(1) with mean reversion:
        x^j_t = alpha_j * x^j_{t-1} + (1 - alpha_j) * mu_j + q^j_t

    Fits one (alpha_j, mu_j) pair per state variable on the training data.
    The innovation covariance Q is the empirical covariance of the residuals
    x_t - [alpha * x_{t-1} + (1-alpha) * mu] across the training period.
    """

    def __init__(self, state_vars, alpha_clip: float = 0.99):
        super().__init__(state_vars)
        self.alpha_clip = alpha_clip
        self.alpha: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame):
        self.alpha = np.zeros(self.n_state)
        self.mu = np.zeros(self.n_state)

        # Collect aligned (T-1, n_state) matrix of residuals to estimate full Q.
        # Use the longest contiguous span per variable; assemble by inner-join
        # of nonmissing rows so Q is computed on a consistent sample.
        df = train_df[self.state_vars].astype(float)
        mask = df.notna().all(axis=1)
        # Use only contiguous samples for innovation covariance estimation.
        df_clean = df.loc[mask].reset_index(drop=True)
        if len(df_clean) < 3:
            raise ValueError("Not enough complete training rows to fit AR(1).")

        X_prev = df_clean.values[:-1]
        X_curr = df_clean.values[1:]
        residuals = np.zeros_like(X_curr)

        for j, var in enumerate(self.state_vars):
            xp, xc = X_prev[:, j], X_curr[:, j]
            # OLS on x_t = alpha * x_{t-1} + intercept
            Z = np.column_stack([xp, np.ones_like(xp)])
            beta, *_ = np.linalg.lstsq(Z, xc, rcond=None)
            alpha_j, intercept_j = float(beta[0]), float(beta[1])
            alpha_j = float(np.clip(alpha_j, -self.alpha_clip, self.alpha_clip))
            mu_j = intercept_j / (1.0 - alpha_j) if abs(1.0 - alpha_j) > 1e-8 \
                else float(np.nanmean(df[var]))
            self.alpha[j] = alpha_j
            self.mu[j] = mu_j
            residuals[:, j] = xc - (alpha_j * xp + (1.0 - alpha_j) * mu_j)

        self.Q = np.cov(residuals.T)
        self._fitted = True
        return self

    def _propagate_impl(self, ensemble):
        # ensemble: (n_state, n_ensemble)
        a = self.alpha[:, None]
        m = self.mu[:, None]
        return a * ensemble + (1.0 - a) * m


# ---------------------------------------------------------------------------
# 3. VAR(1): vector AR(1) with cross-variable dynamics
# ---------------------------------------------------------------------------

class VAR1Propagator(BasePropagator):
    r"""Vector AR(1): x_t = A x_{t-1} + b + q_t.

    A is fit by ridge-regularized least squares; b is the intercept vector.
    If the fitted A has spectral radius >= 1 (unstable), it is rescaled to
    spectral radius 0.99 to keep the propagator stationary.
    """

    def __init__(self, state_vars, ridge: float = 1e-4):
        super().__init__(state_vars)
        self.ridge = ridge
        self.A: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.spectral_radius: Optional[float] = None

    def fit(self, train_df: pd.DataFrame):
        df = train_df[self.state_vars].astype(float)
        mask = df.notna().all(axis=1)
        X = df.loc[mask].values
        if len(X) < self.n_state + 3:
            raise ValueError("Not enough complete training rows to fit VAR(1).")

        X_prev = X[:-1]
        X_curr = X[1:]
        Z = np.column_stack([X_prev, np.ones(len(X_prev))])  # (T-1, n_state+1)

        ZtZ = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        ZtY = Z.T @ X_curr
        coef = np.linalg.solve(ZtZ, ZtY).T              # (n_state, n_state+1)
        self.A = coef[:, :-1]
        self.b = coef[:, -1]

        # Stationarity guard: rescale A if spectral radius >= 0.999
        eigs = np.linalg.eigvals(self.A)
        rho = float(np.max(np.abs(eigs)))
        if rho >= 0.999:
            self.A = self.A * (0.99 / rho)
            self.spectral_radius = 0.99
        else:
            self.spectral_radius = rho

        residuals = X_curr - (X_prev @ self.A.T + self.b)
        self.Q = np.cov(residuals.T)
        self._fitted = True
        return self

    def _propagate_impl(self, ensemble):
        # ensemble: (n_state, n_ensemble); A @ ensemble preserves shape
        return self.A @ ensemble + self.b[:, None]


# ---------------------------------------------------------------------------
# 4. Seasonal climatology (Fourier) + AR(1) on residual anomalies
# ---------------------------------------------------------------------------

class SeasonalAR1Propagator(BasePropagator):
    r"""Decomposes each variable as
        x^j_t = c_j(woy(t)) + a^j_t
    where c_j is a low-order Fourier expansion of week-of-year (annual cycle
    with `n_harmonics` harmonics, default 2), and the anomaly a^j_t evolves
    as AR(1):
        a^j_t = alpha_j * a^j_{t-1} + eps^j_t.

    Requires `train_df` to have a DatetimeIndex or a column named 'date'.
    The propagator must be reset with `start_week_of_year` before the first
    forecast step so it can advance its internal calendar pointer.
    """

    def __init__(self, state_vars, n_harmonics: int = 2, alpha_clip: float = 0.99):
        super().__init__(state_vars)
        self.n_harmonics = int(n_harmonics)
        self.alpha_clip = alpha_clip
        self.alpha: Optional[np.ndarray] = None
        self.season_beta: Optional[np.ndarray] = None  # (n_state, 1 + 2*n_harmonics)

    @staticmethod
    def _extract_woy(df: pd.DataFrame) -> np.ndarray:
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.isocalendar().week.values.astype(float)
        if "date" in df.columns:
            d = pd.to_datetime(df["date"])
            return d.dt.isocalendar().week.values.astype(float)
        raise ValueError(
            "SeasonalAR1Propagator requires train_df to have a DatetimeIndex "
            "or a 'date' column."
        )

    def _fourier_design(self, woy: np.ndarray) -> np.ndarray:
        woy = np.asarray(woy, dtype=float)
        cols = [np.ones_like(woy)]
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.cos(2.0 * np.pi * k * woy / 52.0))
            cols.append(np.sin(2.0 * np.pi * k * woy / 52.0))
        return np.column_stack(cols)

    def _climatology_at(self, woy_value: int) -> np.ndarray:
        d = self._fourier_design(np.array([woy_value], dtype=float))  # (1, K)
        return (self.season_beta @ d.T).flatten()                      # (n_state,)

    def fit(self, train_df: pd.DataFrame):
        woy = self._extract_woy(train_df)
        D = self._fourier_design(woy)
        X = train_df[self.state_vars].values.astype(float)

        self.season_beta = np.zeros((self.n_state, D.shape[1]))
        self.alpha = np.zeros(self.n_state)
        anomaly_matrix = np.zeros_like(X)

        for j, var in enumerate(self.state_vars):
            x = X[:, j]
            mask = ~np.isnan(x)
            beta, *_ = np.linalg.lstsq(D[mask], x[mask], rcond=None)
            self.season_beta[j] = beta
            clim = D @ beta
            anomaly_matrix[:, j] = x - clim

        # AR(1) on anomalies, restricted to rows where ALL state vars are finite,
        # so Q can be computed on a consistent contiguous sample.
        finite_rows = ~np.isnan(anomaly_matrix).any(axis=1)
        A = anomaly_matrix[finite_rows]
        A_prev = A[:-1]
        A_curr = A[1:]
        residuals = np.zeros_like(A_curr)
        for j in range(self.n_state):
            ap, ac = A_prev[:, j], A_curr[:, j]
            denom = float(np.sum(ap * ap))
            alpha_j = float(np.sum(ap * ac) / denom) if denom > 1e-12 else 0.0
            alpha_j = float(np.clip(alpha_j, -self.alpha_clip, self.alpha_clip))
            self.alpha[j] = alpha_j
            residuals[:, j] = ac - alpha_j * ap

        self.Q = np.cov(residuals.T)
        self._fitted = True
        return self

    def _propagate_impl(self, ensemble):
        if self._start_woy is None:
            raise RuntimeError(
                "SeasonalAR1Propagator requires reset(start_week_of_year=...) "
                "to be called before the first propagate()."
            )
        # Anomaly = current ensemble minus current week's climatology.
        clim_curr = self._climatology_at(self.current_week_of_year)
        next_woy = ((self.current_week_of_year - 1 + 1) % 52) + 1
        clim_next = self._climatology_at(next_woy)
        anomaly = ensemble - clim_curr[:, None]
        anomaly_next = self.alpha[:, None] * anomaly
        return clim_next[:, None] + anomaly_next


# ---------------------------------------------------------------------------
# Convenience: fit-and-summarize all four propagators
# ---------------------------------------------------------------------------

def fit_all_propagators(train_df: pd.DataFrame, state_vars: Sequence[str],
                        manual_q_diag: Optional[Sequence[float]] = None,
                        seasonal_n_harmonics: int = 2) -> dict:
    """Fit all four propagators on train_df and return a dict keyed by name.

    The random-walk propagator can be fit to your existing manual Q via
    `manual_q_diag`; pass `MANUAL_Q_DIAG` from shared_config for an apples-
    to-apples comparison with the current baseline.
    """
    propagators = {
        "RandomWalk":   RandomWalkPropagator(state_vars).fit(train_df, manual_q_diag=manual_q_diag),
        "AR1":          AR1Propagator(state_vars).fit(train_df),
        "VAR1":         VAR1Propagator(state_vars).fit(train_df),
        "SeasonalAR1":  SeasonalAR1Propagator(state_vars, n_harmonics=seasonal_n_harmonics).fit(train_df),
    }
    return propagators


def summarize_propagators(propagators: dict) -> pd.DataFrame:
    """Return a small DataFrame summarizing each fitted propagator's parameters."""
    rows = []
    for name, p in propagators.items():
        row = {"propagator": name}
        if hasattr(p, "alpha") and p.alpha is not None:
            for j, var in enumerate(p.state_vars):
                row[f"alpha[{var}]"] = float(p.alpha[j])
        if hasattr(p, "spectral_radius") and p.spectral_radius is not None:
            row["VAR1_spectral_radius"] = p.spectral_radius
        row["Q_trace"] = float(np.trace(p.Q))
        row["Q_diag_geomean"] = float(np.exp(np.mean(np.log(np.abs(np.diag(p.Q)) + 1e-30))))
        rows.append(row)
    return pd.DataFrame(rows)
