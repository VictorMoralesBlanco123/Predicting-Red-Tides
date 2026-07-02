"""
data_pipeline_utils.py
======================

Reusable data-preparation helpers extracted from notebook 01.  These functions
let audit notebooks rebuild branch-specific feature, split, scaling, and
sequence artifacts without copying notebook cells.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

import onset_utils as ou


def calculate_wind_components(df: pd.DataFrame):
    """Calculate meteorological wind U/V components when source columns exist."""
    out = df.copy()
    if "wind_direction" not in out.columns or "wind_speed" not in out.columns:
        return out, []

    out["wind_direction"] = pd.to_numeric(out["wind_direction"], errors="coerce")
    out["wind_speed"] = pd.to_numeric(out["wind_speed"], errors="coerce")
    out[["wind_direction", "wind_speed"]] = out[["wind_direction", "wind_speed"]].fillna(0)

    wind_dir_rad = np.deg2rad(out["wind_direction"])
    wind_speed = out["wind_speed"]
    out["wind_u"] = -wind_speed * np.sin(wind_dir_rad)
    out["wind_v"] = -wind_speed * np.cos(wind_dir_rad)
    out = out.drop(columns=["wind_direction", "wind_speed"])
    return out, ["wind_u", "wind_v"]


def create_target(
    df: pd.DataFrame,
    target_col: str,
    bloom_threshold: float,
    target_binary_col: str,
) -> pd.DataFrame:
    """Create the binary bloom target and drop rows with missing target."""
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found.")

    out = df.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col])
    out[target_binary_col] = (out[target_col] >= bloom_threshold).astype(int)
    return out


def load_and_prepare_data(
    filepath: str,
    datetime_col: str,
    target_col: str,
    bloom_threshold: float,
    target_binary_col: str,
) -> pd.DataFrame:
    """Load raw weekly data and apply the same initial prep as notebook 01."""
    df = pd.read_csv(filepath)
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column {datetime_col!r} not found.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).set_index(datetime_col)
    df, _ = calculate_wind_components(df)
    return create_target(df, target_col, bloom_threshold, target_binary_col)


def add_onset_feature_columns(
    df: pd.DataFrame,
    target_binary_col: str,
    shelf_angle_deg: float = 340.0,
    rate_cols: Optional[Sequence[str]] = None,
    anomaly_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Append the onset-relevant columns used by notebook 01.

    Call this after any driver replacement/smoothing so derived features track
    the exact driver trajectory being evaluated.
    """
    out = df.copy()
    if {"wind_u", "wind_v"}.issubset(out.columns):
        out["wind_alongshore"] = ou.alongshore_wind(
            out["wind_u"], out["wind_v"], shelf_angle_deg=shelf_angle_deg
        )

    rate_cols = list(rate_cols or ["water_temp", "peace_TN", "peace_TP", "peace_discharge", "zos"])
    present_rate_cols = [c for c in rate_cols if c in out.columns]
    if present_rate_cols:
        out = ou.add_rate_of_change_features(out, cols=present_rate_cols, windows=(2, 4, 8), suffix="d")

    anomaly_cols = list(anomaly_cols or ["water_temp", "peace_TN", "peace_TP"])
    present_anomaly_cols = [c for c in anomaly_cols if c in out.columns]
    if present_anomaly_cols:
        out = ou.add_cumulative_anomaly_features(
            out,
            cols=present_anomaly_cols,
            climatology_period=52,
            accumulation_windows=(4, 8, 12),
        )

    if target_binary_col in out.columns:
        wsb = pd.Series(ou.weeks_since_last_bloom(out[target_binary_col]), index=out.index)
        out["weeks_since_bloom"] = wsb.fillna(wsb.max() if wsb.notna().any() else 0)

    return out


def create_features(
    df: pd.DataFrame,
    target_col: str,
    target_binary_col: str,
    base_features: Iterable[str],
    use_enhanced: bool = False,
    basic_cfg: Optional[dict] = None,
    enhanced_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Create the lagged and rolling features from notebook 01."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    source = df.copy()
    featured = source.copy()

    if use_enhanced:
        cfg = enhanced_cfg or {}
        env_lags = cfg.get("env_lags", [1])
        kb_lags = cfg.get("kb_lags", [])
        kb_rolling_windows = cfg.get("kb_rolling_windows", [])
        discharge_rolling_window = cfg.get("discharge_rolling_window", None)
    else:
        cfg = basic_cfg or {}
        env_lags = cfg.get("env_lags", [])
        kb_lags = cfg.get("kb_lags", [])
        kb_rolling_windows = []
        discharge_rolling_window = None

    max_lag_needed = 0

    if target_col in source.columns:
        source[target_col] = pd.to_numeric(source[target_col], errors="coerce")
        for lag in kb_lags:
            featured[f"{target_col}_L{lag}"] = source[target_col].shift(lag)
            max_lag_needed = max(max_lag_needed, lag)

    if use_enhanced and kb_rolling_windows and target_col in source.columns and target_binary_col in source.columns:
        target_binary = source[target_binary_col]
        for i, (start_lag, end_lag) in enumerate(kb_rolling_windows):
            month_lag_id = f"M{i + 1}"
            window_size = end_lag - start_lag + 1
            featured[f"{target_col}_{month_lag_id}_mean"] = (
                source[target_col].rolling(window=window_size, min_periods=1).mean().shift(start_lag)
            )
            featured[f"{target_binary_col}_{month_lag_id}_prop"] = (
                target_binary.rolling(window=window_size, min_periods=1).mean().shift(start_lag)
            )
            max_lag_needed = max(max_lag_needed, end_lag)

    discharge_col = "peace_discharge"
    if use_enhanced and discharge_rolling_window and discharge_col in source.columns:
        start_lag = 1
        window_size = int(discharge_rolling_window)
        featured[f"{discharge_col}_{window_size}w_avg_L{start_lag}"] = (
            source[discharge_col].rolling(window=window_size, min_periods=1).mean().shift(start_lag)
        )
        max_lag_needed = max(max_lag_needed, window_size + start_lag - 1)

    for feature in base_features:
        if feature not in source.columns:
            continue
        source[feature] = pd.to_numeric(source[feature], errors="coerce")
        for lag in env_lags:
            featured[f"{feature}_L{lag}"] = source[feature].shift(lag)
            max_lag_needed = max(max_lag_needed, lag)

    processed = featured.iloc[max_lag_needed:].copy() if max_lag_needed > 0 else featured.copy()
    all_nan_cols = processed.columns[processed.isnull().all()].tolist()
    if all_nan_cols:
        processed = processed.dropna(axis=1, how="all")
    return processed.loc[:, ~processed.columns.duplicated()]


def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
):
    """Chronologically split a sorted DateTimeIndex dataframe."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataframe index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Not enough rows for non-empty train/val/test splits: "
            f"train={n_train}, val={n_val}, test={n_test}."
        )

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    if not train_df.index.max() < val_df.index.min():
        raise AssertionError("Train/validation splits overlap.")
    if not val_df.index.max() < test_df.index.min():
        raise AssertionError("Validation/test splits overlap.")
    return train_df, val_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "Robust",
    scaler_path: Optional[str] = None,
    feature_list_path: Optional[str] = None,
    imputer_path: Optional[str] = None,
):
    """Impute and scale numeric features, fitting only on the training split."""
    if any(df is None or df.empty for df in [train_df, validation_df, test_df]):
        raise ValueError("One or more input dataframes are empty.")

    feature_candidates = [c for c in train_df.columns if c != target_col]
    final_feature_columns = train_df[feature_candidates].select_dtypes(include=np.number).columns.tolist()
    if not final_feature_columns:
        raise ValueError("No numeric feature columns found.")

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler() if scaler_type.lower() == "standard" else RobustScaler()

    train_features = train_df[final_feature_columns]
    impute_needed = train_features.isnull().values.any()
    if impute_needed:
        imputer.fit(train_features)
        temp_train_features = imputer.transform(train_features)
    else:
        temp_train_features = train_features
    scaler.fit(temp_train_features)

    processed = []
    for df in (train_df, validation_df, test_df):
        df_copy = df.copy()
        if impute_needed:
            df_copy[final_feature_columns] = imputer.transform(df_copy[final_feature_columns])
        df_copy[final_feature_columns] = scaler.transform(df_copy[final_feature_columns])
        processed.append(df_copy)

    for path, obj in [
        (scaler_path, scaler),
        (feature_list_path, final_feature_columns),
        (imputer_path, imputer if impute_needed else None),
    ]:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, path)

    return (*processed, scaler, final_feature_columns)
