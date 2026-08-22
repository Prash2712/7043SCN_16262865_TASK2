from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "national_demand_mw"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build demand features using only information available before each timestamp."""
    if "timestamp" not in frame.columns or TARGET not in frame.columns:
        raise ValueError("Expected timestamp and national_demand_mw columns")

    result = frame.sort_values("timestamp").set_index("timestamp").copy()
    target = result[TARGET].astype(float)

    lag_columns: list[str] = []
    for lag in (1, 2, 48, 96, 336):
        column = f"demand_lag_{lag}"
        result[column] = target.shift(lag)
        lag_columns.append(column)

    result["demand_roll_mean_48"] = target.shift(1).rolling(48).mean()
    result["demand_roll_std_48"] = target.shift(1).rolling(48).std()
    result["demand_roll_mean_336"] = target.shift(1).rolling(336).mean()

    half_hour = result.index.hour * 2 + (result.index.minute // 30)
    result["half_hour_sin"] = np.sin(2 * np.pi * half_hour / 48)
    result["half_hour_cos"] = np.cos(2 * np.pi * half_hour / 48)
    result["dow_sin"] = np.sin(2 * np.pi * result.index.dayofweek / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result.index.dayofweek / 7)
    result["day_of_week"] = result.index.dayofweek
    result["month"] = result.index.month
    result["is_weekend"] = (result.index.dayofweek >= 5).astype(int)

    required_history = [
        TARGET,
        *lag_columns,
        "demand_roll_mean_48",
        "demand_roll_std_48",
        "demand_roll_mean_336",
    ]
    # Optional realised outturn fields are deliberately not part of the drop condition.
    # The day-ahead model does not use them unless a future forecast contract is added.
    return result.dropna(subset=required_history).reset_index()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "settlement_date",
        "settlement_period",
        TARGET,
        "transmission_system_demand_mw",
        "embedded_wind_generation_mw",
        "embedded_solar_generation_mw",
    }
    return [column for column in frame.columns if column not in excluded]
