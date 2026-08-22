import numpy as np
import pandas as pd

from gridcast.features import build_features


def test_build_features_uses_prior_demand_only():
    timestamps = pd.date_range("2026-01-01", periods=400, freq="30min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "national_demand_mw": np.arange(400, dtype=float) + 20000,
        }
    )

    features = build_features(frame)
    row = features.iloc[0]
    source_index = frame.index[frame.timestamp == row["timestamp"]][0]

    assert row["demand_lag_1"] == frame.loc[source_index - 1, "national_demand_mw"]
    assert row["demand_lag_48"] == frame.loc[source_index - 48, "national_demand_mw"]
    assert row["demand_lag_336"] == frame.loc[source_index - 336, "national_demand_mw"]
