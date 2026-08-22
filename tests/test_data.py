import pandas as pd

from gridcast.data import normalise_demand


def test_normalise_demand_builds_half_hourly_timestamp():
    raw = pd.DataFrame(
        {
            "SETTLEMENT_DATE": ["01/07/2026", "01/07/2026"],
            "SETTLEMENT_PERIOD": [1, 2],
            "ND": [25000, 25200],
            "TSD": [27000, 27200],
        }
    )

    result = normalise_demand(raw)

    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-07-01 00:00:00")
    assert result.loc[1, "timestamp"] == pd.Timestamp("2026-07-01 00:30:00")
    assert result.loc[0, "national_demand_mw"] == 25000
