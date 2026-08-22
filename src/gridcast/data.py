from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

CKAN_URL = "https://api.neso.energy/api/3/action/datastore_search"
DEFAULT_RESOURCE_ID = "8a4a771c-3929-4e56-93ad-cdf13219dea5"


def fetch_neso_records(
    resource_id: str = DEFAULT_RESOURCE_ID,
    limit: int | None = None,
    page_size: int = 5000,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch a NESO DataStore resource with deterministic pagination."""
    records: list[dict] = []
    offset = 0

    while True:
        remaining = None if limit is None else max(limit - len(records), 0)
        if remaining == 0:
            break
        current_limit = page_size if remaining is None else min(page_size, remaining)
        response = requests.get(
            CKAN_URL,
            params={"resource_id": resource_id, "limit": current_limit, "offset": offset},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError("NESO CKAN API returned success=false")
        batch = payload["result"]["records"]
        records.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(payload["result"]["total"]):
            break

    return pd.DataFrame.from_records(records)


def normalise_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert NESO historic-demand fields into a stable half-hourly contract."""
    columns = {str(column).strip().upper(): column for column in frame.columns}
    required = {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing NESO demand fields: {sorted(missing)}")

    result = pd.DataFrame()
    result["settlement_date"] = pd.to_datetime(frame[columns["SETTLEMENT_DATE"]], dayfirst=True, errors="coerce")
    result["settlement_period"] = pd.to_numeric(frame[columns["SETTLEMENT_PERIOD"]], errors="coerce")
    result["national_demand_mw"] = pd.to_numeric(frame[columns["ND"]], errors="coerce")

    optional = {
        "TSD": "transmission_system_demand_mw",
        "EMBEDDED_WIND_GENERATION": "embedded_wind_generation_mw",
        "EMBEDDED_SOLAR_GENERATION": "embedded_solar_generation_mw",
    }
    for source, target in optional.items():
        if source in columns:
            result[target] = pd.to_numeric(frame[columns[source]], errors="coerce")

    result = result.dropna(subset=["settlement_date", "settlement_period", "national_demand_mw"])
    result = result[result["settlement_period"].between(1, 50)].copy()
    result["timestamp"] = result["settlement_date"] + pd.to_timedelta(
        (result["settlement_period"] - 1) * 30, unit="m"
    )
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    return result.sort_values("timestamp").reset_index(drop=True)


def save_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return output
