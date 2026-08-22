from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from gridcast.data import DEFAULT_RESOURCE_ID, fetch_neso_records, normalise_demand, save_parquet
from gridcast.features import build_features
from gridcast.model import train_and_evaluate

app = typer.Typer(help="GridCast GB forecasting workflow")


@app.command()
def fetch(
    resource_id: str = DEFAULT_RESOURCE_ID,
    output: Path = Path("data/processed/neso_demand.parquet"),
    limit: int | None = None,
) -> None:
    """Fetch and normalise a NESO historic-demand DataStore resource."""
    raw = fetch_neso_records(resource_id=resource_id, limit=limit)
    demand = normalise_demand(raw)
    save_parquet(demand, output)
    typer.echo(f"Saved {len(demand):,} half-hourly records to {output}")


@app.command()
def train(
    input_path: Path = Path("data/processed/neso_demand.parquet"),
    artifact: Path = Path("artifacts/gridcast.joblib"),
    holdout_days: int = 14,
) -> None:
    """Build lagged features, compare to a seasonal baseline and persist the model."""
    demand = pd.read_parquet(input_path)
    features = build_features(demand)
    result = train_and_evaluate(
        features,
        artifact_path=artifact,
        holdout_periods=48 * holdout_days,
    )
    typer.echo(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    app()
