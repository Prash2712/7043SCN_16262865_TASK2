# GridCast GB — Electricity Demand Forecasting & Model Serving

A production-style machine-learning engineering project for **half-hourly Great Britain electricity-demand forecasting** using official National Energy System Operator (NESO) open data.

The repository demonstrates the parts of an ML system that are normally missing from notebook portfolios: API ingestion, data contracts, leakage-safe feature engineering, temporal baselines, hold-out evaluation, artifact persistence, a versioned FastAPI serving contract, Docker and CI.

## Business / engineering problem

Electricity demand changes sharply by time of day, weekday, season and recent system history. A useful forecasting system must do more than fit a regressor: it needs fresh data, a defendable evaluation protocol, a baseline, reproducible feature logic and a deployable prediction interface.

GridCast is built around that complete lifecycle.

## Data source

NESO publishes historic Great Britain demand, interconnector, wind and solar outturn data through its Data Portal and CKAN API.

Historic Demand Data 2026 resource used by default:

- CKAN endpoint: `https://api.neso.energy/api/3/action/datastore_search`
- resource id: `8a4a771c-3929-4e56-93ad-cdf13219dea5`
- official portal: https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2026

NESO notes that historic demand data can be retrospectively corrected, so production workflows should treat source revisions explicitly.

## Architecture

```text
NESO CKAN API
     |
     v
Paginated ingestion
     |
     v
Schema normalisation + timestamp contract
     |
     v
Half-hourly demand history
     |
     v
Lag / rolling / calendar feature pipeline
     |
     +------------------------+
     |                        |
     v                        v
24h seasonal baseline   HistGradientBoosting
     |                        |
     +-----------+------------+
                 v
        chronological hold-out
                 |
                 v
         versioned model bundle
                 |
          +------+------+
          |             |
          v             v
       FastAPI        metrics JSON
          |
          v
       Docker
```

## ML design

The default feature contract is deliberately deployment-aware. It uses prior realised demand and calendar features:

- lag 1 and 2 settlement periods
- lag 48 (same half-hour previous day)
- lag 96
- lag 336 (same half-hour previous week)
- rolling 24-hour mean / standard deviation
- rolling 7-day mean
- half-hour cyclical encoding
- day-of-week cyclical encoding
- month and weekend indicators

Realised future wind or solar output is **not** used as a predictor because it would not be known at forecast time. A later production extension should use forecast weather / embedded-generation variables available before the prediction timestamp.

## Evaluation

Evaluation is chronological; there is no random train/test split.

The final block of observations is held out and the model is compared against a transparent seasonal-naive baseline:

```text
prediction(t) = demand(t - 48 half-hours)
```

The pipeline records:

- model MAE
- model RMSE
- model MAPE
- seasonal-naive MAE
- hold-out size

No performance number is hard-coded in this README. Metrics are generated from the data actually used and persisted next to the model artifact.

## Repository layout

```text
.
├── src/gridcast/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── api.py
│   └── cli.py
├── tests/
├── docs/
│   └── model_card.md
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Run the pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

gridcast fetch
gridcast train
```

For a small ingestion smoke test:

```bash
gridcast fetch --limit 5000
```

Generated files are intentionally ignored by Git:

```text
data/processed/neso_demand.parquet
artifacts/gridcast.joblib
artifacts/gridcast.metrics.json
```

## Serve the model

```bash
uvicorn gridcast.api:app --reload
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

`/predict` enforces the exact feature contract stored with the model bundle and rejects missing or unexpected fields.

## Container

```bash
docker build -t gridcast-gb .
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" gridcast-gb
```

The artifact is mounted at runtime rather than embedded into source control.

## Quality controls

```bash
ruff check src tests
pytest -q
```

GitHub Actions runs both on pushes and pull requests.

## Model governance

See [`docs/model_card.md`](docs/model_card.md) for intended use, feature availability rules, evaluation boundaries, limitations and monitoring recommendations.

## What this project demonstrates

**Data Science**
- time-series feature engineering
- baseline design
- chronological evaluation
- error metrics

**ML Engineering**
- source API ingestion
- packaged feature/model code
- artifact persistence
- versioned inference contract
- FastAPI
- Docker
- CI

**Analytics Engineering**
- explicit schemas
- data-quality failures
- reproducible transformations
- source-aware model inputs

## Suggested repository name

Rename the current university-style shell to:

`gb-energy-demand-forecasting-mlops`

The original reinforcement-learning coursework is preserved on `archive/original-coursework`.

## Author

**Prasanth Balisetty**  
Data Science · Machine Learning Engineering · Analytics
