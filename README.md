# GridCast GB

Half-hourly electricity demand forecasting for Great Britain using National Energy System Operator (NESO) open data.

The baseline in this project matters as much as the model. Electricity demand has strong daily and weekly structure, so a model that cannot beat “use the same half-hour from yesterday” has not earned much complexity.

## Data path

GridCast reads NESO's historic demand data through the CKAN API, normalises timestamps, builds lagged features and evaluates the newest block of observations as a hold-out.

```text
NESO API
  -> paginated ingestion
  -> normalised half-hour history
  -> lag / rolling / calendar features
  -> seasonal-naive baseline + HistGradientBoosting
  -> chronological hold-out
  -> model bundle + metrics JSON
  -> FastAPI
```

The default NESO resource is configured in the package rather than copied into the repository. Historic data can be retrospectively corrected, so the ingestion layer treats source data as revisable rather than assuming an old download is permanent truth.

## Feature availability

The feature set only uses information that could exist at prediction time:

- demand lags at 1, 2, 48, 96 and 336 half-hours
- trailing 24-hour mean and standard deviation
- trailing seven-day mean
- cyclical half-hour and day-of-week encodings
- month and weekend indicators

I deliberately left realised future wind and solar out. They may correlate with demand, but using outturn values from the future would make the offline result meaningless. A sensible extension would add weather or generation **forecasts** available before the target period.

## Evaluation

There is no random train/test split. The last block of feature-complete observations is held out.

The baseline is:

```text
prediction(t) = demand(t - 48 half-hours)
```

The training command records model MAE, RMSE, MAPE and seasonal-naive MAE. I do not hard-code a score in this README because the result should come from the exact data snapshot used for the run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

gridcast fetch
gridcast train
```

For a quicker ingestion check:

```bash
gridcast fetch --limit 5000
```

Generated data and model files stay out of git:

```text
data/processed/neso_demand.parquet
artifacts/gridcast.joblib
artifacts/gridcast.metrics.json
```

## Serving

```bash
uvicorn gridcast.api:app --reload
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`

The saved bundle carries the expected feature names. `/predict` rejects missing and unexpected fields rather than silently changing the inference shape.

A container is included for the API, with the model artifact mounted at runtime:

```bash
docker build -t gridcast-gb .
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" gridcast-gb
```

## Why I used a small tree model

The point of the first implementation was to get the temporal boundary, baseline and serving contract right. HistGradientBoosting is fast to retrain locally and handles non-linear interactions without a large tuning surface. A neural forecaster would be an experiment after the baseline is stable, not a prerequisite for the project to look more sophisticated.

## Checks

```bash
ruff check src tests
pytest -q
```

The model card in `docs/model_card.md` covers intended use, feature availability and the main failure modes.

## Next questions

The next useful work would be rolling-origin backtests, forecast weather features and error slicing by season / settlement period. I would also add drift checks around recent demand distributions before adding more model classes.

```text
src/gridcast/       ingestion, features, model, API and CLI
tests/              unit tests
docs/model_card.md  model boundaries
.github/workflows/  CI
```

**Prasanth Balisetty**