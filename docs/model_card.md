# GridCast GB Model Card

## Intended use

GridCast is a portfolio-grade forecasting service for learning and demonstrating the engineering of half-hourly Great Britain electricity-demand models. It is designed for comparative forecasting, API serving and reproducible evaluation.

It is **not** an operational NESO forecasting system and must not be used for grid-control or market decisions without substantially broader validation.

## Target

`national_demand_mw` from NESO Historic Demand Data.

## Input contract

The default model uses only features that can be constructed from prior realised demand plus calendar information:

- 30-minute and 60-minute demand lags
- 1-day, 2-day and 7-day demand lags
- rolling 1-day and 7-day history
- half-hour cyclical encoding
- day-of-week cyclical encoding
- month and weekend indicators

Realised future wind/solar outturn is intentionally excluded from the default day-ahead feature contract. A production extension should join *forecast* weather/generation variables available at prediction time.

## Evaluation protocol

- chronological split only
- final block reserved as hold-out
- seasonal-naive baseline: demand from 48 settlement periods earlier
- reported metrics: MAE, RMSE and MAPE
- model is refitted on all labelled data only after hold-out evaluation

## Model

`HistGradientBoostingRegressor` from scikit-learn. The choice is deliberately conservative: it handles non-linear interactions, trains quickly, has a small deployment footprint and allows the repository to focus on the surrounding MLOps contract.

## Known limitations

- Bank holidays, weather forecasts and major system events are not yet modelled.
- Daylight-saving settlement-period complexity deserves explicit production handling.
- The 2026 default resource is a convenient live data source, not sufficient long-run training history on its own.
- Point predictions do not provide calibrated uncertainty intervals.
- Data revisions in the NESO portal can change retrospective observations.

## Monitoring recommendations

A deployed version should monitor:

- MAE and bias by half-hour / day of week
- residual drift
- feature missingness
- lag-feature distribution shift
- data freshness and duplicate timestamps
- performance relative to the seasonal-naive baseline

Retraining should be governed by a validated trigger rather than a fixed schedule alone.
