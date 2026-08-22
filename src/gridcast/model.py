from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from gridcast.features import TARGET, feature_columns


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, float]
    artifact_path: Path


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def train_and_evaluate(
    feature_frame: pd.DataFrame,
    artifact_path: str | Path = "artifacts/gridcast.joblib",
    holdout_periods: int = 48 * 14,
) -> TrainingResult:
    """Train after comparing against a 24-hour seasonal-naive hold-out baseline."""
    if len(feature_frame) <= holdout_periods + 100:
        raise ValueError("Insufficient history for the requested chronological hold-out")

    columns = feature_columns(feature_frame)
    train = feature_frame.iloc[:-holdout_periods].copy()
    test = feature_frame.iloc[-holdout_periods:].copy()

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[columns], train[TARGET])
    predictions = model.predict(test[columns])
    baseline = test["demand_lag_48"].to_numpy()
    truth = test[TARGET].to_numpy()

    metrics = {
        "model_mae_mw": float(mean_absolute_error(truth, predictions)),
        "model_rmse_mw": float(mean_squared_error(truth, predictions) ** 0.5),
        "model_mape_pct": _mape(truth, predictions),
        "seasonal_naive_mae_mw": float(mean_absolute_error(truth, baseline)),
        "holdout_periods": float(holdout_periods),
    }

    # Deployment bundle is fitted on all available labelled history only after evaluation.
    model.fit(feature_frame[columns], feature_frame[TARGET])
    artifact = {
        "model": model,
        "feature_columns": columns,
        "metrics": metrics,
        "target": TARGET,
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return TrainingResult(metrics=metrics, artifact_path=path)


def load_artifact(path: str | Path = "artifacts/gridcast.joblib") -> dict:
    return joblib.load(Path(path))
