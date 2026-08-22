import numpy as np
import pandas as pd

from gridcast.features import build_features
from gridcast.model import load_artifact, train_and_evaluate


def test_training_writes_artifact_and_metrics(tmp_path):
    timestamps = pd.date_range("2026-01-01", periods=1800, freq="30min")
    period = np.arange(len(timestamps))
    demand = 30000 + 3500 * np.sin(2 * np.pi * period / 48) + 500 * np.sin(2 * np.pi * period / 336)
    frame = pd.DataFrame({"timestamp": timestamps, "national_demand_mw": demand})
    features = build_features(frame)

    artifact_path = tmp_path / "gridcast.joblib"
    result = train_and_evaluate(features, artifact_path=artifact_path, holdout_periods=96)
    bundle = load_artifact(artifact_path)

    assert artifact_path.exists()
    assert result.metrics["model_mae_mw"] >= 0
    assert result.metrics["seasonal_naive_mae_mw"] >= 0
    assert bundle["feature_columns"]
    assert bundle["target"] == "national_demand_mw"
