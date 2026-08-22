from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gridcast.model import load_artifact

app = FastAPI(
    title="GridCast GB",
    version="0.1.0",
    description="Great Britain electricity-demand forecasting service",
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="Feature values matching the versioned model feature contract"
    )


class PredictionResponse(BaseModel):
    predicted_national_demand_mw: float
    model_version: str = "0.1.0"


@lru_cache(maxsize=1)
def _artifact() -> dict:
    path = Path(os.getenv("GRIDCAST_ARTIFACT", "artifacts/gridcast.joblib"))
    if not path.exists():
        raise FileNotFoundError(path)
    return load_artifact(path)


@app.get("/health")
def health() -> dict[str, bool | str]:
    try:
        _artifact()
        return {"status": "ok", "model_ready": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_ready": False}


@app.get("/model-info")
def model_info() -> dict:
    try:
        artifact = _artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc
    return {
        "model_version": "0.1.0",
        "target": artifact["target"],
        "feature_columns": artifact["feature_columns"],
        "evaluation_metrics": artifact["metrics"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        artifact = _artifact()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model artifact is not available") from exc

    required = artifact["feature_columns"]
    missing = [column for column in required if column not in request.features]
    unexpected = [column for column in request.features if column not in required]
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )

    row = pd.DataFrame([{column: request.features[column] for column in required}])
    prediction = max(0.0, float(artifact["model"].predict(row)[0]))
    return PredictionResponse(predicted_national_demand_mw=prediction)
