from http import HTTPStatus
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from pydantic import BaseModel

from sentiments import evaluation, predict
from sentiments.monitor_prom_middleware import (
    monitor_requests,
    update_system_metrics,
)

app = FastAPI(title="Sentiment-Classifier")

app.middleware("http")(monitor_requests)


class PredictionRequest(BaseModel):
    data: str
    experiment_name: str = "Sentiment-Classifier"
    metric: str = "val_loss"
    mode: str = "ASC"


class EvaluationRequest(BaseModel):
    file_path: str
    experiment_name: str = "Sentiment-Classifier"
    metric: str = "val_loss"
    mode: str = "ASC"
    num_samples: Optional[int] = 1000
    result_fp: Optional[str] = ""


@app.get("/")
def home():
    """Health check"""
    print(
        "API working as expected. Now head over to http://localhost:8000/docs."
    )
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }

    return response


@app.post("/predict/")
async def _predict(request: PredictionRequest):
    results = predict.predict(
        data=request.data,
        experiment_name=request.experiment_name,
        metric=request.metric,
        mode=request.mode,
    )
    return {"result": results}


@app.post("/evaluate/")
async def _evaluate(request: EvaluationRequest):
    metrics = evaluation.evaluate(
        file_path=request.file_path,
        experiment_name=request.experiment_name,
        metric=request.metric,
        mode=request.mode,
        num_samples=request.num_samples,
        result_path=request.result_fp,
    )
    return {"result": metrics}


@app.get("/metrics")
async def metrics():
    update_system_metrics()
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug",
    )
