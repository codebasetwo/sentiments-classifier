from http import HTTPStatus
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from sentiments import evaluation, predict

app = FastAPI(title="Sentiment-Classifier")


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
        "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."
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


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug",
    )
