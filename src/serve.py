from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    client = storage.Client()
    bucket = client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print("Model da duoc tai xuong tu cloud storage.")


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Phai cung cap dung 10 dac trung")
    pred = model.predict([req.features])[0]
    label = "thu_nhap_cao" if pred == 1 else "thu_nhap_thap"
    return {"prediction": int(pred), "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
