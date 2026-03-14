from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from predict_raga import predict_raga

app = FastAPI()


@app.post("/predict")

async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:

        tmp.write(await file.read())

        temp_path = tmp.name


    result = predict_raga(temp_path)

    os.remove(temp_path)

    return result