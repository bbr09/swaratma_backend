from fastapi import FastAPI, UploadFile, File
import shutil
import os

from predict_raga import predict_raga

app = FastAPI()


# Home route (so / does not show "Not Found")
@app.get("/")
def home():
    return {"status": "Swaratma API running"}


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Create temporary audio file
    temp_file = "temp_audio.wav"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run ML model
        result = predict_raga(temp_file)

        return result

    except Exception as e:

        return {
            "raga": "Error",
            "arohanam": "",
            "avarohanam": "",
            "message": str(e)
        }

    finally:
        # Delete temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)