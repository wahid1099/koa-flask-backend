










from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from utils import load_keras_model_from_drive, preprocess_array_image, predict
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI(title="Knee OA Classifier API")

# Mount static folder for HTML/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google Drive model ID
FILE_ID = "1Z4w3t6eZ7GEpOo_tsKwoqrWbl_zCPx4R"
model = load_keras_model_from_drive(FILE_ID)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read())).convert('RGB')
        img_array = np.array(image)
        preprocessed = preprocess_array_image(img_array)
        result = predict(model, preprocessed)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
