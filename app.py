from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
from utils import load_keras_model_from_drive, preprocess_array_image, predict

app = FastAPI(title="Knee OA Classifier API")

FILE_ID = "1Z4w3t6eZ7GEpOo_tsKwoqrWbl_zCPx4R"
model = load_keras_model_from_drive(FILE_ID)

# -------------------
# Serve HTML UI
# -------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# -------------------
# Prediction endpoint
# -------------------
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
