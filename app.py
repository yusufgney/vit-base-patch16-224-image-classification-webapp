from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import os
from model_utils import load_model, predict_image

app = FastAPI(title="ViT Image Classifier API")

# Global model variables
processor, model, device = None, None, None

@app.on_event("startup")
async def startup_event():
    global processor, model, device
    processor, model, device = load_model()
    if not model:
        print("Failed to load model on startup.")

# Serve static files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictions = predict_image(image, processor, model, device, threshold=0.1)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
