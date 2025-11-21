from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.predict import Predictor


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()


# -------------------------------
# CORS (required for Next.js)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Frontend domains can replace "*" later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Load the Thar vs Wrangler model
# (Loaded only once at startup)
# -------------------------------
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "thar_wrangler_mobilenetv2.pth"
)
model_path = os.path.abspath(model_path)

predictor = Predictor(model_path)



# -------------------------------
# Prediction API
# -------------------------------
@app.post("/api/predict")
async def predict_api(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        response = predictor.predict_from_bytes(image_bytes)
        return response

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return {"error": str(e)}



# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Backend Running!"}
