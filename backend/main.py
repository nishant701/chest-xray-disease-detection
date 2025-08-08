from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.preprocessing import preprocess_image, load_trained_model, predict_disease
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["disease_detection"]
predictions_collection = db["predictions"]

model = load_trained_model()
class_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Pneumonia"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = preprocess_image(contents)
    label, confidence = predict_disease(model, img, class_labels)

    predictions_collection.insert_one({
        "filename": file.filename,
        "predicted_class": label,
        "confidence": confidence,
        "timestamp": datetime.utcnow()
    })

    return {"predicted_class": label, "confidence": confidence}
