from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Adjust sys.path to import from src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'src'))
sys.path.append(SRC_DIR)

# Import your prediction function from src
from predictor import predict_review

app = FastAPI()

class ReviewRequest(BaseModel):
    review: str

@app.get("/")
def root():
    return {"message": "ML API is live"}

@app.post("/predict")
def predict(req: ReviewRequest):
    prediction = predict_review(req.review)
    return {"prediction": prediction}
