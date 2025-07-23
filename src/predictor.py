import os
import joblib

# Load model once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.pkl')
model = joblib.load(MODEL_PATH)

def predict_review(text: str) -> str:
    return model.predict([text])[0]
