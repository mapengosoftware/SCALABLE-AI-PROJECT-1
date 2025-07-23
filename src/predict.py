import os
import joblib

# Define base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the model
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.pkl')

# Load the trained pipeline
pipeline = joblib.load(MODEL_PATH)

# Prediction loop
while True:
    review = input("Enter a movie review (or type 'exit' to quit): ").strip()
    if review.lower() == 'exit':
        break
    prediction = pipeline.predict([review])[0]
    print(f"Prediction: {prediction}\n")
