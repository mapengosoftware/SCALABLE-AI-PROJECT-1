import os
import joblib
import pandas as pd

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'imdb_sample.csv')

# Load model
model = joblib.load(MODEL_PATH)

# Load data
data = pd.read_csv(DATA_PATH)
X = data['text']
y = data['label']

# Evaluate
accuracy = model.score(X, y)
print(f"✅ Evaluation Accuracy: {accuracy:.2f}")
