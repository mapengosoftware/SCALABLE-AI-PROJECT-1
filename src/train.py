import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Define base directory (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'imdb_sample.csv')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Ensure labels are strings
df['label'] = df['label'].astype(str)

# Drop labels that appear fewer than 2 times (needed for stratify)
df = df.groupby('label').filter(lambda x: len(x) >= 2)

# Check if enough data remains
if df['label'].nunique() < 2:
    raise ValueError("Not enough valid classes to train a model.")

# Features and labels
X = df['text']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),     # unigrams + bigrams
        max_df=0.9,
        min_df=1
    )),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, MODEL_PATH)

# Evaluate model
y_pred = pipeline.predict(X_test)
acc = pipeline.score(X_test, y_test)

print(f"\n✅ Model accuracy: {acc:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
