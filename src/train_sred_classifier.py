from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Find the main project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
DATA_PATH = BASE_DIR / "data" / "sred_training_data.csv"
MODEL_PATH = BASE_DIR / "sred_classifier.joblib"


# Load the training data
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create the model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("Model performance:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"Model saved as {MODEL_PATH}")