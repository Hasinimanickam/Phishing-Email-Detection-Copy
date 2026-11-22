
import pandas as pd
import re
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("dataset/cleaned_emails.csv")

df["label"] = df["Email Type"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

df = df.dropna(subset=["label"])

# Extract URL features
def extract_url_features(text):
    text = str(text).lower()

    urls = re.findall(r'https?://\S+|www\.\S+', text)

    suspicious_words = [
        "verify", "account", "password", "login",
        "bank", "urgent", "click", "confirm"
    ]

    return [
        len(urls),
        int(bool(re.search(r'bit\.ly|tinyurl|t\.co', text))),
        int("http://" in text),
        sum(word in text for word in suspicious_words)
    ]

# Separate text and labels
X_text = df["Email Text"].astype(str)
y = df["label"]

# Split BEFORE fitting the vectorizer
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit TF-IDF only on training data
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Extract URL features
X_train_url = csr_matrix([
    extract_url_features(text)
    for text in X_train_text
])

X_test_url = csr_matrix([
    extract_url_features(text)
    for text in X_test_text
])

# Combine features
X_train = hstack([X_train_tfidf, X_train_url])
X_test = hstack([X_test_tfidf, X_test_url])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL TRAINING COMPLETED")
print("------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Generate detailed evaluation metrics
report = classification_report(
    y_test,
    y_pred,
    target_names=["Safe", "Phishing"],
    output_dict=True
)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Safe", "Phishing"]
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create folders
os.makedirs("static", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Save evaluation metrics for charts
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "labels": ["Safe", "Phishing"],
    "precision": [
        round(report["Safe"]["precision"] * 100, 2),
        round(report["Phishing"]["precision"] * 100, 2)
    ],
    "recall": [
        round(report["Safe"]["recall"] * 100, 2),
        round(report["Phishing"]["recall"] * 100, 2)
    ],
    "f1_score": [
        round(report["Safe"]["f1-score"] * 100, 2),
        round(report["Phishing"]["f1-score"] * 100, 2)
    ]
}

with open("model/model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save confusion matrix image
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Email Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("static/confusion_matrix.png")
plt.close()

# Save model and vectorizer
joblib.dump(model, "model/phishing_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved!")
print("Confusion matrix image saved!")
print("Evaluation metrics saved to model/model_metrics.json!")
