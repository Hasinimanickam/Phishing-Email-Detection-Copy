
from flask import Flask, render_template, request
import joblib
import re
import json
import sqlite3
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Database configuration
DATABASE = "prediction_history.db"


# Create database and history table
def init_db():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT NOT NULL,
            result TEXT NOT NULL,
            safe_probability REAL,
            phishing_probability REAL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# Save prediction history
def save_prediction(email_text, result,
                    safe_probability, phishing_probability):

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        INSERT INTO prediction_history
        (email_text, result, safe_probability,
         phishing_probability, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        email_text,
        result,
        safe_probability,
        phishing_probability,
        datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")
    ))

    conn.commit()
    conn.close()


# Get previous predictions
def get_prediction_history():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    history = conn.execute("""
        SELECT * FROM prediction_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return history


# Initialize database
init_db()


# Load trained model and vectorizer
model = joblib.load("model/phishing_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Load model evaluation metrics
with open("model/model_metrics.json", "r") as file:
    model_metrics = json.load(file)


# Model performance
accuracy = model_metrics["accuracy"]


# Dataset statistics
safe_count = 10979
phishing_count = 6557
total_emails = safe_count + phishing_count

safe_percentage = round(
    (safe_count / total_emails) * 100, 2
)

phishing_percentage = round(
    (phishing_count / total_emails) * 100, 2
)


# Extract URL features
def extract_url_features(text):

    text = str(text).lower()

    urls = re.findall(
        r'https?://\S+|www\.\S+', text
    )

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


# Identify suspicious indicators
def detect_suspicious_patterns(text):

    text_lower = text.lower()
    patterns = []

    urls = re.findall(
        r'https?://\S+|www\.\S+', text_lower
    )

    if urls:
        patterns.append(f"Contains {len(urls)} link(s)")

    if re.search(r'bit\.ly|tinyurl|t\.co', text_lower):
        patterns.append("Contains a shortened URL")

    if "http://" in text_lower:
        patterns.append("Contains a non-encrypted HTTP link")

    suspicious_words = [
        "verify", "account", "password", "login",
        "bank", "urgent", "click", "confirm"
    ]

    found_words = [
        word for word in suspicious_words
        if word in text_lower
    ]

    if found_words:
        patterns.append(
            "Suspicious words: " + ", ".join(found_words)
        )

    return patterns


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    email_text = ""

    safe_probability = None
    phishing_probability = None
    suspicious_patterns = []
    recommendations = []

    if request.method == "POST":

        email_text = request.form.get(
            "email", ""
        ).strip()

        if email_text:

            # Extract features
            text_features = vectorizer.transform(
                [email_text]
            )

            url_features = csr_matrix([
                extract_url_features(email_text)
            ])

            final_features = hstack([
                text_features,
                url_features
            ])

            # Prediction
            prediction = model.predict(
                final_features
            )[0]

            # Calculate probabilities
            probabilities = model.predict_proba(
                final_features
            )[0]

            classes = list(model.classes_)

            safe_probability = round(
                float(
                    probabilities[classes.index(0)]
                ) * 100, 2
            )

            phishing_probability = round(
                float(
                    probabilities[classes.index(1)]
                ) * 100, 2
            )

            if prediction == 1:

                result = "Phishing"

                recommendations = [
                    "Do not click links in this email.",
                    "Do not share passwords or banking details.",
                    "Do not download unexpected attachments.",
                    "Verify the sender through an official channel."
                ]

            else:

                result = "Safe"

                recommendations = [
                    "Always verify the sender's email address.",
                    "Avoid opening unexpected attachments.",
                    "Do not share sensitive information unnecessarily."
                ]

            # Detect suspicious indicators
            suspicious_patterns = detect_suspicious_patterns(
                email_text
            )

            # Save prediction in SQLite
            save_prediction(
                email_text,
                result,
                safe_probability,
                phishing_probability
            )

    # Retrieve previous predictions
    prediction_history = get_prediction_history()

    return render_template(
        "index.html",
        result=result,
        email_text=email_text,
        accuracy=accuracy,
        safe_count=safe_count,
        phishing_count=phishing_count,
        total_emails=total_emails,
        safe_percentage=safe_percentage,
        phishing_percentage=phishing_percentage,
        safe_probability=safe_probability,
        phishing_probability=phishing_probability,
        suspicious_patterns=suspicious_patterns,
        recommendations=recommendations,
        model_metrics=model_metrics,
        prediction_history=prediction_history
    )


if __name__ == "__main__":
    app.run(debug=True)
