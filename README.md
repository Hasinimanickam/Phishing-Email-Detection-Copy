# Phishing Email Detection System

A Machine Learning-based web application that detects whether an email is **Safe** or **Phishing** using Natural Language Processing (NLP) and Machine Learning techniques.

The application is developed using Python and Flask, with a responsive interface that provides email analysis, prediction probabilities, suspicious indicators, and visual insights.

## Project Overview

Phishing attacks are a common cybersecurity threat used to steal sensitive information through fraudulent emails.

This project uses a Machine Learning model to analyse email content and classify it as Safe or Phishing.

## Features

- **Email Classification:** Predicts whether an email is Safe or Phishing.
- **Prediction Probability:** Displays Safe and Phishing probability percentages.
- **Suspicious Indicators:** Highlights potentially suspicious email characteristics.
- **Security Recommendations:** Provides safety guidance based on the analysis.
- **Data Visualization:** Displays model insights through Bar, Pie, and Line charts.
- **Confusion Matrix:** Visualizes the classification performance.
- **Prediction History:** Stores previous predictions using SQLite.
- **Dark and Light Modes:** Allows users to switch between themes.
- **Responsive Design:** Supports desktop and mobile screens.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF Vectorization |
| ML Algorithm | Logistic Regression |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |

## Machine Learning Model

The system uses **TF-IDF Vectorization** to convert email text into numerical features.

A **Logistic Regression** classifier is trained to distinguish between Safe and Phishing emails.

Additional URL-related and suspicious keyword features are included to support classification.

### Model Performance

**Accuracy: 97.72%**

The model was evaluated using a separate test dataset.

| Metric | Safe | Phishing |
|---|---:|---:|
| Precision | 0.98 | 0.98 |
| Recall | 0.99 | 0.96 |
| F1-Score | 0.98 | 0.97 |

## Project Structure


Phishing_Email_Detection/
│
├── app.py
├── inspect_dataset.py
├── preprocess_data.py
├── train_model.py
│
├── dataset/
│   ├── emails.csv
│   └── cleaned_emails.csv
│
├── model/
│   ├── phishing_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metrics.json
│
├── static/
│   ├── style.css
│   └── confusion_matrix.png
│
├── templates/
│   └── index.html
│
└── .gitignore


## Installation and Setup

### 1. Clone the Repository


git clone https://github.com/Hasinimanickam/Phishing-Email-Detection-Copy.git


### 2. Navigate to the Project Folder


cd Phishing-Email-Detection-Copy


### 3. Create a Virtual Environment


python -m venv venv


### 4. Activate the Virtual Environment (Windows)


venv\Scripts\activate

### 5. Install Dependencies


pip install flask pandas numpy scikit-learn joblib matplotlib seaborn scipy


### 6. Run the Application

python app.py


### 7. Open in Browser

Visit:


http://127.0.0.1:5000


## How It Works

1. The user enters an email message.
2. The application processes the email content.
3. TF-IDF converts the text into numerical features.
4. Logistic Regression analyses the extracted features.
5. The application predicts Safe or Phishing.
6. Prediction probabilities and security recommendations are displayed.
7. The prediction is stored in the SQLite history database.

## Future Improvements

- Explore additional Machine Learning algorithms.
- Improve detection of sophisticated phishing emails.
- Expand the dataset with more diverse email examples.
- Evaluate model performance on new and unseen email sources.

## Author

**Hasini Manickam**

B.Tech – Artificial Intelligence and Data Science

## Disclaimer

This project is developed for educational and cybersecurity awareness purposes. Machine Learning predictions are not guaranteed to identify every phishing email. Always verify suspicious messages independently.

---

⭐ If you find this project interesting, consider giving the repository a star!