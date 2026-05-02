# HeartGuard AI – Heart Disease Prediction Web App

A production-ready Flask web application that predicts cardiovascular disease risk
using machine learning, trained on the UCI Cleveland Heart Disease Dataset.

---

## Features

- **Dual-model training** – Logistic Regression vs Random Forest with automatic best-model selection
- **13-feature clinical form** – validated input for all standard cardiac biomarkers
- **Real-time prediction** – probability score + confidence level on every submission
- **Responsive UI** – works on desktop, tablet, and mobile
- **Render-ready** – gunicorn + `runtime.txt` for zero-config cloud deployment

---

## Dataset

| Property | Value |
|----------|-------|
| Source   | UCI Cleveland Heart Disease Dataset |
| Rows     | 303 patients |
| Features | 13 clinical attributes |
| Target   | Binary (1 = heart disease, 0 = no disease) |

**Feature descriptions**

| Column    | Description |
|-----------|-------------|
| age       | Age in years |
| sex       | 1 = male, 0 = female |
| cp        | Chest pain type (0–3) |
| trestbps  | Resting blood pressure (mm Hg) |
| chol      | Serum cholesterol (mg/dl) |
| fbs       | Fasting blood sugar > 120 mg/dl (1/0) |
| restecg   | Resting ECG results (0–2) |
| thalach   | Maximum heart rate achieved |
| exang     | Exercise induced angina (1/0) |
| oldpeak   | ST depression induced by exercise |
| slope     | Slope of peak exercise ST segment (0–2) |
| ca        | Major vessels coloured by fluoroscopy (0–4) |
| thal      | Thalassemia type (0–3) |

---

## Project Structure

```
heart-disease-ml/
├── model/
│   ├── heart_disease_dataset.csv   ← dataset
│   ├── train_model.py              ← training script
│   ├── model.pkl                   ← saved best model (generated)
│   ├── scaler.pkl                  ← fitted StandardScaler (generated)
│   └── model_info.pkl              ← metadata (generated)
├── templates/
│   ├── index.html                  ← prediction form
│   └── result.html                 ← results page
├── static/
│   └── style.css                   ← responsive CSS
├── app.py                          ← Flask application
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

---

## Installation & Running Locally

### 1. Clone / navigate to the project

```bash
cd heart-disease-ml
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (generates model.pkl, scaler.pkl, model_info.pkl)

```bash
python model/train_model.py
```

### 5. Start the Flask development server

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

---

## Deployment on Render

1. Push your repository to GitHub (make sure `model.pkl`, `scaler.pkl`, and
   `model_info.pkl` are **committed** – they are not in `.gitignore`).

2. On [render.com](https://render.com) create a new **Web Service**:
   - **Build Command:** `pip install -r requirements.txt && python model/train_model.py`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3

3. Deploy. Render will auto-detect `runtime.txt` for the Python version.

---

## Model Performance (typical results)

| Metric    | Logistic Regression | Random Forest |
|-----------|---------------------|---------------|
| Accuracy  | ~85 %               | ~88 %         |
| Precision | ~84 %               | ~87 %         |
| Recall    | ~87 %               | ~90 %         |
| F1-score  | ~85 %               | ~88 %         |

*Exact numbers vary slightly with each training run due to train/test split.*

---

## Disclaimer

This application is for **educational purposes only** and does not constitute
medical advice. Always consult a qualified healthcare professional for diagnosis
and treatment decisions.

---

## Tech Stack

- **Backend** – Python 3.11, Flask 3, scikit-learn, pandas, numpy
- **Frontend** – HTML5, CSS3 (custom, no framework), vanilla JS
- **Deployment** – Render, gunicorn
