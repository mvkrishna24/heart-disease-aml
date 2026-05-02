"""
Heart Disease Prediction - Flask Application
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


def load_artifacts():
    """Load model, scaler, and metadata from disk."""
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_info.pkl'), 'rb') as f:
        model_info = pickle.load(f)
    return model, scaler, model_info


model, scaler, model_info = load_artifacts()

FIELD_RANGES = {
    'age':      (1,   120,  'Age (years)'),
    'trestbps': (50,  250,  'Resting Blood Pressure (mm Hg)'),
    'chol':     (50,  600,  'Serum Cholesterol (mg/dl)'),
    'thalach':  (50,  250,  'Max Heart Rate Achieved'),
    'oldpeak':  (0.0, 10.0, 'ST Depression (oldpeak)'),
}


def validate_inputs(form):
    """Return (cleaned_values_list, error_string_or_None)."""
    errors = []

    def get_float(key, label, lo, hi):
        raw = form.get(key, '').strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return None
        return val

    age      = get_float('age',      'Age',                1,   120)
    trestbps = get_float('trestbps', 'Resting BP',        50,   250)
    chol     = get_float('chol',     'Cholesterol',       50,   600)
    thalach  = get_float('thalach',  'Max Heart Rate',    50,   250)
    oldpeak  = get_float('oldpeak',  'ST Depression',    0.0,  10.0)

    def get_int_choice(key, label, choices):
        raw = form.get(key, '').strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            val = int(raw)
        except ValueError:
            errors.append(f"{label} must be an integer.")
            return None
        if val not in choices:
            errors.append(f"{label} must be one of {choices}.")
            return None
        return val

    sex     = get_int_choice('sex',     'Sex',            [0, 1])
    cp      = get_int_choice('cp',      'Chest Pain Type',[0, 1, 2, 3])
    fbs     = get_int_choice('fbs',     'Fasting Blood Sugar', [0, 1])
    restecg = get_int_choice('restecg', 'Resting ECG',   [0, 1, 2])
    exang   = get_int_choice('exang',   'Exercise Angina',[0, 1])
    slope   = get_int_choice('slope',   'ST Slope',      [0, 1, 2])
    ca      = get_int_choice('ca',      'Major Vessels',  [0, 1, 2, 3, 4])
    thal    = get_int_choice('thal',    'Thalassemia',   [0, 1, 2, 3])

    if errors:
        return None, '; '.join(errors)

    features = [age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal]
    return features, None


@app.route('/')
def index():
    return render_template('index.html', model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    features, error = validate_inputs(request.form)

    if error:
        return render_template('index.html',
                               model_info=model_info,
                               error=error,
                               form_data=request.form)

    features_arr = np.array(features, dtype=float).reshape(1, -1)
    features_scaled = scaler.transform(features_arr)

    prediction = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)
    disease_prob = round(float(proba[1]) * 100, 1)

    result = {
        'prediction': prediction,
        'label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected',
        'confidence': confidence,
        'disease_prob': disease_prob,
        'is_positive': prediction == 1,
    }

    input_summary = {
        'Age': int(features[0]),
        'Sex': 'Male' if features[1] == 1 else 'Female',
        'Chest Pain': ['Typical Angina', 'Atypical Angina',
                       'Non-anginal Pain', 'Asymptomatic'][int(features[2])],
        'Resting BP': f"{int(features[3])} mm Hg",
        'Cholesterol': f"{int(features[4])} mg/dl",
        'Fasting BS': 'High (>120)' if features[5] == 1 else 'Normal',
        'Max HR': int(features[7]),
        'Exercise Angina': 'Yes' if features[8] == 1 else 'No',
        'ST Depression': features[9],
    }

    return render_template('result.html',
                           result=result,
                           model_info=model_info,
                           input_summary=input_summary)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', model_info=model_info,
                           error='Page not found.'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', model_info=model_info,
                           error='Internal server error. Please try again.'), 500


if __name__ == '__main__':
    app.run(debug=True)
