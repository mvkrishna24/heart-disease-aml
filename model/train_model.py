"""
Heart Disease Prediction - Model Training Script
Dataset: UCI Cleveland Heart Disease Dataset
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"\nShape after cleaning: {df.shape}")

    X = df.drop('target', axis=1)
    y = df['target']
    return X, y


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1-score : {f1_score(y_test, preds):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, preds)}")
    print(f"\nClassification Report:\n{classification_report(y_test, preds)}")
    return accuracy_score(y_test, preds)


def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'heart_disease_dataset.csv')

    X, y = load_and_preprocess(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)
    lr_acc = evaluate_model("Logistic Regression", lr, X_test_sc, y_test)

    cv_lr = cross_val_score(lr, X_train_sc, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy (LR): {cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

    # --- Model 2: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train_sc, y_train)
    rf_acc = evaluate_model("Random Forest", rf, X_test_sc, y_test)

    cv_rf = cross_val_score(rf, X_train_sc, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy (RF): {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

    # --- Select best model ---
    if rf_acc >= lr_acc:
        best_model, best_name, best_acc = rf, "Random Forest", rf_acc
    else:
        best_model, best_name, best_acc = lr, "Logistic Regression", lr_acc

    print(f"\n>>> Best Model: {best_name}  |  Accuracy: {best_acc:.4f}")

    # --- Save artifacts ---
    with open(os.path.join(base_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(base_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    model_info = {
        'name': best_name,
        'accuracy': round(best_acc * 100, 2),
        'features': list(X.columns)
    }
    with open(os.path.join(base_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump(model_info, f)

    print(f"\nSaved: model.pkl, scaler.pkl, model_info.pkl  =>  {base_dir}")
    return model_info


if __name__ == '__main__':
    train()
