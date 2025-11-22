#!/usr/bin/env python3
"""
Train a simple RandomForest classifier to predict the next draw number
from lag features. This is an educational baseline.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import joblib
import os

def build_lag_features(df, col='number', n_lags=8):
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df[col].shift(lag)
    return df

def prepare_dataset(df, n_lags=8):
    df = build_lag_features(df, n_lags=n_lags)
    df = df.dropna().reset_index(drop=True)
    X = df[[f'lag_{i}' for i in range(1, n_lags+1)]].astype(int)
    y = df['number'].astype(int)
    return X, y, df

def train(args):
    df = pd.read_csv(args.data, parse_dates=['date'])
    X, y, df_full = prepare_dataset(df, n_lags=args.n_lags)

    # Optionally restrict number range (infer unique labels)
    labels = np.sort(y.unique())
    print(f"Unique labels: {labels.shape[0]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    try:
        top3 = top_k_accuracy_score(y_val, clf.predict_proba(X_val), k=3)
    except Exception:
        top3 = None

    print(f"Validation accuracy: {acc:.4f}")
    if top3 is not None:
        print(f"Top-3 accuracy: {top3:.4f}")

    # Save model + metadata
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump({
        'model': clf,
        'n_lags': args.n_lags,
    }, args.model)
    print(f"Saved model to {args.model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample_data.csv', help='CSV with columns: date,number')
    parser.add_argument('--model', default='model.joblib', help='Output model path')
    parser.add_argument('--n_lags', type=int, default=8)
    parser.add_argument('--n_estimators', type=int, default=200)
    args = parser.parse_args()
    train(args)
