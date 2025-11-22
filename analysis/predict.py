#!/usr/bin/env python3
"""
Load trained model and produce top-K predicted numbers with probabilities.
"""

import argparse
import pandas as pd
import joblib
import numpy as np

def last_row_features(df, n_lags):
    df = df.sort_values('date').reset_index(drop=True)
    last = df.iloc[-n_lags:]['number'].astype(int).tolist()
    if len(last) < n_lags:
        raise ValueError("Not enough rows to build lag features")
    # order: lag_1 is previous draw, so reverse last list
    last = last[::-1]
    return np.array(last).reshape(1, -1)

def predict(args):
    data = pd.read_csv(args.data, parse_dates=['date'])
    meta = joblib.load(args.model)
    clf = meta['model']
    n_lags = meta.get('n_lags', 8)

    feat = last_row_features(data, n_lags)
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(feat)[0]
        classes = clf.classes_
        sorted_idx = probs.argsort()[::-1]
        for i in sorted_idx[:args.topk]:
            print(f"{classes[i]}: {probs[i]:.4f}")
    else:
        pred = clf.predict(feat)
        print("Predicted:", pred[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample_data.csv', help='CSV history')
    parser.add_argument('--model', default='model.joblib')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()
    predict(args)
