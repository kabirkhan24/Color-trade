# Color-trade — Analysis / Number Prediction (Educational)

This repository originally contained a Flutter project. This branch adds an `analysis/` folder with a reproducible, educational analytics prototype for "Amar Club Wingo" number analysis.

Important: This code is for data analysis, visualization, and experimentation only. It does NOT (and cannot) guarantee 100% prediction accuracy for random draws or lottery-style games. Use responsibly.

Contents added:
- analysis/requirements.txt — Python dependencies
- analysis/sample_data.csv — small example history (fake/demo data)
- analysis/train.py — training script (feature engineering + RandomForest)
- analysis/predict.py — load model and produce top-K predictions with probabilities
- analysis/streamlit_app.py — small Streamlit UI to try uploading history and showing predictions
- analysis/README.md — instructions for the analysis code

Quick start (local):
1. Create and activate a Python 3.9+ virtualenv
2. cd analysis
3. pip install -r requirements.txt
4. python train.py --data sample_data.csv --model model.joblib
5. python predict.py --model model.joblib --topk 10
Or run the UI:
   streamlit run streamlit_app.py

This is educational — results depend entirely on data quality, feature engineering, and randomness.