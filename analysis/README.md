This analysis folder contains a small prototype showing how you can:

- Load a history CSV (date, number)
- Create simple lag features (previous draws)
- Train a RandomForest classifier to estimate probability distribution over numbers
- Serve predictions through a Streamlit UI

Notes:
- The model and pipeline are intentionally simple to keep the example self-contained.
- For production or stronger experiments, add more data, advanced features (frequency windows, time-series models, embeddings), cross-validation, and probabilistic calibration.
- Do NOT assume guaranteed accuracy for truly random processes.