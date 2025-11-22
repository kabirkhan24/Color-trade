import streamlit as st
import pandas as pd
import joblib
import tempfile
import os
import subprocess

st.title("Amar Club Wingo — Analysis (Educational)")

st.markdown("""
Upload a CSV history (columns: date,number). You can either:
- Upload a CSV and train a quick model, or
- Upload a previously saved `model.joblib` and run predictions.
""")

uploaded = st.file_uploader("Upload history CSV", type=['csv'])
uploaded_model = st.file_uploader("Upload model.joblib (optional)", type=['joblib', 'pkl'])

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['date'])
    st.write("Preview:", df.tail(10))
    n_lags = st.slider("Number of lag features", min_value=3, max_value=20, value=8)
    if st.button("Train model on uploaded data"):
        with st.spinner("Training..."):
            tmpdir = tempfile.mkdtemp()
            csv_path = os.path.join(tmpdir, "data.csv")
            model_path = os.path.join(tmpdir, "model.joblib")
            df.to_csv(csv_path, index=False)
            # Run training script
            subprocess.check_call(["python", "analysis/train.py", "--data", csv_path, "--model", model_path, "--n_lags", str(n_lags)])
            st.success("Training finished. Loading model...")
            model = joblib.load(model_path)
            st.write("Model saved at:", model_path)
            st.download_button("Download model.joblib", data=open(model_path,"rb").read(), file_name="model.joblib")
            st.session_state['model_path'] = model_path

if uploaded_model is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
    tmp.write(uploaded_model.read())
    tmp.flush()
    tmp.close()
    st.session_state['model_path'] = tmp.name
    st.success("Uploaded model loaded")

if 'model_path' in st.session_state:
    st.write("Model available at:", st.session_state['model_path'])
    topk = st.number_input("Top-K predictions", min_value=1, max_value=50, value=5)
    if st.button("Show predictions"):
        if uploaded is None:
            st.error("Upload a history CSV so we can build features for the latest draw.")
        else:
            model_meta = joblib.load(st.session_state['model_path'])
            clf = model_meta['model']
            n_lags = model_meta.get('n_lags', 8)
            df_local = df.sort_values('date').reset_index(drop=True)
            if len(df_local) < n_lags:
                st.error("Not enough history rows for selected n_lags")
            else:
                last = df_local.iloc[-n_lags:]['number'].astype(int).tolist()[::-1]
                feat = [last]
                if hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba(feat)[0]
                    classes = clf.classes_
                    pairs = sorted(list(zip(classes, probs)), key=lambda x: x[1], reverse=True)[:topk]
                    st.table({"number":[p[0] for p in pairs],"probability":[round(float(p[1]),4) for p in pairs]})
                else:
                    st.write("Predicted:", clf.predict(feat)[0])
