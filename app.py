import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "RandomForest.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="centered")
st.title("Iris Flower Predictor")
st.write("Enter the flower measurements below to predict the species.")

try:
    model, scaler, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
with col2:
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)

col3, col4 = st.columns(2)
with col3:
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
with col4:
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
    features_scaled = scaler.transform(features)
    prediction_idx = model.predict(features_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
    st.success(f"Predicted species: {prediction_label}")
    st.caption(f"Supported classes: {', '.join(label_encoder.classes_)}")
