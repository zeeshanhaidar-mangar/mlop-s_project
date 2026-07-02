import joblib
import numpy as np
import streamlit as st

model = joblib.load('models/RandomForest.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

st.title("Iris Classifier 🌸")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    x = np.array([[sl, sw, pl, pw]])
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]

    st.success(f"Prediction: {label}")
