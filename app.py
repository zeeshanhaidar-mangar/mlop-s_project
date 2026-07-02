import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

# Paths
MODEL_PATH = os.path.join('models', 'RandomForest.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

# Load models
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print("Error loading models:", e)

# Home route (UI)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction_idx = model.predict(features_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_idx])[0]

        return jsonify({
            "success": True,
            "prediction": prediction_label
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# IMPORTANT: Render-compatible server start
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
