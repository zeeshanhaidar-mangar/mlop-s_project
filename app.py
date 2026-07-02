import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Load models
# -----------------------------
MODEL_PATH = os.path.join('models', 'RandomForest.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("Models loaded successfully.")

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


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

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        pred = model.predict(features_scaled)[0]
        label = label_encoder.inverse_transform([pred])[0]

        return jsonify({
            "success": True,
            "prediction": label
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


# -----------------------------
# IMPORTANT: only run locally
# -----------------------------
if __name__ == "__main__":
    # Safe for Render / local only
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
