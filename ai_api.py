from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the AI model
MODEL_PATH = "model/model.h5"
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get("features", [])
        
        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Make prediction
        prediction = model.predict([features])
        confidence = model.predict_proba([features])

        return jsonify({
            "prediction": int(prediction[0]),
            "confidence": confidence[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
