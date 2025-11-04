import cv2
import numpy as np
import requests
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.alerting import send_sms_alert
import threading
import datetime

# Load the trained model and class indices
model_path = "models/plant_health_model.keras"
model = load_model(model_path)

class_indices = {
    "Healthy": 0,
    "Disease": 1,
    "Pest": 2,
    # Add more classes related to plant health
}

# Reverse mapping of indices to class names
class_names = {v: k for k, v in class_indices.items()}

# API configuration
API_KEY = "AIzaSyDoc7dPNaA2hpqCnJDlk3ZA7d-rYk7RVA4"
API_ENDPOINT = "https://api.example.com/generate_report"

# A flag to control the video capture loop
running = True

def predict_from_frame(model, frame, class_names):
    """Preprocess the frame and make a prediction for plant-related health issues."""
    frame_resized = cv2.resize(frame, (224, 224))
    frame_preprocessed = preprocess_input(np.expand_dims(frame_resized, axis=0))
    prediction = model.predict(frame_preprocessed)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Only consider plant-related classes for prediction
    if class_id in class_names:
        class_name = class_names[class_id]
    else:
        class_name = "Unknown Class"
    
    return class_name, confidence

def send_prediction_to_api(class_name, confidence, frame):
    """Send prediction details to the API for report generation."""
    timestamp = datetime.datetime.now().isoformat()
    _, frame_encoded = cv2.imencode('.jpg', frame)
    image_data = frame_encoded.tobytes()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "timestamp": timestamp,
        "prediction": class_name,
        "confidence": confidence,
        "image": image_data.hex()  # Convert bytes to hex for JSON compatibility
    }

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Report successfully generated:", response.json())
    else:
        print("Failed to generate report:", response.status_code, response.text)

def real_time_detection_process():
    """Handles video capture and real-time detection for plant health only."""
    global running

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        prediction, confidence = predict_from_frame(model, frame, class_names)

        label = f"{prediction} ({confidence * 100:.2f}%)"
        color = (0, 255, 0) if "Healthy" in prediction else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame with detection label
        cv2.imshow("Plant Health Detection", frame)

        # Send alert and data if plant disease or pest is detected with high confidence
        if confidence > 0.8 and ("Disease" in prediction or "Pest" in prediction):
            alert_message = f"Plant health alert: {prediction} detected!"
            send_sms_alert(alert_message)

            # Send data to the API for report generation
            send_prediction_to_api(prediction, confidence, frame)

        # Stop the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()

def start_video_capture():
    """Start the real-time detection process in a separate thread."""
    video_thread = threading.Thread(target=real_time_detection_process)
    video_thread.start()

if __name__ == "__main__":
    start_video_capture()
