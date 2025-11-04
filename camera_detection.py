import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils.alerting import send_sms_alert
import threading
import time
import json

# Constants for paths
MODEL_PATH = "models/plant_health_model.keras"
CLASS_INDICES_PATH = "class_indices.json"

# Load model and class indices from JSON
try:
    # Load the class indices from JSON file
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    # Load the trained model
    model = load_model(MODEL_PATH, compile=False)
    print("Model and class indices loaded successfully.")
except Exception as e:
    print(f"Error loading model or class indices: {e}")
    exit()

# Reverse mapping for class indices to class names
class_names = {v: k for k, v in class_indices.items()}

# A flag for controlling video capture loop
running = True

def predict_from_frame(model, frame, class_names):
    """Preprocess the frame and make prediction."""
    frame_resized = cv2.resize(frame, (224, 224))  # Resize once for better performance
    frame_preprocessed = preprocess_input(np.expand_dims(frame_resized, axis=0))  # Preprocess for MobileNetV2
    prediction = model.predict(frame_preprocessed)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Ensure that class_id is within the range of class_names
    class_name = class_names.get(class_id, "Unknown Class")
    
    return class_name, confidence

def real_time_detection_process():
    """Handles video capture and real-time detection."""
    global running
    cap = cv2.VideoCapture(0)  # Use default webcam (0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
    print("Webcam is successfully opened.")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Predict class and confidence from the current frame
        prediction, confidence = predict_from_frame(model, frame, class_names)

        # Display the prediction on the frame
        label = f"{prediction} ({confidence * 100:.2f}%)"
        color = (0, 255, 0) if "Healthy" in prediction else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame with the prediction
        cv2.imshow("Real-Time Plant Health Detection", frame)

        # Send SMS alert if prediction indicates disease or pest
        if confidence > 0.8 and ("Diseased" in prediction or "Pest" in prediction):
            alert_message = f"Plant health alert: {prediction} detected!"
            send_sms_alert(alert_message)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting the application.")
            break

    cap.release()
    cv2.destroyAllWindows()

def start_video_capture():
    """Start the real-time detection in a separate thread."""
    video_thread = threading.Thread(target=real_time_detection_process)
    video_thread.start()

if __name__ == "__main__":
    start_video_capture()

    # Keep the program running until the webcam is stopped by user input
    while running:
        pass

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
