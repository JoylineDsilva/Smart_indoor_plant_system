import os
import random
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from twilio.rest import Client
import cv2
import requests
from google.cloud import vision


# Initialize Flask app
app = Flask(__name__)

# Twilio configuration
TWILIO_PHONE_NUMBER = '13203029813'
TO_PHONE_NUMBER = '918714827597'
ACCOUNT_SID = 'AC4758c602d583664de6227457148199db'
AUTH_TOKEN = '0a088e7270059e79986241387f84a1d9'

# Initialize Twilio Client
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Plant.id API configuration (Example)
PLANT_ID_API_KEY = 'your_plantid_api_key'

# Google Cloud Vision API configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_your_google_credentials.json"

# Configurations
UPLOAD_FOLDER = 'uploads/'
REPORT_FOLDER = 'reports/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route - Index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Generate report
        report = generate_report(file_path)

        # Send SMS alert if prediction indicates disease or pest
        if report['prediction'] in ["Diseased", "Pest Infested"]:
            send_sms_alert(report)

        # Render report page
        return render_template('report.html', report=report, report_filename=filename)

    return redirect(url_for('index'))

# Route for real-time detection (live camera feed)
@app.route('/real_time_detection_process')
def real_time_detection_process():
    return render_template('real_time.html')

# Video feed route for live camera stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to capture video frames using OpenCV
def generate_camera_frames():
    cap = cv2.VideoCapture(0)  # Use the default camera (0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Example of adding detection overlay text (mock detection)
        prediction = random.choice(["Healthy", "Diseased", "Pest Infested"])
        confidence = random.uniform(0.7, 1.0)
        cv2.putText(frame, f"{prediction} ({confidence * 100:.2f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for downloading report
@app.route('/download_report/<filename>')
def download_report(filename):
    report_path = os.path.join(REPORT_FOLDER, f'{filename}.json')
    return send_file(report_path, as_attachment=True)

# Function to generate a mock report
def generate_report(file_path):
    prediction = random.choice(["Healthy", "Diseased", "Pest Infested"])
    confidence = random.uniform(0.7, 1.0)
    full_report = f"This is a detailed analysis of the plant health for the image {file_path}. The plant appears to be {prediction.lower()}."

    report_data = {
        'prediction': prediction,
        'confidence': confidence,
        'full_report': full_report
    }

    # Save the report to a JSON file
    report_filename = os.path.splitext(os.path.basename(file_path))[0] + '.json'
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    with open(report_path, 'w') as report_file:
        json.dump(report_data, report_file, indent=4)

    return report_data

# Function to send SMS alerts using Twilio
def send_sms_alert(report):
    message = f"Plant health alert! The plant is {report['prediction']} with a confidence of {report['confidence']*100:.2f}%. Please take necessary action."
    try:
        twilio_client.messages.create(
            to=TO_PHONE_NUMBER, 
            from_=TWILIO_PHONE_NUMBER,
            body=message
        )
        print("SMS sent successfully.")
    except Exception as e:
        print(f"Error sending SMS: {e}")

# Plant.id API Prediction
def plantid_predict(image_url):
    url = "https://api.plant.id/v2/identify"
    headers = {
        "Api-Key": PLANT_ID_API_KEY,
        "Content-Type": "application/json",
    }

    data = {
        "images": [image_url],
        "organs": ["leaf"],  # Modify if needed
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()

    if response.status_code == 200 and response_json.get('suggestions'):
        plant = response_json['suggestions'][0]
        return {
            "species": plant.get('plant_name'),
            "confidence": plant.get('probability'),
            "description": plant.get('wiki_description')
        }
    else:
        return {"error": "Could not identify plant"}

# Google Cloud Vision API Prediction
def google_vision_predict(image_url):
    client = vision.ImageAnnotatorClient()

    # Load image from URL
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = image_url
    
    # Perform label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # If successful, return the labels and confidence scores
    result = []
    for label in labels:
        result.append({
            "description": label.description,
            "confidence": label.score
        })
    return result

# Route for AI-based prediction (Plant.id or Google Vision)
@app.route('/predict_ai', methods=['POST'])
def predict_ai():
    try:
        data = request.json
        image_url = data.get("image_url")  # Send image URL from the front-end

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400

        # Choose the AI API (Plant.id or Google Vision)
        api_choice = data.get("api_choice", "google")  # Default to Google Vision if not specified

        if api_choice == "plantid":
            result = plantid_predict(image_url)
        else:
            result = google_vision_predict(image_url)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure the upload and report folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    # Run the app
    app.run(debug=True)
