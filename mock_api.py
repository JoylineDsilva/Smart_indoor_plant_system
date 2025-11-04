from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    file_path = data.get("file_path", "Unknown File")
    
    # Mock response
    return jsonify({
        "prediction": "Healthy",
        "confidence": 0.98,
        "details": f"This is a mock response for the file {file_path}."
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)
