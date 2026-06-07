from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from predict import predict_disease
from utils.validation import validate_image

app = Flask(__name__)

# Explicit CORS configuration for production (Render)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "api": "AnarMitra ML Backend",
        "version": "1.0.0",
        "status": "Running",
        "description": "Pomegranate Disease Detection API",
        "endpoints": {
            "GET  /":        "API info (this response)",
            "GET  /health":  "Health check",
            "POST /predict": "Predict disease from image (multipart/form-data, field: 'file')"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend Running"
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "file" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(image_path)

    # Validate
    if not validate_image(image_path):
        return jsonify({
            "error": "Invalid image"
        }), 400

    result = predict_disease(image_path)

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)