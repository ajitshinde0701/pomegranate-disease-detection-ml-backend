from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from predict import predict_disease
from utils.validation import validate_image

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Backend Running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["file"]

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(image_path)

    # Validate
    if not validate_image(image_path):
        return jsonify({
            "error": "Invalid image"
        })

    result = predict_disease(image_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)