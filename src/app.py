from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from predict import predict_custom_image  # Import the function from predict.py

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file!"})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Use predict.py to classify the image
        predicted_digit = predict_custom_image(file_path)

        if predicted_digit is None:
            return jsonify({"error": "Prediction failed!"})

        return jsonify({"prediction": predicted_digit})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)  # Use port 10000 for Render

