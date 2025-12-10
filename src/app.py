from flask import Flask, render_template, request, jsonify
import os
from essaie1 import read_plate_text, predict_plate_category, determine_plate_type
import cv2
import numpy as np
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    data = request.form.get("cropped_image")
    mode = request.form.get("ocr_mode", "A")  # default Auto

    if not data:
        return jsonify({"error": "No image received"}), 400

    # Convert base64 image to OpenCV
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save cropped image
    save_path = os.path.join(UPLOAD_FOLDER, "cropped_plate.jpg")
    cv2.imwrite(save_path, img)

    # Run OCR with mode
    plate_text = read_plate_text(img, mode=mode)

    # CNN Prediction
    cnn_result = predict_plate_category(save_path)

    # Heuristic plate type
    plate_type, _ = determine_plate_type(plate_text, img)

    result = {
        "text": plate_text,
        "type": plate_type
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
