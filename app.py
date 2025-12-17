import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing.iris_localization import detect_iris, detect_pupil

from config import IMG_SIZE, MODEL_PATH
from preprocessing.semipolar_extraction import iris_ring_to_rect

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===============================
# CORE PREDICT
# ===============================
def cnn_predict(img):
    if img is None:
        return None, "invalid_image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ============================
    # VALIDASI FOTO MATA
    # ============================
    pupil = detect_pupil(gray)
    iris  = detect_iris(gray)

    if pupil is None or iris is None:
        return None, "not_eye"

    # ============================
    # SEMIPOLAR TRANSFORMATION
    # ============================
    polar = iris_ring_to_rect(
        pupilCenter=(pupil[0], pupil[1]),
        pupilContour=None,
        irisContour=None,
        Image=img
    )

    polar_gray = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY)
    polar_gray = cv2.resize(polar_gray, IMG_SIZE)
    polar_gray = polar_gray / 255.0

    x = polar_gray.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    prob = float(model.predict(x)[0][0])
    return prob, "ok"

# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded"
        }), 400

    file = request.files["file"]
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    img = cv2.imread(path)

    prob, status = cnn_predict(img)

    if status == "invalid_image":
        return jsonify({
            "status": "error",
            "message": "Invalid image file"
        }), 400

    if status == "not_eye":
        return jsonify({
            "status": "error",
            "message": "Uploaded image is not an eye / iris image"
        }), 422

    return jsonify({
        "status": "success",
        "label": "diabetes" if prob >= 0.5 else "control",
        "score": round(prob, 4),
        "score_percent": round(prob * 100, 2)
    })

# ===============================
@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    img = cv2.imread(path)
    prob, polar = cnn_predict(img)

    _, bpolar = cv2.imencode(".png", polar)

    return jsonify({
        "label": "diabetes" if prob >= 0.5 else "control",
        "score": round(prob, 4),
        "score_percent": round(prob * 100, 2),
        "semipolar_image": base64.b64encode(bpolar).decode()
    })

@app.route("/")
def health():
    return jsonify({"status": "CNN iris diabetes detector OK"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)