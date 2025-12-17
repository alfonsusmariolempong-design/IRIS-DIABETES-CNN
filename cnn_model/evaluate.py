import cv2
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MODEL_PATH
from .model import build_model

model = tf.keras.models.load_model(MODEL_PATH)

def evaluate_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

    prob = model.predict(img)[0][0]
    label = "diabetes" if prob >= 0.5 else "control"

    return label, prob

if __name__ == "_main_":
    label, score = evaluate_image("test.jpg")
    print(label, score)