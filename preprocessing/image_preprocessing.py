import cv2
import numpy as np

def preprocess_image(img):
    """
    Input  : BGR image
    Output : grayscale image (enhanced)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced