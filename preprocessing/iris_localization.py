import cv2
import numpy as np

def detect_pupil(gray):
    """
    Deteksi pupil menggunakan area gelap
    """
    thresh = np.percentile(gray, 5)
    binary = np.uint8(gray < thresh) * 255
    binary = cv2.medianBlur(binary, 7)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)

    return int(x), int(y), int(r)


def detect_iris(gray):
    """
    Deteksi iris menggunakan Hough Circle
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=80,
        maxRadius=200
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    return int(x), int(y), int(r)

def localize_iris(gray):
    pupil = detect_pupil(gray)
    iris = detect_iris(gray)

    if pupil is None or iris is None:
        return None
    
    return {
        "pupil": pupil, #(x, y, z)
        "iris": iris    #(x, y, z)
    }