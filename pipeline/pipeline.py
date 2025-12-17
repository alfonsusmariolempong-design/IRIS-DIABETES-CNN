import cv2
import numpy as np
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.iris_localization import detect_iris, detect_pupil
from preprocessing.iris_normalization import normalize_iris
from preprocessing.semipolar_extraction import iris_ring_to_rect
from config import IMG_SIZE

def full_pipeline(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    if img is (None):
        raise ValueError("Gambar tidak ditemukan")

    # 2. Preprocessing
    img = preprocess_image(img)

    # 3. Iris & pupil localization
    pupil, iris = detect_iris, detect_pupil(img)
    

    # 4. Normalization (optional smoothing)
    norm = normalize_iris(img)

    # 5. Semipolar transformation
    polar = iris_ring_to_rect(
        pupilCenter=(pupil["cx"], pupil["cy"]),
        pupilContour=pupil["contour"],
        irisContour=iris["contour"],
        Image=img
    )

    # 6. Resize untuk CNN
    polar_gray = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY)
    polar_gray = cv2.resize(polar_gray, IMG_SIZE)

    # 7. Normalisasi CNN
    polar_gray = polar_gray / 255.0
    polar_gray = polar_gray.reshape(
        IMG_SIZE[0], IMG_SIZE[1], 1
    )

    return polar_gray