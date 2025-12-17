# semiPolar_extraction.py
import cv2
import numpy as np
from config import IMG_SIZE

def iris_ring_to_rect(pupilCenter, pupilContour, irisContour, image):
    normalHeight, normalWidth = IMG_SIZE

    polarImg = np.zeros(
        (normalHeight, normalWidth, image.shape[2]),
        dtype=np.uint8
    )

    polCol = 0
    theta = 0.0

    while theta < 360:
        px, py = pupilCenter

        # Cari batas pupil
        while cv2.pointPolygonTest(
            pupilContour, (int(px), int(py)), False
        ) == 1:
            px += np.cos(np.deg2rad(theta))
            py += np.sin(np.deg2rad(theta))

        x_pupil = int(px - np.cos(np.deg2rad(theta)))
        y_pupil = int(py - np.sin(np.deg2rad(theta)))

        # Cari batas iris
        while cv2.pointPolygonTest(
            irisContour, (int(px), int(py)), False
        ) == 1:
            px += np.cos(np.deg2rad(theta))
            py += np.sin(np.deg2rad(theta))

        x_iris = int(px - np.cos(np.deg2rad(theta)))
        y_iris = int(py - np.sin(np.deg2rad(theta)))

        r = 0.0
        polRow = 0
        while r <= 1.0 and polRow < normalHeight:
            x = int((1 - r) * x_pupil + r * x_iris)
            y = int((1 - r) * y_pupil + r * y_iris)
            polarImg[polRow, polCol] = image[y, x]
            r += 1.0 / (normalHeight - 1)
            polRow += 1

        theta += 360.0 / normalWidth
        polCol += 1

    return polarImg