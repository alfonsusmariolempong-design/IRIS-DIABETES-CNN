import numpy as np
import cv2
from preprocessing.semipolar_extraction import iris_ring_to_rect
from config import IMG_SIZE


def circle_to_contour(cx, cy, r, points=360):
    """
    Convert circle parameters to contour points
    """
    contour = []
    for t in np.linspace(0, 2 * np.pi, points):
        x = int(cx + r * np.cos(t))
        y = int(cy + r * np.sin(t))
        contour.append([x, y])

    return np.array(contour).reshape((-1, 1, 2)).astype(np.int32)


def normalize_iris(img, pupil, iris):
    """
    Input:
        img   : original BGR image
        pupil : (cx, cy, r)
        iris  : (cx, cy, r)

    Output:
        semipolar normalized iris image
    """

    pupilContour = circle_to_contour(*pupil)
    irisContour  = circle_to_contour(*iris)

    polar = iris_ring_to_rect(
        pupilCenter=(pupil[0], pupil[1]),
        pupilContour=pupilContour,
        irisContour=irisContour,
        Image=img
    )

    polar_gray = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY)
    polar_gray = cv2.resize(polar_gray, IMG_SIZE)

    return polar_gray