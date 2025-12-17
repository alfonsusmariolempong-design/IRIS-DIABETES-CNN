import cv2
from .image_preprocessing import preprocess_image

img = cv2.imread("dataset/train/normal/1_Normal_Kiri.jpg")
out = preprocess_image(img)

print("Preprocessing OK, shape", out.shape)