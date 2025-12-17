import cv2
from pipeline.pipeline import full_pipeline

def load_image_for_cnn(image_path):
    return full_pipeline(image_path)