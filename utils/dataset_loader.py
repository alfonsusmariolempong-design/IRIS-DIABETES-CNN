import os
import numpy as np
from utils.image_loader import load_image_for_cnn

def load_dataset(base_dir):
    X, y = [], []

    for label, cls in enumerate(["control", "diabetes"]):
        class_dir = os.path.join(base_dir, cls)

        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            try:
                img = load_image_for_cnn(img_path)
                X.append(img)
                y.append(label)
            except:
                continue

    return np.array(X), np.array(y)