import cv2
import numpy as np


from preprocessing.image_preprocessing import preprocess_image
from preprocessing.iris_localization import detect_iris, detect_pupil
from preprocessing.iris_normalization import normalize_iris
from preprocessing.semipolar_extraction import iris_ring_to_rect

def test_pipeline(image_path):
    print("📥 Load image...")
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Gagal membaca gambar")

    print("✅ Image loaded:", img.shape)

    print("⚙️ Preprocessing...")
    pre = preprocess_image(img)
    print("✅ Preprocessing OK:", pre.shape)

    print("🎯 Iris detect...")
    iris = detect_iris, detect_pupil(pre)
    print("✅ Iris detect:", iris.shape)

    print("🔄 Iris normalization...")
    norm = normalize_iris(iris)
    print("✅ Iris normalized:", norm.shape)

    print("🌀 Semipolar transform...")
    semi = iris_ring_to_rect(norm)
    print("✅ Semipolar output:", semi.shape)

    print("\n🎉 PIPELINE BERHASIL TANPA ERROR")

if __name__ == "_main_":
    test_image = "dataset/train/normal/1_Normal_Kiri.jpg"  # ganti sesuai file kamu
    test_pipeline(test_image)