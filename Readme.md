Iris Diabetes Detection using CNN

Project ini adalah sistem deteksi diabetes berbasis citra iris mata menggunakan
metode Convolutional Neural Network (CNN) dengan preprocessing semipolar transformation.
Sistem ini dirancang untuk keperluan penelitian dan dapat diintegrasikan sebagai API
menggunakan Flask serta di-deploy ke Railway dan dihubungkan ke FlutterFlow.

Struktur Proyek

project_diabetes_iris_cnn/
├── app.py
├── config.py
├── requirements.txt
├── Procfile
├── README.md
│
├── uploads/
│
├── preprocessing/
│   ├── image_preprocessing.py
│   ├── iris_localization.py
│   ├── iris_normalization.py
│   └── semipolar_transformation.py
│
├── cnn_model/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── iris_semipolar_cnn.h5
│
├── utils/
│   ├── image_loader.py
│   ├── dataset_loader.py
│   ├── pipeline.py
│   └── metrics.py
│
└── data/
    ├── train/
    │   ├── diabetes/
    │   └── normal/
    └── test/
        ├── diabetes/
        └── normal/

Alur Sistem

1. Input citra iris mata
2. Image preprocessing (grayscale, enhancement)
3. Iris localization (pupil & iris)
4. Semipolar transformation
5. Normalisasi citra
6. CNN training / inference
7. Output klasifikasi (diabetes / normal)

Training Model

Jalankan perintah berikut dari root project:
python -m cnn_model.train

Model terbaik akan disimpan sebagai:
cnn_model/iris_semipolar_cnn.h5

Evaluasi Model

python -m cnn_model.evaluate

Menjalankan API Lokal

python app.py

Akses API:
http://127.0.0.1:8080/predict

Deploy ke Railway

Gunakan Procfile:
web: gunicorn app:app

Pastikan requirements.txt sudah lengkap.

Catatan

- Dataset wajib sudah terlabel (normal & diabetes)
- Model harus dilatih terlebih dahulu sebelum inference
- Proyek ini ditujukan untuk penelitian, bukan diagnosis medis
