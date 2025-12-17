# config.py

# ===============================
# IMAGE CONFIG
# ===============================
IMG_HEIGHT = 201
IMG_WIDTH  = 720
IMG_CHANNELS = 1   # grayscale

IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ===============================
# TRAINING CONFIG
# ===============================
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4

# ===============================
# PATH
# ===============================
DATASET_DIR = "dataset"
MODEL_PATH = "cnn_model/iris_semipolar_cnn.h5"