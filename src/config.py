"""
Configuration file for Face Recognition App
Centralizes all paths, thresholds, and settings
"""
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
EDGEFACE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "edgeface", "checkpoints")
EDGEFACE_MODEL_NAME = "edgeface_xs_gamma_06"
EDGEFACE_MODEL_PATH = os.path.join(EDGEFACE_CHECKPOINT_DIR, f"{EDGEFACE_MODEL_NAME}.pt")

FACE_DETECTOR_MODEL = os.path.join(PROJECT_ROOT, "models", "face_detection_yunet_2023mar.onnx")

# Data paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, "embeddings.npy")

# Recognition settings
RECOGNITION_THRESHOLD = 0.8  # Distance threshold for face recognition
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence for face detection
MIN_FACE_SIZE = 50           # Minimum face size in pixels

# Training settings
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 10
BATCH_SIZE = 32
TRIPLET_MARGIN = 1.0

# UI settings
WINDOW_TITLE = "Face Recognition System"
WINDOW_SIZE = "1200x720"
MIN_WINDOW_SIZE = (800, 600)
VIDEO_FPS = 30  # Frames per second for video feed

# Model input size
MODEL_INPUT_SIZE = (112, 112)

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Create necessary directories
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EDGEFACE_CHECKPOINT_DIR, exist_ok=True)

# Call on import
ensure_directories()
