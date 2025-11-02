import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.recognition import FaceRecognizer
from src.core.training import train_model
from src.ui.main_window import MainWindow
from config import EDGEFACE_MODEL_PATH, FACE_DETECTOR_MODEL

if __name__ == "__main__":
    logger.info("Starting Face Recognition Application")

    # Check if required models exist
    if not os.path.exists(EDGEFACE_MODEL_PATH):
        logger.error(f"EdgeFace model not found at {EDGEFACE_MODEL_PATH}")
        print(f"ERROR: EdgeFace model not found at {EDGEFACE_MODEL_PATH}")
        print("Please ensure the model file exists before running the application.")
        sys.exit(1)

    if not os.path.exists(FACE_DETECTOR_MODEL):
        logger.error(f"Face detector model not found at {FACE_DETECTOR_MODEL}")
        print(f"ERROR: Face detector model not found at {FACE_DETECTOR_MODEL}")
        print("Please ensure the model file exists before running the application.")
        sys.exit(1)

    try:
        # Initialize recognizer
        logger.info("Initializing face recognizer...")
        recognizer = FaceRecognizer(model_path=EDGEFACE_MODEL_PATH)
        logger.info("Face recognizer initialized successfully")

        # Start GUI
        logger.info("Starting GUI...")
        app = MainWindow(recognizer=recognizer, trainer=train_model)
        app.mainloop()

    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        print(f"ERROR: Application failed to start: {e}")
        sys.exit(1)
