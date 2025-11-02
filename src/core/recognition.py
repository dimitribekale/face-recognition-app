import os
import sys
import cv2
import torch
import numpy as np
import logging
from typing import Optional, Tuple, Dict, List

# Add project root to path for edgeface import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from edgeface.backbones import get_model
from config import (
    EDGEFACE_MODEL_PATH, EDGEFACE_MODEL_NAME, FACE_DETECTOR_MODEL,
    EMBEDDINGS_FILE, RECOGNITION_THRESHOLD, CONFIDENCE_THRESHOLD,
    MIN_FACE_SIZE, MODEL_INPUT_SIZE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Face recognition system using EdgeFace model and YuNet detector"""

    def __init__(self, model_path: str = None, recognition_threshold: float = None):
        """
        Initialize face recognizer

        Args:
            model_path: Path to the face recognition model (default: from config)
            recognition_threshold: Distance threshold for recognition (default: from config)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model_path = model_path or EDGEFACE_MODEL_PATH
        self.recognition_threshold = recognition_threshold or RECOGNITION_THRESHOLD

        self.model = self._load_model(self.model_path)
        self.face_detector = self._initialize_detector()
        self.known_face_embeddings: Dict[str, np.ndarray] = {}
        self._load_embeddings()

    def _initialize_detector(self) -> cv2.FaceDetectorYN:
        """Initialize YuNet face detector"""
        try:
            if not os.path.exists(FACE_DETECTOR_MODEL):
                raise FileNotFoundError(f"Face detector model not found at {FACE_DETECTOR_MODEL}")
            detector = cv2.FaceDetectorYN.create(FACE_DETECTOR_MODEL, "", (0, 0))
            logger.info("Face detector initialized successfully")
            return detector
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load face recognition model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            model = get_model(EDGEFACE_MODEL_NAME)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_embeddings(self) -> None:
        """Load known face embeddings from file"""
        try:
            if os.path.exists(EMBEDDINGS_FILE):
                self.known_face_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
                logger.info(f"Loaded {len(self.known_face_embeddings)} face embeddings")
            else:
                logger.info("No existing embeddings found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            self.known_face_embeddings = {}

    def _save_embeddings(self) -> None:
        """Save known face embeddings to file"""
        try:
            np.save(EMBEDDINGS_FILE, self.known_face_embeddings)
            logger.info(f"Saved {len(self.known_face_embeddings)} face embeddings")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def detect_faces(self, frame: np.ndarray) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Detect faces in a frame

        Args:
            frame: Input image frame (BGR)

        Returns:
            List of face bounding boxes [(x, y, w, h), ...] or None if no faces
        """
        h, w, _ = frame.shape
        self.face_detector.setInputSize((w, h))

        try:
            _, faces = self.face_detector.detect(frame)
            if faces is None or len(faces) == 0:
                return None

            # Filter faces by size and confidence
            valid_faces = []
            for face in faces:
                x, y, w, h = map(int, face[:4])
                confidence = face[14] if len(face) > 14 else 1.0  # Get confidence if available

                if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE and confidence >= CONFIDENCE_THRESHOLD:
                    valid_faces.append((x, y, w, h))

            return valid_faces if valid_faces else None
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None

    def _get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for a face image

        Args:
            face_img: Face image (RGB, any size)

        Returns:
            Face embedding vector or None on error
        """
        try:
            # Ensure RGB
            if len(face_img.shape) == 2:  # Grayscale
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)

            # Resize to model input size
            face_img = cv2.resize(face_img, MODEL_INPUT_SIZE)

            # Convert to tensor
            face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            face_tensor = face_tensor.to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(face_tensor).cpu().numpy()

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def register_face(self, frame: np.ndarray, user_name: str) -> Tuple[bool, str]:
        """
        Register a face from a frame

        Args:
            frame: Input image frame (BGR)
            user_name: Name to associate with the face

        Returns:
            (success: bool, message: str)
        """
        # Detect faces
        faces = self.detect_faces(frame)

        if faces is None or len(faces) == 0:
            return False, "No face detected in frame. Please ensure face is visible."

        if len(faces) > 1:
            return False, f"Multiple faces detected ({len(faces)}). Please ensure only one face is visible."

        # Get the detected face
        x, y, w, h = faces[0]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_img = rgb_frame[y:y+h, x:x+w]

        # Generate embedding
        embedding = self._get_face_embedding(face_img)

        if embedding is None:
            return False, "Failed to generate face embedding."

        # Save embedding
        self.known_face_embeddings[user_name] = embedding
        self._save_embeddings()

        logger.info(f"Successfully registered face for '{user_name}'")
        return True, f"Successfully registered '{user_name}'"

    def recognize_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Recognize faces in a frame and draw bounding boxes

        Args:
            frame: Input image frame (BGR)

        Returns:
            (annotated_frame, recognition_results)
            recognition_results: List of dicts with 'name', 'confidence', 'bbox'
        """
        results = []

        # Detect faces
        faces = self.detect_faces(frame)

        if faces is None:
            return frame, results

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        for (x, y, w, h) in faces:
            # Ensure bounding box is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame_width, x + w)
            y2 = min(frame_height, y + h)

            # Skip if invalid box
            if x2 <= x or y2 <= y:
                continue

            # Extract face with boundary-safe coordinates
            face_img = rgb_frame[y:y2, x:x2]

            # Skip if face region is empty
            if face_img.size == 0:
                continue

            # Generate embedding
            embedding = self._get_face_embedding(face_img)

            if embedding is None:
                continue

            # Find closest match
            name = "Unknown"
            min_dist = float('inf')

            if len(self.known_face_embeddings) > 0:
                for known_name, known_embedding in self.known_face_embeddings.items():
                    dist = np.linalg.norm(embedding - known_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        name = known_name

                # Check threshold
                if min_dist > self.recognition_threshold:
                    name = "Unknown"

            # Calculate confidence (inverse of distance, normalized)
            confidence = max(0, 1 - (min_dist / 2)) if min_dist != float('inf') else 0

            # Store result with corrected bbox
            results.append({
                'name': name,
                'confidence': confidence,
                'distance': min_dist if min_dist != float('inf') else None,
                'bbox': (x, y, x2 - x, y2 - y)  # (x, y, width, height)
            })

            # Draw on frame using corrected coordinates
            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)  # Green for known, orange for unknown
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            # Draw name and confidence
            label = f"{name} ({confidence*100:.0f}%)" if name != "Unknown" else "Unknown"
            label_y = max(y - 10, 10)  # Ensure label is visible at top
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, results

    def update_threshold(self, new_threshold: float) -> None:
        """Update recognition threshold"""
        self.recognition_threshold = new_threshold
        logger.info(f"Recognition threshold updated to {new_threshold}")

    def delete_user(self, user_name: str) -> bool:
        """Delete a user's face embedding"""
        if user_name in self.known_face_embeddings:
            del self.known_face_embeddings[user_name]
            self._save_embeddings()
            logger.info(f"Deleted face embedding for '{user_name}'")
            return True
        return False

    def get_registered_users(self) -> List[str]:
        """Get list of registered users"""
        return list(self.known_face_embeddings.keys())
