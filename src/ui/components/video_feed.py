"""Video feed component for camera display."""

import cv2
import logging
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class VideoFeed:
    """Camera video feed with scanning effects."""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.current_frame = None
        self.photo = None
        self.wave_rings = []

    def update(self, recognizer=None, scanning=False):
        """
        Update video feed and optionally perform recognition.

        Args:
            recognizer: Face recognizer instance
            scanning: Whether currently scanning

        Returns:
            Tuple of (recognition_data, success)
        """
        ret, frame = self.cap.read()
        recognition_data = None

        if not ret or frame is None:
            return None, False

        try:
            self.current_frame = frame

            # Perform recognition if scanning
            if scanning and recognizer:
                display_frame, results = recognizer.recognize_faces(frame)

                # Check for recognized faces
                if results and len(results) > 0:
                    result = results[0]
                    if result['name'] != "Unknown":
                        recognition_data = result
            else:
                display_frame = frame

            # Resize for display
            display_frame = cv2.resize(display_frame, (640, 480))

            # Convert to RGB
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Create PIL image
            img = Image.fromarray(rgb)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=img)

            return recognition_data, True

        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return None, False

    def draw(self, canvas, scanning=False):
        """Draw video feed on canvas."""
        if not self.photo:
            return

        w = canvas.winfo_width()
        h = canvas.winfo_height()

        # Draw video feed centered
        canvas.create_image(
            w // 2, h // 2,
            image=self.photo,
            tags="video"
        )

        # Add scanning wave effects on top
        if scanning:
            self._draw_wave_effects(canvas, w, h)

    def _draw_wave_effects(self, canvas, w, h):
        """Draw wave ring effects during scanning."""
        cx, cy = w // 2, h // 2

        for ring in self.wave_rings:
            alpha = int(100 * (1 - ring / 250))
            if alpha > 0:
                canvas.create_oval(
                    cx - ring, cy - ring,
                    cx + ring, cy + ring,
                    outline=f"#{0:02x}{alpha:02x}{200:02x}",
                    width=2,
                    tags="video"
                )

    def add_wave_ring(self):
        """Add a new wave ring."""
        self.wave_rings.append(0)

    def update_wave_rings(self):
        """Update wave ring animations."""
        self.wave_rings = [r + 10 for r in self.wave_rings if r < 250]

    def get_current_frame(self):
        """Get current camera frame."""
        return self.current_frame

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
