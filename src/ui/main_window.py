"""Main window for Pendo face recognition application."""

import tkinter as tk
from tkinter import simpledialog
import os
import logging
import cv2

from config import DATASET_DIR
from .components import AnimatedBackground, ScanButton, VideoFeed, RecognitionCard
from .utils import load_circular_icon, load_window_icon, CYAN

logger = logging.getLogger(__name__)


class MainWindow(tk.Tk):
    """Main application window."""

    def __init__(self, recognizer, trainer):
        super().__init__()

        self.recognizer = recognizer
        self.trainer = trainer

        # Setup window
        self._setup_window()

        # Initialize components
        self._init_components()

        # Setup event handlers
        self._setup_events()

        # Start animation loop
        self.update_idletasks()
        self.after(100, self._update_video)
        self.after(16, self._animate)

    def _setup_window(self):
        """Configure window properties."""
        self.title("Pendo")

        # Window size and position
        width, height = 900, 700
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        self.geometry(f"{width}x{height}+{x}+{y}")
        self.configure(bg="#1a1f2e")

        # Set window icon
        icon = load_window_icon("generated_image.png")
        if icon:
            self.iconphoto(True, icon)
            self._icon_photo = icon

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _init_components(self):
        """Initialize UI components."""
        # Canvas
        self.canvas = tk.Canvas(
            self,
            bg="#1a1f2e",
            highlightthickness=0,
            width=900,
            height=700
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Background
        self.background = AnimatedBackground(900, 700, node_count=20)

        # Video feed
        self.video_feed = VideoFeed(camera_index=0)

        # Scan button
        self.scan_button = ScanButton(
            self.canvas,
            x=900 - 55,
            y=55,
            radius=35
        )
        icon = load_circular_icon("generated_image.png", (60, 60))
        if icon:
            self.scan_button.set_icon(icon)

        # Recognition card
        self.recognition_card = RecognitionCard(width=280, height=500)

        # State
        self.scanning = False
        self.pulse_value = 0
        self.scan_progress = 0
        self.video_running = True

    def _setup_events(self):
        """Setup event handlers."""
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)

    def _update_video(self):
        """Update video feed."""
        if not self.video_running:
            return

        recognition_data, success = self.video_feed.update(
            self.recognizer if self.scanning else None,
            self.scanning
        )

        # Update recognition card if face found
        if recognition_data:
            self.recognition_card.set_data(recognition_data)

        self.after(30, self._update_video)

    def _animate(self):
        """Main animation loop."""
        if not self.video_running:
            return

        # Clear canvas
        self.canvas.delete("all")

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Update animations
        self._update_animations(w, h)

        # Draw layers
        self.background.draw(self.canvas)
        self.video_feed.draw(self.canvas, self.scanning)
        self.recognition_card.draw(self.canvas)
        self._draw_navigation()
        self.scan_button.draw()

        self.after(16, self._animate)

    def _update_animations(self, w, h):
        """Update all animation states."""
        # Pulse for scan button
        self.pulse_value = (self.pulse_value + 1) % 100
        self.scan_button.update_pulse(self.pulse_value)

        # Background nodes
        self.background.update(w, h)

        # Scanning animations
        if self.scanning:
            self.scan_progress = (self.scan_progress + 1.5) % 100
            self.scan_button.update_progress(self.scan_progress)

            # Wave rings
            if int(self.scan_progress) % 20 == 0:
                self.video_feed.add_wave_ring()
            self.video_feed.update_wave_rings()

        # Recognition card animation
        target_x = w - 160 if self.recognition_card.data else w + 300
        self.recognition_card.update_position(target_x)

    def _draw_navigation(self):
        """Draw top navigation bar."""
        # Menu icon (hamburger)
        for i in range(3):
            y = 25 + i * 8
            self.canvas.create_line(
                25, y, 45, y,
                fill=CYAN, width=2, tags=("nav", "menu_icon")
            )

    def _on_mouse_motion(self, event):
        """Handle mouse motion for cursor changes."""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        cursor = ""

        # Check menu icon
        if 25 <= event.x <= 45 and 17 <= event.y <= 41:
            cursor = "hand2"
        # Check scan button
        elif self.scan_button.is_clicked(event.x, event.y):
            cursor = "hand2"
        # Check action buttons
        elif self.recognition_card.is_button_clicked(event.x, event.y, h):
            cursor = "hand2"

        self.canvas.config(cursor=cursor)

    def _on_click(self, event):
        """Handle mouse clicks."""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # Menu icon
        if 25 <= event.x <= 45 and 17 <= event.y <= 41:
            logger.info("Menu clicked")
            # TODO: Implement menu
            return

        # Scan button
        if self.scan_button.is_clicked(event.x, event.y):
            self._toggle_scanning(w)
            return

        # Action buttons
        button = self.recognition_card.is_button_clicked(event.x, event.y, h)
        if button:
            self._handle_action_button(button, w)

    def _toggle_scanning(self, window_width):
        """Toggle scanning state."""
        self.scanning = not self.scanning
        self.scan_button.set_scanning(self.scanning)

        if self.scanning:
            logger.info("Scan started")
            self.scan_progress = 0
            self.video_feed.wave_rings = []
            self.recognition_card.clear_data()
            self.recognition_card.x_offset = window_width + 300
        else:
            logger.info("Scan stopped")
            self.scan_progress = 0
            self.video_feed.wave_rings = []

    def _handle_action_button(self, button, window_width):
        """Handle action button clicks."""
        if button == "rescan":
            logger.info("Rescan clicked")
            self.scanning = True
            self.scan_button.set_scanning(True)
            self.scan_progress = 0
            self.video_feed.wave_rings = []
            self.recognition_card.clear_data()
            self.recognition_card.x_offset = window_width + 300

        elif button == "save":
            logger.info("Save clicked")
            frame = self.video_feed.get_current_frame()
            if frame is not None:
                name = simpledialog.askstring("Register", "Enter name:", parent=self)
                if name:
                    success, message = self.recognizer.register_face(frame, name)
                    if success:
                        user_dir = os.path.join(DATASET_DIR, name)
                        os.makedirs(user_dir, exist_ok=True)
                        existing = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')])
                        cv2.imwrite(os.path.join(user_dir, f"{existing + 1}.jpg"), frame)
                        logger.info(f"Face saved for {name}")

        elif button == "share":
            logger.info("Share clicked (not implemented yet)")

    def _on_closing(self):
        """Clean up on window close."""
        self.scanning = False
        self.video_running = False
        self.video_feed.release()
        self.destroy()
