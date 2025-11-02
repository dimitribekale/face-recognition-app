"""Scan button component for top-right corner."""

import math
import tkinter as tk
from ..utils.colors import CYAN, GREEN, BG_DARK, BG_MEDIUM


class ScanButton:
    """Circular scan button with pulsating animation."""

    def __init__(self, canvas, x, y, radius=35):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.radius = radius
        self.scanning = False
        self.progress = 0
        self.pulse_value = 0
        self.icon = None

    def set_icon(self, icon):
        """Set the icon image for the button."""
        self.icon = icon

    def set_scanning(self, scanning):
        """Set scanning state."""
        self.scanning = scanning
        if not scanning:
            self.progress = 0

    def update_progress(self, progress):
        """Update scan progress (0-100)."""
        self.progress = progress

    def update_pulse(self, pulse_value):
        """Update pulse animation value."""
        self.pulse_value = pulse_value

    def draw(self):
        """Draw the scan button."""
        if not self.scanning:
            self._draw_idle()
        else:
            self._draw_scanning()

    def _draw_idle(self):
        """Draw button in idle state with pulsating ring."""
        # Pulsating ring
        pulse_alpha = int(100 + 50 * math.sin(self.pulse_value / 10))
        for i in range(2):
            radius = self.radius + i * 3 + (self.pulse_value % 5)
            self.canvas.create_oval(
                self.x - radius, self.y - radius,
                self.x + radius, self.y + radius,
                outline=f"#{0:02x}{pulse_alpha:02x}{255:02x}",
                width=2,
                tags="nav"
            )

        # Main circle
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=BG_DARK,
            outline=CYAN,
            width=3,
            tags=("nav", "scan_btn")
        )

        # Glow effect
        for i in range(3, 0, -1):
            alpha = 40 - i * 10
            self.canvas.create_oval(
                self.x - self.radius - i*2, self.y - self.radius - i*2,
                self.x + self.radius + i*2, self.y + self.radius + i*2,
                outline=f"#{0:02x}{alpha:02x}{200:02x}",
                width=1,
                tags="nav"
            )

        # Icon in center
        if self.icon:
            self.canvas.create_image(
                self.x, self.y,
                image=self.icon,
                tags="nav"
            )
        else:
            # Fallback: simple circle
            self.canvas.create_oval(
                self.x - 15, self.y - 15,
                self.x + 15, self.y + 15,
                fill=CYAN,
                outline="",
                tags="nav"
            )

    def _draw_scanning(self):
        """Draw button in scanning state with progress ring."""
        # Progress ring (fills clockwise)
        extent = -(self.progress * 3.6)
        self.canvas.create_arc(
            self.x - 40, self.y - 40,
            self.x + 40, self.y + 40,
            start=90, extent=extent,
            outline=GREEN,
            width=6,
            style=tk.ARC,
            tags="nav"
        )

        # Background ring
        self.canvas.create_oval(
            self.x - 40, self.y - 40,
            self.x + 40, self.y + 40,
            outline=BG_MEDIUM,
            width=6,
            tags="nav"
        )

        # Progress text
        self.canvas.create_text(
            self.x, self.y,
            text=f"{int(self.progress)}%",
            font=("Helvetica Neue", 12, "bold"),
            fill=CYAN,
            tags="nav"
        )

    def is_clicked(self, click_x, click_y):
        """Check if button was clicked."""
        dist = math.sqrt((click_x - self.x)**2 + (click_y - self.y)**2)
        return dist < 40
