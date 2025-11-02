"""Recognition card component for displaying recognition results."""

import math
from ..utils.colors import CYAN, GREEN, BG_DARK, BG_MEDIUM, TEXT_WHITE, TEXT_GRAY


class RecognitionCard:
    """Glassmorphic card displaying recognition results."""

    def __init__(self, width=280, height=500):
        self.width = width
        self.height = height
        self.x_offset = 1200  # Start off-screen
        self.alpha = 0
        self.data = None

    def set_data(self, recognition_data):
        """Set recognition data to display."""
        self.data = recognition_data

    def clear_data(self):
        """Clear recognition data."""
        self.data = None

    def update_position(self, target_x, speed=20):
        """Update card position with animation."""
        if self.data:
            # Slide in
            if self.x_offset > target_x:
                self.x_offset -= speed
            else:
                self.x_offset = target_x

            if self.alpha < 1.0:
                self.alpha += 0.05
        else:
            # Slide out
            if self.x_offset < 1200:
                self.x_offset += speed
            self.alpha = max(0, self.alpha - 0.05)

    def draw(self, canvas):
        """Draw the recognition card."""
        if not self.data:
            return

        h = canvas.winfo_height()
        cx = self.x_offset
        cy = h // 2

        # Background
        self._draw_background(canvas, cx, cy)

        # User icon
        self._draw_user_icon(canvas, cx, cy)

        # Name and confidence
        self._draw_info(canvas, cx, cy)

        # Action buttons
        self._draw_action_buttons(canvas, cx, cy)

    def _draw_background(self, canvas, cx, cy):
        """Draw card background with glow."""
        # Glassmorphic background
        canvas.create_rectangle(
            cx - self.width//2, cy - self.height//2,
            cx + self.width//2, cy + self.height//2,
            fill=BG_DARK,
            outline="",
            tags="card"
        )

        # Outer glow
        for i in range(5, 0, -1):
            alpha = 80 - i * 15
            canvas.create_rectangle(
                cx - self.width//2 - i*2, cy - self.height//2 - i*2,
                cx + self.width//2 + i*2, cy + self.height//2 + i*2,
                outline=f"#{0:02x}{alpha:02x}{200:02x}",
                width=2,
                tags="card"
            )

        # Main border
        canvas.create_rectangle(
            cx - self.width//2, cy - self.height//2,
            cx + self.width//2, cy + self.height//2,
            outline=CYAN,
            width=3,
            tags="card"
        )

        # Corner decorations
        self._draw_corners(canvas, cx, cy)

    def _draw_corners(self, canvas, cx, cy):
        """Draw L-shaped corner decorations."""
        corner_size = 15
        corners = [
            (-self.width//2, -self.height//2),
            (self.width//2, -self.height//2),
            (-self.width//2, self.height//2),
            (self.width//2, self.height//2)
        ]

        for dx, dy in corners:
            x, y = cx + dx, cy + dy
            canvas.create_line(
                x, y, x + (corner_size if dx < 0 else -corner_size), y,
                fill=GREEN, width=2, tags="card"
            )
            canvas.create_line(
                x, y, x, y + (corner_size if dy < 0 else -corner_size),
                fill=GREEN, width=2, tags="card"
            )

    def _draw_user_icon(self, canvas, cx, cy):
        """Draw user icon at top of card."""
        icon_y = cy - self.height//2 + 80

        # Circle background
        canvas.create_oval(
            cx - 50, icon_y - 50,
            cx + 50, icon_y + 50,
            fill=BG_MEDIUM,
            outline=CYAN,
            width=2,
            tags="card"
        )

        # User silhouette - head
        canvas.create_oval(
            cx - 18, icon_y - 20,
            cx + 18, icon_y,
            fill=CYAN,
            outline="",
            tags="card"
        )

        # User silhouette - body
        canvas.create_arc(
            cx - 30, icon_y - 5,
            cx + 30, icon_y + 45,
            start=0, extent=180,
            fill=CYAN,
            outline="",
            tags="card"
        )

    def _draw_info(self, canvas, cx, cy):
        """Draw name, confidence bar, and percentage."""
        name = self.data['name']
        confidence = self.data['confidence']

        # Name
        name_y = cy - self.height//2 + 170
        canvas.create_text(
            cx, name_y,
            text=name.upper(),
            font=("Helvetica Neue", 20, "bold"),
            fill=TEXT_WHITE,
            tags="card"
        )

        # Confidence label
        conf_y = name_y + 40
        canvas.create_text(
            cx, conf_y,
            text="MATCH CONFIDENCE",
            font=("Helvetica Neue", 10),
            fill=TEXT_GRAY,
            tags="card"
        )

        # Confidence bar
        self._draw_confidence_bar(canvas, cx, conf_y + 20, confidence)

        # Percentage
        canvas.create_text(
            cx, conf_y + 65,
            text=f"{int(confidence * 100)}%",
            font=("Helvetica Neue", 24, "bold"),
            fill=GREEN,
            tags="card"
        )

    def _draw_confidence_bar(self, canvas, cx, bar_y, confidence):
        """Draw gradient confidence bar."""
        bar_w = 200
        bar_h = 18
        bar_x = cx - bar_w//2

        # Background
        canvas.create_rectangle(
            bar_x, bar_y,
            bar_x + bar_w, bar_y + bar_h,
            fill=BG_MEDIUM,
            outline=CYAN,
            width=1,
            tags="card"
        )

        # Filled portion (gradient)
        filled_w = int(bar_w * confidence)
        for i in range(0, filled_w, 5):
            ratio = i / bar_w
            g = int(255 * ratio)
            b = int(200 - 100 * ratio)
            canvas.create_rectangle(
                bar_x + i, bar_y + 2,
                bar_x + i + 5, bar_y + bar_h - 2,
                fill=f"#{0:02x}{g:02x}{b:02x}",
                outline="",
                tags="card"
            )

    def _draw_action_buttons(self, canvas, cx, cy):
        """Draw action buttons (RESCAN, SAVE, SHARE)."""
        btn_start_y = cy + 50
        btn_spacing = 80
        buttons = [("RESCAN", "↻"), ("SAVE", "↓"), ("SHARE", "→")]

        for i, (label, icon) in enumerate(buttons):
            btn_y = btn_start_y + i * btn_spacing

            # Button glow
            for j in range(2, 0, -1):
                alpha = 40 - j * 10
                canvas.create_oval(
                    cx - 25 - j, btn_y - 25 - j,
                    cx + 25 + j, btn_y + 25 + j,
                    outline=f"#{0:02x}{alpha:02x}{150:02x}",
                    width=1,
                    tags=("card", f"action_{label.lower()}")
                )

            # Button circle
            canvas.create_oval(
                cx - 25, btn_y - 25,
                cx + 25, btn_y + 25,
                fill=BG_MEDIUM,
                outline=CYAN,
                width=2,
                tags=("card", f"action_{label.lower()}")
            )

            # Icon
            canvas.create_text(
                cx, btn_y - 2,
                text=icon,
                font=("Helvetica Neue", 20, "bold"),
                fill=CYAN,
                tags=("card", f"action_{label.lower()}")
            )

            # Label
            canvas.create_text(
                cx, btn_y + 35,
                text=label,
                font=("Helvetica Neue", 10),
                fill=TEXT_GRAY,
                tags=("card", f"action_{label.lower()}")
            )

    def is_button_clicked(self, click_x, click_y, canvas_height):
        """Check if an action button was clicked."""
        if not self.data:
            return None

        btn_start_y = canvas_height // 2 + 50
        btn_spacing = 80
        labels = ["rescan", "save", "share"]

        for i, label in enumerate(labels):
            btn_y = btn_start_y + i * btn_spacing
            dist = math.sqrt((click_x - self.x_offset)**2 + (click_y - btn_y)**2)

            if dist < 30:
                return label

        return None
