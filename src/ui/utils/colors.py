"""Color palette for Pendo UI."""

# Primary colors
CYAN = "#00d4ff"
GREEN = "#00ff88"
PINK = "#ff0066"

# Background colors
BG_DARK = "#1a1f2e"
BG_DARKER = "#0a0a0a"
BG_MEDIUM = "#2a3f5e"

# Text colors
TEXT_WHITE = "#ffffff"
TEXT_GRAY = "#88a0c0"

# Transparent variations
def hex_with_alpha(hex_color, alpha):
    """Convert hex color with alpha value (0-255)."""
    return f"#{0:02x}{alpha:02x}{int(hex_color[5:7], 16):02x}"
