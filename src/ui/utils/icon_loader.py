"""Utility functions for loading and processing icons."""

import os
import logging
from PIL import Image, ImageTk, ImageDraw

logger = logging.getLogger(__name__)


def load_circular_icon(icon_name, size=(60, 60)):
    """
    Load an icon and make it circular.

    Args:
        icon_name: Name of the icon file
        size: Tuple of (width, height)

    Returns:
        ImageTk.PhotoImage or None
    """
    try:
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "icons",
            icon_name
        )

        if not os.path.exists(icon_path):
            logger.warning(f"Icon not found: {icon_path}")
            return None

        # Load and resize
        icon_img = Image.open(icon_path)
        icon_img = icon_img.resize(size, Image.Resampling.LANCZOS)

        # Create circular mask
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)

        # Apply mask
        icon_img.putalpha(mask)

        return ImageTk.PhotoImage(icon_img)

    except Exception as e:
        logger.error(f"Failed to load icon: {e}")
        return None


def load_window_icon(icon_name):
    """
    Load an icon for window title bar.

    Args:
        icon_name: Name of the icon file

    Returns:
        ImageTk.PhotoImage or None
    """
    try:
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "icons",
            icon_name
        )

        if not os.path.exists(icon_path):
            logger.warning(f"Icon not found: {icon_path}")
            return None

        icon_img = Image.open(icon_path)
        return ImageTk.PhotoImage(icon_img)

    except Exception as e:
        logger.error(f"Failed to load window icon: {e}")
        return None
