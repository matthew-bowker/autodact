"""Centralized UI style constants for Autodact."""

# Primary colors
PRIMARY_BLUE = "#4a86c8"
PRIMARY_BLUE_HOVER = "#3a6fb0"
PRIMARY_BLUE_PRESSED = "#2d5a8f"

# Neutral colors
BACKGROUND_LIGHT = "#f8f8f8"
BACKGROUND_PANEL = "#fafafa"
BACKGROUND_HIGHLIGHT = "#eef4fc"
BACKGROUND_ACTIVE = "#d6e8fc"

# Text colors
TEXT_PRIMARY = "#222"
TEXT_SECONDARY = "#555"
TEXT_TERTIARY = "#666"
TEXT_DISABLED = "#888"
TEXT_MUTED = "#999"
TEXT_PLACEHOLDER = "#aaa"

# Border colors
BORDER_DEFAULT = "#ddd"
BORDER_LIGHT = "#e8e8e8"
BORDER_DARK = "#aaa"
BORDER_FOCUS = PRIMARY_BLUE

# Semantic colors
SUCCESS_GREEN = "#5cb85c"
SUCCESS_GREEN_HOVER = "#4cae4c"
ERROR_RED = "#d9534f"
ERROR_RED_HOVER = "#c9302c"
WARNING_YELLOW = "#f0ad4e"

# Spacing constants
SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 12
SPACING_LG = 16
SPACING_XL = 20

# Border radius
RADIUS_SM = 3
RADIUS_MD = 4
RADIUS_LG = 8


def button_style(bg_color: str, hover_color: str, text_color: str = "white") -> str:
    """Generate consistent button stylesheet."""
    return (
        f"QPushButton {{ "
        f"background-color: {bg_color}; color: {text_color}; "
        f"font-size: 14px; font-weight: bold; border-radius: {RADIUS_MD}px; "
        f"border: none; padding: 8px 16px; }} "
        f"QPushButton:hover {{ background-color: {hover_color}; }} "
        f"QPushButton:pressed {{ background-color: {hover_color}; }} "
        f"QPushButton:disabled {{ background-color: #ccc; color: {TEXT_DISABLED}; }}"
    )


def panel_style() -> str:
    """Generate consistent panel/group box stylesheet."""
    return (
        f"QGroupBox {{ "
        f"border: 1px solid {BORDER_LIGHT}; border-radius: {RADIUS_MD}px; "
        f"margin-top: 20px; padding-top: 20px; background-color: {BACKGROUND_PANEL}; }} "
        f"QGroupBox::title {{ "
        f"subcontrol-origin: margin; subcontrol-position: top left; "
        f"left: 12px; padding: 0 6px; color: {TEXT_PRIMARY}; "
        f"font-weight: bold; font-size: 13px; }}"
    )
