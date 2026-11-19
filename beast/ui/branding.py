"""Branding constants shared by the CLI and the terminal UI.

Keeping the ASCII art and the palette in one module means the launch screen,
the header and the status bar always agree.
"""

from __future__ import annotations

APP_NAME = "BEAST"
APP_TITLE = "BEAST AI CODING AGENT"
TAGLINE = "Local AI coding agent — everything runs on your machine."

#: Compact 3-line wordmark, 19 columns wide, used in the persistent header.
LOGO_MARK = (
    "█▀▄ █▀▀ ▄▀▄ █▀▀ ▀█▀\n"
    "█▀▄ █▀▀ █▀█ ▀▀█  █ \n"
    "▀▀  ▀▀▀ ▀ ▀ ▀▀▀  ▀ "
)

#: Full splash logo, 44 columns wide, shown in the welcome panel.
SPLASH_LOGO = (
    " ██████╗ ███████╗ █████╗ ███████╗████████╗\n"
    " ██╔══██╗██╔════╝██╔══██╗██╔════╝╚══██╔══╝\n"
    " ██████╔╝█████╗  ███████║███████╗   ██║   \n"
    " ██╔══██╗██╔══╝  ██╔══██║╚════██║   ██║   \n"
    " ██████╔╝███████╗██║  ██║███████║   ██║   \n"
    " ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   "
)

SPLASH_WIDTH = 46

# Palette — a restrained dark terminal palette with one warm accent.
BG = "#0b0e14"
SURFACE = "#11151d"
PANEL = "#161b25"
BORDER = "#262d3b"
BORDER_FOCUS = "#f0a44a"
TEXT = "#d7dbe3"
MUTED = "#7d8799"
ACCENT = "#f0a44a"
USER = "#6fb3e0"
SUCCESS = "#5fbf7f"
ERROR = "#e06c75"
WARNING = "#e5c07b"

PROMPT_PLACEHOLDER = "Describe what you want to build..."
