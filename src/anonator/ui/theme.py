"""
Theme configuration for Anonator UI.

Implements Claude Code inspired dark theme with clean, professional styling.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ColorPalette:
    """Color palette for the application theme."""

    # Background colors
    bg_primary: str = "#0d0d0d"
    bg_secondary: str = "#1a1a1a"
    bg_tertiary: str = "#252525"
    bg_hover: str = "#2a2a2a"

    # Border colors
    border_primary: str = "#3a3a3a"
    border_secondary: str = "#2a2a2a"
    border_focus: str = "#5b9dd9"

    # Text colors
    text_primary: str = "#e8e8e8"
    text_secondary: str = "#999999"
    text_tertiary: str = "#666666"

    # Accent colors
    accent_primary: str = "#5b9dd9"
    accent_hover: str = "#6daee6"

    # Status colors
    success: str = "#4caf50"
    warning: str = "#ff9800"
    error: str = "#f44336"
    hipaa: str = "#ef5350"

    # Button colors
    button_primary_bg: str = "#5b9dd9"
    button_primary_hover: str = "#6daee6"
    button_primary_text: str = "#ffffff"

    button_secondary_bg: str = "#2a2a2a"
    button_secondary_hover: str = "#3a3a3a"
    button_secondary_text: str = "#e8e8e8"


@dataclass
class Typography:
    """Typography settings for the application."""

    font_family: str = "Segoe UI"
    font_family_mono: str = "Consolas"

    # Font sizes
    size_xs: int = 9
    size_sm: int = 10
    size_base: int = 11
    size_lg: int = 13
    size_xl: int = 15
    size_2xl: int = 18
    size_3xl: int = 24

    # Font weights
    weight_normal: str = "normal"
    weight_medium: str = "normal"
    weight_bold: str = "bold"


@dataclass
class Spacing:
    """Spacing and sizing constants."""

    # Padding
    pad_xs: int = 4
    pad_sm: int = 8
    pad_base: int = 12
    pad_lg: int = 16
    pad_xl: int = 24
    pad_2xl: int = 32

    # Border radius
    radius_sm: int = 4
    radius_base: int = 6
    radius_lg: int = 8

    # Border width
    border_width: int = 1
    border_width_thick: int = 2


class Theme:
    """Main theme class combining all styling elements."""

    def __init__(self):
        self.colors = ColorPalette()
        self.typography = Typography()
        self.spacing = Spacing()

    def get_ttk_style_config(self) -> Dict:
        """Get ttk style configuration dictionary."""
        return {
            'TFrame': {
                'configure': {
                    'background': self.colors.bg_primary
                }
            },
            'Card.TFrame': {
                'configure': {
                    'background': self.colors.bg_secondary,
                    'borderwidth': self.spacing.border_width,
                    'relief': 'flat'
                }
            },
            'TLabel': {
                'configure': {
                    'background': self.colors.bg_primary,
                    'foreground': self.colors.text_primary,
                    'font': (self.typography.font_family, self.typography.size_base)
                }
            },
            'Secondary.TLabel': {
                'configure': {
                    'background': self.colors.bg_primary,
                    'foreground': self.colors.text_secondary,
                    'font': (self.typography.font_family, self.typography.size_sm)
                }
            },
            'Heading.TLabel': {
                'configure': {
                    'background': self.colors.bg_primary,
                    'foreground': self.colors.text_primary,
                    'font': (self.typography.font_family, self.typography.size_2xl, self.typography.weight_bold)
                }
            },
            'TCombobox': {
                'configure': {
                    'fieldbackground': self.colors.bg_tertiary,
                    'background': self.colors.bg_tertiary,
                    'foreground': self.colors.text_primary,
                    'borderwidth': self.spacing.border_width,
                    'relief': 'flat'
                }
            },
            'TEntry': {
                'configure': {
                    'fieldbackground': self.colors.bg_tertiary,
                    'foreground': self.colors.text_primary,
                    'borderwidth': self.spacing.border_width,
                    'relief': 'flat'
                }
            },
            'Horizontal.TProgressbar': {
                'configure': {
                    'background': self.colors.accent_primary,
                    'troughcolor': self.colors.bg_tertiary,
                    'borderwidth': 0,
                    'thickness': 8
                }
            }
        }


# Global theme instance
THEME = Theme()
