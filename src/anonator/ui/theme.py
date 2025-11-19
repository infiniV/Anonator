"""
Theme configuration for Anonator UI.

Implements professional brown dark theme with clean, flat design.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ColorPalette:
    """Color palette for the application theme."""

    # Background colors - Professional brown dark theme
    bg_primary: str = "#1A0F0A"
    bg_secondary: str = "#2B1810"
    bg_tertiary: str = "#3D2418"
    bg_hover: str = "#4A2D1F"

    # Border colors - Subtle brown borders for clean look
    border_primary: str = "#4A2D1F"
    border_secondary: str = "#3D2418"
    border_focus: str = "#D4A574"

    # Text colors - Cream tones for high contrast readability
    text_primary: str = "#FFF8DC"
    text_secondary: str = "#D4C5B0"
    text_tertiary: str = "#A89680"

    # Accent colors - Golden brown accents
    accent_primary: str = "#D4A574"
    accent_hover: str = "#E5B885"
    accent_light: str = "#F0C896"

    # Status colors - Muted flat colors
    success: str = "#8FAF6F"
    warning: str = "#D4A574"
    error: str = "#C17B5F"
    hipaa: str = "#C17B5F"

    # Button colors
    button_primary_bg: str = "#D4A574"
    button_primary_hover: str = "#E5B885"
    button_primary_text: str = "#1A0F0A"

    button_secondary_bg: str = "#3D2418"
    button_secondary_hover: str = "#4A2D1F"
    button_secondary_text: str = "#FFF8DC"


@dataclass
class Typography:
    """Typography settings for the application."""

    font_family: str = "Segoe UI"
    font_family_mono: str = "Consolas"

    # Font sizes - Refined scale with better hierarchy
    size_xs: int = 10
    size_sm: int = 11
    size_base: int = 13
    size_lg: int = 15
    size_xl: int = 18
    size_2xl: int = 24
    size_3xl: int = 32

    # Font weights
    weight_normal: str = "normal"
    weight_medium: str = "normal"
    weight_bold: str = "bold"


@dataclass
class Spacing:
    """Spacing and sizing constants."""

    # Padding - Generous spacing for refined look
    pad_xs: int = 8
    pad_sm: int = 12
    pad_base: int = 16
    pad_lg: int = 20
    pad_xl: int = 32
    pad_2xl: int = 48

    # Border radius - Smoother corners for polished appearance
    radius_sm: int = 8
    radius_base: int = 12
    radius_lg: int = 16

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
                    'thickness': 6
                }
            }
        }


# Global theme instance
THEME = Theme()
