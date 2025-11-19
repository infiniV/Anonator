"""
Theme configuration for Anonator UI.

Implements a premium dark theme with warm orange accents based on the provided palette.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ColorPalette:
    """Color palette for the application theme."""

    # Background colors
    bg_primary: str = "#2a2024"    # --background
    bg_secondary: str = "#392f35"  # --card / --popover / --sidebar
    bg_tertiary: str = "#30272c"   # --muted
    bg_hover: str = "#463a41"      # --input / --border (used for hover states)

    # Border colors
    border_primary: str = "#463a41"   # --border
    border_secondary: str = "#392f35" # --card
    border_focus: str = "#ff7e5f"     # --ring

    # Text colors
    text_primary: str = "#f2e9e4"     # --foreground
    text_secondary: str = "#d7c6bc"   # --muted-foreground
    text_tertiary: str = "#ce6a57"    # --chart-5

    # Accent colors - Orange Primary
    accent_primary: str = "#ff7e5f"   # --primary
    accent_hover: str = "#e66a4f"     # Darker shade of primary
    accent_light: str = "#feb47b"     # --accent

    # Status colors
    success: str = "#22c55e"          # Standard Green
    warning: str = "#ffcaa7"          # --chart-3
    error: str = "#e63946"            # --destructive
    hipaa: str = "#e63946"            # --destructive

    # Button colors
    button_primary_bg: str = "#ff7e5f"       # --primary
    button_primary_hover: str = "#e66a4f"    # Darker primary
    button_primary_text: str = "#ffffff"     # --primary-foreground

    # Made secondary button lighter for better visibility
    button_secondary_bg: str = "#463a41"     # --secondary
    button_secondary_hover: str = "#554a50"  # Lighter secondary
    button_secondary_text: str = "#f2e9e4"   # --secondary-foreground


@dataclass
class Typography:
    """Typography settings for the application."""

    font_family: str = "Segoe UI"
    font_family_mono: str = "Consolas"

    # Font sizes - Modern scale
    size_xs: int = 11
    size_sm: int = 12
    size_base: int = 14
    size_lg: int = 16
    size_xl: int = 20
    size_2xl: int = 28
    size_3xl: int = 36

    # Font weights
    weight_normal: str = "normal"
    weight_medium: str = "bold"  # Tkinter font weight
    weight_bold: str = "bold"


@dataclass
class Spacing:
    """Spacing and sizing constants."""

    # Padding
    pad_xs: int = 4
    pad_sm: int = 8
    pad_base: int = 12  # Reduced from 16
    pad_lg: int = 20    # Reduced from 24
    pad_xl: int = 28    # Reduced from 32
    pad_2xl: int = 40

    # Border radius
    radius_sm: int = 6
    radius_base: int = 8
    radius_lg: int = 12

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
