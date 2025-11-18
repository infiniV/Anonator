"""
Modern styled widgets for Anonator UI.

Provides reusable UI components with consistent styling.
"""

import tkinter as tk
from tkinter import ttk
from .theme import THEME


class Card(tk.Frame):
    """A card-style container with rounded appearance and border."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=THEME.colors.bg_secondary,
            highlightbackground=THEME.colors.border_primary,
            highlightthickness=THEME.spacing.border_width,
            **kwargs
        )


class Button(tk.Button):
    """Styled button widget."""

    def __init__(self, parent, text="", variant="primary", **kwargs):
        if variant == "primary":
            bg = THEME.colors.button_primary_bg
            fg = THEME.colors.button_primary_text
            active_bg = THEME.colors.button_primary_hover
        else:
            bg = THEME.colors.button_secondary_bg
            fg = THEME.colors.button_secondary_text
            active_bg = THEME.colors.button_secondary_hover

        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            font=(THEME.typography.font_family, THEME.typography.size_base, THEME.typography.weight_medium),
            bd=0,
            highlightthickness=0,
            relief='flat',
            cursor='hand2',
            padx=THEME.spacing.pad_lg,
            pady=THEME.spacing.pad_base,
            **kwargs
        )

        self.bind('<Enter>', lambda e: self.config(bg=active_bg))
        self.bind('<Leave>', lambda e: self.config(bg=bg))


class Label(tk.Label):
    """Styled label widget."""

    def __init__(self, parent, text="", variant="primary", **kwargs):
        if variant == "primary":
            fg = THEME.colors.text_primary
            font_size = THEME.typography.size_base
        elif variant == "secondary":
            fg = THEME.colors.text_secondary
            font_size = THEME.typography.size_sm
        elif variant == "heading":
            fg = THEME.colors.text_primary
            font_size = THEME.typography.size_2xl
        else:
            fg = THEME.colors.text_primary
            font_size = THEME.typography.size_base

        super().__init__(
            parent,
            text=text,
            fg=fg,
            bg=THEME.colors.bg_primary,
            font=(THEME.typography.font_family, font_size),
            **kwargs
        )


class SectionLabel(tk.Label):
    """Label for section headers."""

    def __init__(self, parent, text="", **kwargs):
        super().__init__(
            parent,
            text=text,
            fg=THEME.colors.text_primary,
            bg=THEME.colors.bg_secondary,
            font=(THEME.typography.font_family, THEME.typography.size_lg, THEME.typography.weight_bold),
            anchor='w',
            **kwargs
        )


class FieldLabel(tk.Label):
    """Label for form fields."""

    def __init__(self, parent, text="", **kwargs):
        super().__init__(
            parent,
            text=text,
            fg=THEME.colors.text_secondary,
            bg=THEME.colors.bg_secondary,
            font=(THEME.typography.font_family, THEME.typography.size_sm),
            anchor='w',
            **kwargs
        )


class Entry(tk.Entry):
    """Styled entry widget."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=THEME.colors.bg_tertiary,
            fg=THEME.colors.text_primary,
            insertbackground=THEME.colors.text_primary,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            bd=0,
            highlightthickness=THEME.spacing.border_width,
            highlightbackground=THEME.colors.border_primary,
            highlightcolor=THEME.colors.border_focus,
            relief='flat',
            **kwargs
        )


class DropZone(tk.Frame):
    """Styled drop zone for file uploads."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=THEME.colors.bg_secondary,
            highlightbackground=THEME.colors.border_primary,
            highlightthickness=THEME.spacing.border_width_thick,
            **kwargs
        )

        self.label = tk.Label(
            self,
            text="Drop video file here or click to browse",
            fg=THEME.colors.text_secondary,
            bg=THEME.colors.bg_secondary,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            cursor='hand2'
        )
        self.label.pack(expand=True, fill=tk.BOTH)

        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.label.bind('<Enter>', self._on_enter)
        self.label.bind('<Leave>', self._on_leave)

    def _on_enter(self, event):
        self.config(
            bg=THEME.colors.bg_hover,
            highlightbackground=THEME.colors.accent_primary
        )
        self.label.config(
            bg=THEME.colors.bg_hover,
            fg=THEME.colors.accent_primary
        )

    def _on_leave(self, event):
        self.config(
            bg=THEME.colors.bg_secondary,
            highlightbackground=THEME.colors.border_primary
        )
        self.label.config(
            bg=THEME.colors.bg_secondary,
            fg=THEME.colors.text_secondary
        )

    def set_text(self, text):
        self.label.config(text=text)


class Separator(tk.Frame):
    """Horizontal separator line."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=THEME.colors.border_secondary,
            height=1,
            **kwargs
        )
