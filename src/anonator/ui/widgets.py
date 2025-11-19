"""
Modern styled widgets for Anonator UI using CustomTkinter.

Provides reusable UI components with consistent styling.
"""

import customtkinter as ctk


class Card(ctk.CTkFrame):
    """A card-style container with rounded appearance."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=12,
            **kwargs
        )


class Button(ctk.CTkButton):
    """Styled button widget with refined appearance."""

    def __init__(self, parent, text="", variant="primary", **kwargs):
        # CustomTkinter buttons auto-style based on theme
        super().__init__(
            parent,
            text=text,
            corner_radius=10,
            font=("Segoe UI", 12, "bold"),
            height=36,
            **kwargs
        )


class Label(ctk.CTkLabel):
    """Styled label widget."""

    def __init__(self, parent, text="", variant="primary", bg=None, **kwargs):
        # Ignore bg parameter (kept for compatibility)
        if variant == "heading":
            font = ("Segoe UI", 24, "bold")
        elif variant == "secondary":
            font = ("Segoe UI", 10)
        else:
            font = ("Segoe UI", 12)

        super().__init__(
            parent,
            text=text,
            font=font,
            **kwargs
        )


class SectionLabel(ctk.CTkLabel):
    """Label for section headers with refined typography."""

    def __init__(self, parent, text="", bg=None, **kwargs):
        # Ignore bg parameter (kept for compatibility)
        super().__init__(
            parent,
            text=text,
            font=("Segoe UI", 14, "bold"),
            anchor='w',
            **kwargs
        )


class FieldLabel(ctk.CTkLabel):
    """Label for form fields."""

    def __init__(self, parent, text="", bg=None, **kwargs):
        # Ignore bg parameter (kept for compatibility)
        super().__init__(
            parent,
            text=text,
            font=("Segoe UI", 10),
            anchor='w',
            **kwargs
        )


class Entry(ctk.CTkEntry):
    """Refined entry widget with better visual hierarchy."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=8,
            font=("Segoe UI", 11),
            height=30,
            **kwargs
        )


class DropZone(ctk.CTkFrame):
    """Refined drop zone for file uploads with polished appearance."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=12,
            **kwargs
        )

        # Main instruction text
        self.label = ctk.CTkLabel(
            self,
            text="Drop video file here or click to browse",
            font=("Segoe UI", 11),
            cursor='hand2'
        )
        self.label.pack(expand=True, pady=(12, 4))

        # Hint text
        self.hint = ctk.CTkLabel(
            self,
            text="Supported formats: MP4, AVI, MOV",
            font=("Segoe UI", 9),
            cursor='hand2'
        )
        self.hint.pack(pady=(0, 12))

    def set_text(self, text):
        # Update text when file is selected
        if "Selected:" in text:
            self.label.configure(text=text, font=("Segoe UI", 11, "bold"))
            self.hint.configure(text="")
        else:
            self.label.configure(text=text, font=("Segoe UI", 11))
            self.hint.configure(text="Supported formats: MP4, AVI, MOV")


class Separator(ctk.CTkFrame):
    """Horizontal separator line."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=1,
            **kwargs
        )
