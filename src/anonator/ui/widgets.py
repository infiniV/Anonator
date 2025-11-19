"""
Modern styled widgets for Anonator UI using CustomTkinter.

Provides reusable UI components with consistent styling based on the global theme.
"""

import customtkinter as ctk
from anonator.ui.theme import THEME


class Card(ctk.CTkFrame):
    """A card-style container with rounded appearance."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_secondary,
            border_width=THEME.spacing.border_width,
            border_color=THEME.colors.border_primary,
            **kwargs
        )


class Button(ctk.CTkButton):
    """Styled button widget with refined appearance."""

    def __init__(self, parent, text="", variant="primary", **kwargs):
        if variant == "primary":
            fg_color = THEME.colors.button_primary_bg
            hover_color = THEME.colors.button_primary_hover
            text_color = THEME.colors.button_primary_text
            text_color_disabled = THEME.colors.text_primary
        else:
            fg_color = THEME.colors.button_secondary_bg
            hover_color = THEME.colors.button_secondary_hover
            text_color = THEME.colors.button_secondary_text
            text_color_disabled = THEME.colors.text_secondary

        super().__init__(
            parent,
            text=text,
            corner_radius=THEME.spacing.radius_base,
            font=(THEME.typography.font_family, THEME.typography.size_sm, THEME.typography.weight_bold),
            height=40,
            fg_color=fg_color,
            hover_color=hover_color,
            text_color=text_color,
            text_color_disabled=text_color_disabled,
            border_width=0,
            **kwargs
        )


class Label(ctk.CTkLabel):
    """Styled label widget."""

    def __init__(self, parent, text="", variant="primary", bg=None, **kwargs):
        # Ignore bg parameter (kept for compatibility)
        if variant == "heading":
            font = (THEME.typography.font_family, THEME.typography.size_2xl, THEME.typography.weight_bold)
            text_color = THEME.colors.text_primary
        elif variant == "secondary":
            font = (THEME.typography.font_family, THEME.typography.size_sm)
            text_color = THEME.colors.text_secondary
        else:
            font = (THEME.typography.font_family, THEME.typography.size_base)
            text_color = THEME.colors.text_primary

        super().__init__(
            parent,
            text=text,
            font=font,
            text_color=text_color,
            **kwargs
        )


class SectionLabel(ctk.CTkLabel):
    """Label for section headers with refined typography."""

    def __init__(self, parent, text="", bg=None, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=(THEME.typography.font_family, THEME.typography.size_lg, THEME.typography.weight_bold),
            text_color=THEME.colors.text_primary,
            anchor='w',
            **kwargs
        )


class FieldLabel(ctk.CTkLabel):
    """Label for form fields."""

    def __init__(self, parent, text="", bg=None, **kwargs):
        super().__init__(
            parent,
            text=text,
            font=(THEME.typography.font_family, THEME.typography.size_sm, THEME.typography.weight_medium),
            text_color=THEME.colors.text_secondary,
            anchor='w',
            **kwargs
        )


class Entry(ctk.CTkEntry):
    """Refined entry widget with better visual hierarchy."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=THEME.spacing.radius_base,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            height=36,
            fg_color=THEME.colors.bg_tertiary,
            border_color=THEME.colors.border_secondary,
            text_color=THEME.colors.text_primary,
            placeholder_text_color=THEME.colors.text_tertiary,
            **kwargs
        )


class DropZone(ctk.CTkFrame):
    """Refined drop zone for file uploads with polished appearance."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_tertiary,
            border_width=2,
            border_color=THEME.colors.border_secondary,
            **kwargs
        )

        # Main instruction text
        self.label = ctk.CTkLabel(
            self,
            text="Drop video file here or click to browse",
            font=(THEME.typography.font_family, THEME.typography.size_base, THEME.typography.weight_medium),
            text_color=THEME.colors.accent_primary,
            cursor='hand2'
        )
        self.label.pack(expand=True, pady=(20, 4))

        # Hint text
        self.hint = ctk.CTkLabel(
            self,
            text="Supported formats: MP4, AVI, MOV",
            font=(THEME.typography.font_family, THEME.typography.size_xs),
            text_color=THEME.colors.text_tertiary,
            cursor='hand2'
        )
        self.hint.pack(pady=(0, 20))

    def set_text(self, text):
        # Update text when file is selected
        if "Selected:" in text:
            self.label.configure(text=text, font=(THEME.typography.font_family, THEME.typography.size_base, "bold"))
            self.hint.configure(text="")
            self.configure(border_color=THEME.colors.accent_primary)
        else:
            self.label.configure(text=text, font=(THEME.typography.font_family, THEME.typography.size_base, THEME.typography.weight_medium))
            self.hint.configure(text="Supported formats: MP4, AVI, MOV")
            self.configure(border_color=THEME.colors.border_secondary)


class Separator(ctk.CTkFrame):
    """Horizontal separator line."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=1,
            fg_color=THEME.colors.border_secondary,
            **kwargs
        )


class LogBox(ctk.CTkTextbox):
    """Read-only text box for logs."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            corner_radius=THEME.spacing.radius_base,
            fg_color=THEME.colors.bg_tertiary,
            text_color=THEME.colors.text_secondary,
            font=(THEME.typography.font_family_mono, THEME.typography.size_xs),
            activate_scrollbars=True,
            **kwargs
        )
        self.configure(state="disabled")

    def log(self, message):
        self.configure(state="normal")
        self.insert("end", f"{message}\n")
        self.see("end")
        self.configure(state="disabled")


class CustomAlertDialog(ctk.CTkToplevel):
    """Custom styled alert dialog."""

    def __init__(self, parent, title, message, on_ok=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x200")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.configure(fg_color=THEME.colors.bg_secondary)

        # Center content
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # Icon/Title
        title_label = ctk.CTkLabel(
            content, 
            text=title, 
            font=(THEME.typography.font_family, THEME.typography.size_xl, "bold"),
            text_color=THEME.colors.text_primary
        )
        title_label.pack(pady=(0, 10))

        # Message
        msg_label = ctk.CTkLabel(
            content, 
            text=message, 
            font=(THEME.typography.font_family, THEME.typography.size_base),
            text_color=THEME.colors.text_secondary,
            wraplength=360
        )
        msg_label.pack(pady=(0, 20))

        # Button
        ok_btn = Button(
            content, 
            text="OK", 
            variant="primary", 
            command=self._on_ok,
            width=100
        )
        ok_btn.pack()

        self.on_ok = on_ok

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _on_ok(self):
        if self.on_ok:
            self.on_ok()
        self.destroy()
