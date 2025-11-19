import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from anonator.ui.theme import THEME


class FrameComparisonViewer:
    def __init__(self, parent, max_width=480, max_height=270):
        self.parent = parent
        self.max_width = max_width
        self.max_height = max_height

        self.container = ctk.CTkFrame(parent, fg_color="transparent")

        grid_container = ctk.CTkFrame(self.container, fg_color="transparent")
        grid_container.pack(fill=tk.BOTH, expand=True)

        self.frame_original = ctk.CTkFrame(
            grid_container,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_secondary,
            border_width=THEME.spacing.border_width,
            border_color=THEME.colors.border_primary
        )
        self.frame_original.grid(row=0, column=0, sticky='nsew', padx=(0, THEME.spacing.pad_sm), pady=(0, 0))

        grid_container.columnconfigure(0, weight=1)
        grid_container.columnconfigure(1, weight=1)
        grid_container.rowconfigure(0, weight=1)

        label_container_orig = ctk.CTkFrame(self.frame_original, fg_color="transparent")
        label_container_orig.pack(fill=tk.X, padx=THEME.spacing.pad_base, pady=(THEME.spacing.pad_base, THEME.spacing.pad_sm))

        self.label_original = ctk.CTkLabel(
            label_container_orig,
            text="Original",
            font=(THEME.typography.font_family, THEME.typography.size_sm, "bold"),
            text_color=THEME.colors.text_primary,
            anchor='w'
        )
        self.label_original.pack(anchor='w')

        self.canvas_original = tk.Canvas(
            self.frame_original,
            width=max_width,
            height=max_height,
            bg=THEME.colors.bg_tertiary,
            highlightthickness=0
        )
        self.canvas_original.pack(padx=THEME.spacing.pad_base, pady=(0, THEME.spacing.pad_base))

        self.frame_processed = ctk.CTkFrame(
            grid_container,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_secondary,
            border_width=THEME.spacing.border_width,
            border_color=THEME.colors.border_primary
        )
        self.frame_processed.grid(row=0, column=1, sticky='nsew', padx=(THEME.spacing.pad_sm, 0), pady=(0, 0))

        label_container_proc = ctk.CTkFrame(self.frame_processed, fg_color="transparent")
        label_container_proc.pack(fill=tk.X, padx=THEME.spacing.pad_base, pady=(THEME.spacing.pad_base, THEME.spacing.pad_sm))

        self.label_processed = ctk.CTkLabel(
            label_container_proc,
            text="Anonymized",
            font=(THEME.typography.font_family, THEME.typography.size_sm, "bold"),
            text_color=THEME.colors.text_primary,
            anchor='w'
        )
        self.label_processed.pack(anchor='w')

        self.canvas_processed = tk.Canvas(
            self.frame_processed,
            width=max_width,
            height=max_height,
            bg=THEME.colors.bg_tertiary,
            highlightthickness=0
        )
        self.canvas_processed.pack(padx=THEME.spacing.pad_base, pady=(0, THEME.spacing.pad_base))

        self._original_image = None
        self._processed_image = None

    def update_frames(self, original_frame: np.ndarray, processed_frame: np.ndarray):
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        original_scaled = self._scale_image(original_rgb)
        processed_scaled = self._scale_image(processed_rgb)

        self._original_image = ImageTk.PhotoImage(Image.fromarray(original_scaled))
        self._processed_image = ImageTk.PhotoImage(Image.fromarray(processed_scaled))

        self.canvas_original.delete("all")
        self.canvas_processed.delete("all")

        self.canvas_original.create_image(
            self.max_width // 2,
            self.max_height // 2,
            image=self._original_image
        )
        self.canvas_processed.create_image(
            self.max_width // 2,
            self.max_height // 2,
            image=self._processed_image
        )

    def _scale_image(self, img_array: np.ndarray) -> np.ndarray:
        h, w = img_array.shape[:2]
        scale = min(self.max_width / w, self.max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if scale < 1:
            return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
