import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


class FrameComparisonViewer:
    def __init__(self, parent, max_width=400, max_height=300):
        self.parent = parent
        self.max_width = max_width
        self.max_height = max_height

        self.container = tk.Frame(parent, bg='#000000')
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_original = tk.Frame(self.container, bg='#000000')
        self.frame_original.pack(side=tk.LEFT, padx=5)

        self.label_original = tk.Label(
            self.frame_original,
            text="Original",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        )
        self.label_original.pack()

        self.canvas_original = tk.Canvas(
            self.frame_original,
            width=max_width,
            height=max_height,
            bg='#000000',
            highlightthickness=0
        )
        self.canvas_original.pack()

        self.frame_processed = tk.Frame(self.container, bg='#000000')
        self.frame_processed.pack(side=tk.LEFT, padx=5)

        self.label_processed = tk.Label(
            self.frame_processed,
            text="Anonymized",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        )
        self.label_processed.pack()

        self.canvas_processed = tk.Canvas(
            self.frame_processed,
            width=max_width,
            height=max_height,
            bg='#000000',
            highlightthickness=0
        )
        self.canvas_processed.pack()

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
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
