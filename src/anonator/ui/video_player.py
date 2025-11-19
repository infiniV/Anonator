"""
Video player widget for playing processed videos.
"""

import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from .widgets import Card, Button, SectionLabel
from anonator.ui.theme import THEME


class VideoPlayer:
    def __init__(self, parent):
        self.parent = parent
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playback_thread = None
        self._stop_flag = threading.Event()

        self.container = Card(parent)

        self._create_ui()

    def _create_ui(self):
        content = ctk.CTkFrame(self.container, fg_color="transparent")
        content.pack(fill=tk.BOTH, expand=True, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        header = ctk.CTkFrame(content, fg_color="transparent")
        header.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        SectionLabel(header, text="Processed Video").pack(side=tk.LEFT, anchor='w')

        self.status_label = ctk.CTkLabel(
            header,
            text="No video loaded",
            font=(THEME.typography.font_family, THEME.typography.size_sm),
            text_color=THEME.colors.text_secondary
        )
        self.status_label.pack(side=tk.RIGHT)

        canvas_container = ctk.CTkFrame(
            content,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_tertiary,
            border_width=THEME.spacing.border_width,
            border_color=THEME.colors.border_secondary
        )
        canvas_container.pack(fill=tk.BOTH, expand=True, pady=(0, THEME.spacing.pad_base))

        self.canvas = tk.Canvas(
            canvas_container,
            bg=THEME.colors.bg_tertiary,
            highlightthickness=0,
            height=180
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=THEME.spacing.pad_base, pady=THEME.spacing.pad_base)

        self.placeholder_text = self.canvas.create_text(
            0, 0,
            text="Processed video will appear here after completion",
            fill=THEME.colors.text_tertiary,
            font=(THEME.typography.font_family, THEME.typography.size_sm),
            anchor='center'
        )
        self.canvas.bind('<Configure>', self._center_placeholder)

        controls_frame = ctk.CTkFrame(content, fg_color="transparent")
        controls_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=100,
            variable=self.progress_var,
            command=self._on_seek,
            fg_color=THEME.colors.bg_tertiary,
            progress_color=THEME.colors.accent_primary,
            button_color=THEME.colors.accent_primary,
            button_hover_color=THEME.colors.accent_hover
        )
        self.progress_slider.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        button_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        button_frame.pack()

        self.play_button = Button(
            button_frame,
            text="Play",
            variant="primary",
            command=self._toggle_play,
            state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=(0, THEME.spacing.pad_base))

        self.stop_button = Button(
            button_frame,
            text="Stop",
            variant="secondary",
            command=self._stop_video,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, THEME.spacing.pad_lg))

        self.time_label = ctk.CTkLabel(
            button_frame,
            text="00:00 / 00:00",
            font=(THEME.typography.font_family_mono, THEME.typography.size_sm),
            text_color=THEME.colors.text_secondary
        )
        self.time_label.pack(side=tk.LEFT)

        self._photo = None

    def _center_placeholder(self, event=None):
        self.canvas.coords(self.placeholder_text, event.width // 2, event.height // 2)

    def load_video(self, video_path: str):
        if self.cap:
            self.cap.release()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            self.status_label.configure(text="Error: Could not open video")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0

        duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.status_label.configure(text=f"{self.total_frames} frames, {duration:.1f}s")

        self.play_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress_slider.configure(to=self.total_frames - 1)

        self.canvas.delete(self.placeholder_text)

        self._show_frame(0)

    def _toggle_play(self):
        if self.is_playing:
            self._pause_video()
        else:
            self._play_video()

    def _play_video(self):
        if not self.cap or self.is_playing:
            return

        self.is_playing = True
        self.play_button.configure(text="Pause")
        self._stop_flag.clear()

        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _pause_video(self):
        self.is_playing = False
        self.play_button.configure(text="Play")
        self._stop_flag.set()

    def _stop_video(self):
        self._pause_video()
        self.current_frame = 0
        self._show_frame(0)
        self.progress_var.set(0)

    def _playback_loop(self):
        frame_delay = 1.0 / self.fps if self.fps > 0 else 0.033

        while self.is_playing and not self._stop_flag.is_set():
            start_time = time.time()

            if self.current_frame >= self.total_frames - 1:
                self.parent.after(0, self._stop_video)
                break

            self.parent.after(0, lambda f=self.current_frame: self._show_frame(f))
            self.current_frame += 1

            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)

    def _show_frame(self, frame_number: int):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 500
            canvas_height = 180

        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if scale < 1:
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self._photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self._photo,
            anchor='center'
        )

        self.progress_var.set(frame_number)

        current_time = frame_number / self.fps if self.fps > 0 else 0
        total_time = self.total_frames / self.fps if self.fps > 0 else 0

        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)

        self.time_label.configure(text=f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")

    def _on_seek(self, value):
        if not self.cap or self.is_playing:
            return

        frame_number = int(float(value))
        self.current_frame = frame_number
        self._show_frame(frame_number)

    def cleanup(self):
        self._pause_video()
        if self.cap:
            self.cap.release()
