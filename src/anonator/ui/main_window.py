import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import queue
from pathlib import Path

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

from anonator.core.processor import VideoProcessor, ProgressData
from anonator.ui.frame_viewer import FrameComparisonViewer
from anonator.ui.video_player import VideoPlayer
from anonator.core.config import HIPAA_MODE, STANDARD_MODE, get_mode_config
from anonator.ui.widgets import Card, Button, Label, SectionLabel, FieldLabel, Entry, DropZone, Separator


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Anonator - Video Face Anonymization")
        self.root.geometry("1400x900")

        self.processor = VideoProcessor(progress_callback=self._on_progress)
        self.progress_queue = queue.Queue()

        self.input_path = None
        self.output_path = None
        self.is_processing = False

        self._setup_ui()
        self._setup_drag_drop()
        self._poll_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        container = ctk.CTkFrame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        left_pane = self._create_left_pane(container)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, padx=(20, 12), pady=20)

        separator = ctk.CTkFrame(container, width=1)
        separator.pack(side=tk.LEFT, fill=tk.Y, pady=20)

        right_pane = self._create_right_pane(container)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 20), pady=20)

    def _create_left_pane(self, parent):
        """Create left sidebar with controls."""
        pane = ctk.CTkFrame(parent, width=380)
        pane.pack_propagate(False)

        # Create scrollable frame
        scrollable = ctk.CTkScrollableFrame(pane, fg_color="transparent")
        scrollable.pack(fill=tk.BOTH, expand=True)

        # Header section
        header = ctk.CTkFrame(scrollable, fg_color="transparent")
        header.pack(fill=tk.X, pady=(0, 16), padx=20)

        title = Label(header, text="ANONATOR", variant="heading")
        title.pack(anchor='w')

        subtitle = Label(header, text="Video Face Anonymization", variant="secondary")
        subtitle.pack(anchor='w', pady=(4, 0))

        # Card spacing
        card_spacing = 12

        file_card = Card(scrollable)
        file_card.pack(fill=tk.X, pady=(0, card_spacing), padx=20)
        self._create_file_section(file_card)

        settings_card = Card(scrollable)
        settings_card.pack(fill=tk.X, pady=(0, card_spacing), padx=20)
        self._create_settings_section(settings_card)

        hipaa_card = Card(scrollable)
        hipaa_card.pack(fill=tk.X, pady=(0, card_spacing), padx=20)
        self._create_hipaa_section(hipaa_card)

        return pane

    def _create_file_section(self, parent):
        """Create file selection section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill=tk.BOTH, padx=16, pady=16)

        SectionLabel(section, text="Input File").pack(anchor='w', pady=(0, 8))

        self.drop_zone = DropZone(section, height=80)
        self.drop_zone.pack(fill=tk.X, pady=(0, 10))
        self.drop_zone.label.bind('<Button-1>', lambda e: self._browse_file())
        self.drop_zone.hint.bind('<Button-1>', lambda e: self._browse_file())
        self.drop_zone.pack_propagate(False)

        self.file_label = Label(section, text="No file selected", variant="secondary")
        self.file_label.pack(anchor='w')

    def _create_settings_section(self, parent):
        """Create settings section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill=tk.BOTH, padx=16, pady=16)

        SectionLabel(section, text="Processing Settings").pack(anchor='w', pady=(0, 8))

        # Grid layout for settings
        settings_grid = ctk.CTkFrame(section, fg_color="transparent")
        settings_grid.pack(fill=tk.X, pady=(0, 8))
        settings_grid.columnconfigure(0, weight=0)
        settings_grid.columnconfigure(1, weight=1)

        # Mode
        FieldLabel(settings_grid, text="Anonymization Mode").grid(row=0, column=0, sticky='w', pady=(0, 8), padx=(0, 12))

        self.mode_var = tk.StringVar(value=STANDARD_MODE.anonymization_mode)
        mode_options = ["blur", "solid", "mosaic"]
        self.mode_menu = ctk.CTkOptionMenu(
            settings_grid,
            variable=self.mode_var,
            values=mode_options,
            width=140,
            corner_radius=8
        )
        self.mode_menu.grid(row=0, column=1, sticky='ew', pady=(0, 8))

        # Threshold
        FieldLabel(settings_grid, text="Detection Threshold").grid(row=1, column=0, sticky='w', pady=(0, 8), padx=(0, 12))

        self.threshold_var = tk.DoubleVar(value=STANDARD_MODE.detection_threshold)
        self.threshold_entry = Entry(settings_grid, textvariable=self.threshold_var, width=140)
        self.threshold_entry.grid(row=1, column=1, sticky='ew', pady=(0, 8))

        # Mask Scale
        FieldLabel(settings_grid, text="Mask Scale").grid(row=2, column=0, sticky='w', padx=(0, 12))

        self.mask_scale_var = tk.DoubleVar(value=STANDARD_MODE.mask_scale)
        self.mask_scale_entry = Entry(settings_grid, textvariable=self.mask_scale_var, width=140)
        self.mask_scale_entry.grid(row=2, column=1, sticky='ew')

        Separator(section).pack(fill=tk.X, pady=10)

        # Checkboxes
        check_frame = ctk.CTkFrame(section, fg_color="transparent")
        check_frame.pack(fill=tk.X)

        self.multi_pass_var = tk.BooleanVar(value=STANDARD_MODE.multi_pass_enabled)
        self.multi_pass_check = ctk.CTkCheckBox(
            check_frame,
            text="Multi-Pass Detection",
            variable=self.multi_pass_var,
            font=("Segoe UI", 12)
        )
        self.multi_pass_check.pack(anchor='w', pady=(0, 6))

        self.keep_audio_var = tk.BooleanVar(value=STANDARD_MODE.keep_audio)
        self.keep_audio_check = ctk.CTkCheckBox(
            check_frame,
            text="Keep Audio Track",
            variable=self.keep_audio_var,
            font=("Segoe UI", 12)
        )
        self.keep_audio_check.pack(anchor='w')

    def _create_hipaa_section(self, parent):
        """Create HIPAA mode section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill=tk.BOTH, padx=16, pady=16)

        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(fill=tk.X, pady=(0, 6))

        hipaa_label = ctk.CTkLabel(
            header,
            text="HIPAA Mode",
            font=("Segoe UI", 16, "bold"),
            text_color="#C17B5F"
        )
        hipaa_label.pack(side=tk.LEFT)

        self.hipaa_mode_var = tk.BooleanVar(value=True)
        self.hipaa_mode_check = ctk.CTkSwitch(
            header,
            text="",
            variable=self.hipaa_mode_var,
            command=self._on_hipaa_mode_toggle
        )
        self.hipaa_mode_check.pack(side=tk.RIGHT)

        warning = ctk.CTkLabel(
            section,
            text="Ultra-sensitive detection for medical compliance",
            font=("Segoe UI", 10),
            anchor='w'
        )
        warning.pack(anchor='w')

        self._on_hipaa_mode_toggle()

    def _create_actions_section(self, parent):
        """Create action buttons section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill=tk.BOTH, padx=20, pady=20)

        self.start_button = Button(
            section,
            text="Start Processing",
            variant="primary",
            command=self._start_processing,
            state=tk.DISABLED
        )
        self.start_button.pack(fill=tk.X, pady=(0, 10))

        self.cancel_button = Button(
            section,
            text="Cancel",
            variant="secondary",
            command=self._cancel_processing,
            state=tk.DISABLED
        )
        self.cancel_button.pack(fill=tk.X)

    def _create_progress_section(self, parent):
        """Create progress section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill=tk.BOTH, padx=20, pady=20)

        SectionLabel(section, text="Progress").pack(anchor='w', pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ctk.CTkProgressBar(
            section,
            variable=self.progress_var,
            mode='determinate',
            height=8
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill=tk.X, pady=(0, 8))

        self.progress_label = Label(section, text="Ready", variant="secondary")
        self.progress_label.pack(anchor='w')

    def _create_right_pane(self, parent):
        """Create right pane with video preview."""
        pane = ctk.CTkFrame(parent, fg_color="transparent")

        title = Label(pane, text="Preview", variant="heading")
        title.pack(anchor='w', pady=(0, 16))

        # Grid container for 2x2 layout
        grid = ctk.CTkFrame(pane, fg_color="transparent")
        grid.pack(fill=tk.BOTH, expand=True)

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        # Row 1: Frame viewers
        frame_viewer_container = ctk.CTkFrame(grid, fg_color="transparent")
        frame_viewer_container.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=(0, 16))
        self.frame_viewer = FrameComparisonViewer(frame_viewer_container)
        self.frame_viewer.container.pack(fill=tk.BOTH, expand=True)

        # Row 2 Left: Actions and Progress
        actions_progress_frame = ctk.CTkFrame(grid, fg_color="transparent")
        actions_progress_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 8))

        actions_card = Card(actions_progress_frame)
        actions_card.pack(fill=tk.X, pady=(0, 12))
        self._create_actions_section(actions_card)

        progress_card = Card(actions_progress_frame)
        progress_card.pack(fill=tk.X)
        self._create_progress_section(progress_card)

        # Row 2 Right: Video Player
        video_frame = ctk.CTkFrame(grid, fg_color="transparent")
        video_frame.grid(row=1, column=1, sticky='nsew', padx=(8, 0))
        self.video_player = VideoPlayer(video_frame)
        self.video_player.container.pack(fill=tk.BOTH, expand=True)

        return pane

    def _setup_drag_drop(self):
        if DND_AVAILABLE:
            try:
                self.drop_zone.drop_target_register(DND_FILES)
                self.drop_zone.dnd_bind('<<Drop>>', self._on_drop)
            except Exception:
                # CustomTkinter doesn't support tkinterdnd2
                # Users can still use the browse button
                pass

    def _on_drop(self, event):
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        self._load_file(file_path)

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._load_file(file_path)

    def _on_hipaa_mode_toggle(self):
        if self.hipaa_mode_var.get():
            self.mode_var.set(HIPAA_MODE.anonymization_mode)
            self.threshold_var.set(HIPAA_MODE.detection_threshold)
            self.mask_scale_var.set(HIPAA_MODE.mask_scale)
            self.multi_pass_var.set(HIPAA_MODE.multi_pass_enabled)
            self.keep_audio_var.set(HIPAA_MODE.keep_audio)

            if HIPAA_MODE.lock_ui_controls:
                self.mode_menu.configure(state='disabled')
                self.threshold_entry.configure(state='disabled')
                self.mask_scale_entry.configure(state='disabled')
                self.multi_pass_check.configure(state='disabled')
                self.keep_audio_check.configure(state='disabled')
        else:
            self.mode_menu.configure(state='normal')
            self.threshold_entry.configure(state='normal')
            self.mask_scale_entry.configure(state='normal')
            self.multi_pass_check.configure(state='normal')
            self.keep_audio_check.configure(state='normal')

    def _load_file(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            messagebox.showerror("Error", "File not found")
            return

        if file_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            messagebox.showerror("Error", "Invalid file format")
            return

        self.input_path = str(file_path)
        self.output_path = str(file_path.parent / f"{file_path.stem}_anonymized{file_path.suffix}")

        self.file_label.configure(text=f"{file_path.name}")
        self.start_button.configure(state=tk.NORMAL)
        self.drop_zone.set_text(f"Selected: {file_path.name}")

    def _start_processing(self):
        if not self.input_path:
            return

        self.is_processing = True
        self.start_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.configure(text="Starting...")

        self.processor.process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            anonymization_mode=self.mode_var.get(),
            threshold=self.threshold_var.get(),
            mask_scale=self.mask_scale_var.get(),
            multi_pass=self.multi_pass_var.get(),
            keep_audio=self.keep_audio_var.get()
        )

    def _cancel_processing(self):
        self.processor.cancel()
        self.progress_label.configure(text="Cancelling...")
        self.cancel_button.configure(state=tk.DISABLED)

    def _on_progress(self, progress_data: ProgressData):
        self.progress_queue.put(('progress', progress_data))

    def _poll_queue(self):
        try:
            while True:
                msg_type, data = self.progress_queue.get_nowait()
                if msg_type == 'progress':
                    self._update_ui_progress(data)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_queue)

    def _update_ui_progress(self, progress_data: ProgressData):
        progress_percent = (progress_data.frame_number / progress_data.total_frames) * 100
        self.progress_var.set(progress_percent)

        elapsed_min = int(progress_data.elapsed_time // 60)
        elapsed_sec = int(progress_data.elapsed_time % 60)

        remaining_time = 0
        if progress_data.fps > 0:
            remaining_frames = progress_data.total_frames - progress_data.frame_number
            remaining_time = remaining_frames / progress_data.fps

        remaining_min = int(remaining_time // 60)
        remaining_sec = int(remaining_time % 60)

        status_text = (
            f"Frame {progress_data.frame_number}/{progress_data.total_frames} | "
            f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d} | "
            f"Remaining: {remaining_min:02d}:{remaining_sec:02d} | "
            f"Speed: {progress_data.fps:.1f} fps"
        )
        self.progress_label.configure(text=status_text)

        if progress_data.original_frame.size > 0:
            self.frame_viewer.update_frames(
                progress_data.original_frame,
                progress_data.anonymized_frame
            )

        if progress_data.frame_number >= progress_data.total_frames:
            self._processing_complete()

    def _processing_complete(self):
        self.is_processing = False
        self.start_button.configure(state=tk.NORMAL)
        self.cancel_button.configure(state=tk.DISABLED)
        self.progress_label.configure(text="Complete!")

        if self.output_path:
            self.video_player.load_video(self.output_path)

        messagebox.showinfo("Success", f"Video saved to:\n{self.output_path}")

    def _on_close(self):
        """Cleanup resources before closing."""
        if self.is_processing:
            self.processor.cancel()

        if hasattr(self, 'video_player'):
            self.video_player.cleanup()

        self.root.destroy()
