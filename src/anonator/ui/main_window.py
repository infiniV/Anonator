import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
from anonator.ui.theme import THEME
from anonator.ui.widgets import Card, Button, Label, SectionLabel, FieldLabel, Entry, DropZone, Separator


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Anonator - Video Face Anonymization")
        self.root.geometry("1400x900")
        self.root.configure(bg=THEME.colors.bg_primary)
        self._enable_dpi_awareness()

        self.processor = VideoProcessor(progress_callback=self._on_progress)
        self.progress_queue = queue.Queue()

        self.input_path = None
        self.output_path = None
        self.is_processing = False

        self._setup_styles()
        self._setup_ui()
        self._setup_drag_drop()
        self._poll_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _enable_dpi_awareness(self):
        """Enable DPI awareness for crisp rendering on high-DPI displays."""
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TCombobox',
                       fieldbackground=THEME.colors.bg_tertiary,
                       background=THEME.colors.bg_tertiary,
                       foreground=THEME.colors.text_primary,
                       borderwidth=0,
                       arrowcolor=THEME.colors.text_secondary)
        style.map('TCombobox',
                 fieldbackground=[('readonly', THEME.colors.bg_tertiary)],
                 selectbackground=[('readonly', THEME.colors.bg_tertiary)])

        style.configure('Horizontal.TProgressbar',
                       background=THEME.colors.accent_primary,
                       troughcolor=THEME.colors.bg_tertiary,
                       borderwidth=0,
                       thickness=6)

    def _setup_ui(self):
        container = tk.Frame(self.root, bg=THEME.colors.bg_primary)
        container.pack(fill=tk.BOTH, expand=True)

        left_pane = self._create_left_pane(container)
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, padx=THEME.spacing.pad_xl, pady=THEME.spacing.pad_xl)

        separator = tk.Frame(container, bg=THEME.colors.border_primary, width=1)
        separator.pack(side=tk.LEFT, fill=tk.Y)

        right_pane = self._create_right_pane(container)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=THEME.spacing.pad_xl, pady=THEME.spacing.pad_xl)

    def _create_left_pane(self, parent):
        """Create left sidebar with controls."""
        pane = tk.Frame(parent, bg=THEME.colors.bg_primary, width=420)
        pane.pack_propagate(False)

        title = Label(pane, text="ANONATOR", variant="heading")
        title.pack(anchor='w', pady=(0, THEME.spacing.pad_xl))

        subtitle = Label(pane, text="Video Face Anonymization", variant="secondary")
        subtitle.pack(anchor='w', pady=(0, THEME.spacing.pad_2xl))

        file_card = Card(pane)
        file_card.pack(fill=tk.X, pady=(0, THEME.spacing.pad_xl))
        self._create_file_section(file_card)

        settings_card = Card(pane)
        settings_card.pack(fill=tk.X, pady=(0, THEME.spacing.pad_xl))
        self._create_settings_section(settings_card)

        hipaa_card = Card(pane)
        hipaa_card.pack(fill=tk.X, pady=(0, THEME.spacing.pad_xl))
        self._create_hipaa_section(hipaa_card)

        actions_card = Card(pane)
        actions_card.pack(fill=tk.X, pady=(0, THEME.spacing.pad_xl))
        self._create_actions_section(actions_card)

        progress_card = Card(pane)
        progress_card.pack(fill=tk.X)
        self._create_progress_section(progress_card)

        return pane

    def _create_file_section(self, parent):
        """Create file selection section."""
        section = tk.Frame(parent, bg=THEME.colors.bg_secondary)
        section.pack(fill=tk.BOTH, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        SectionLabel(section, text="Input File").pack(anchor='w', pady=(0, THEME.spacing.pad_base))

        self.drop_zone = DropZone(section, height=100)
        self.drop_zone.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))
        self.drop_zone.label.bind('<Button-1>', lambda e: self._browse_file())
        self.drop_zone.pack_propagate(False)

        self.file_label = Label(section, text="No file selected", variant="secondary")
        self.file_label.pack(anchor='w')

    def _create_settings_section(self, parent):
        """Create settings section."""
        section = tk.Frame(parent, bg=THEME.colors.bg_secondary)
        section.pack(fill=tk.BOTH, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        SectionLabel(section, text="Processing Settings").pack(anchor='w', pady=(0, THEME.spacing.pad_lg))

        mode_frame = tk.Frame(section, bg=THEME.colors.bg_secondary)
        mode_frame.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        FieldLabel(mode_frame, text="Anonymization Mode").pack(anchor='w', pady=(0, THEME.spacing.pad_xs))

        self.mode_var = tk.StringVar(value=STANDARD_MODE.anonymization_mode)
        mode_options = ["blur", "solid", "mosaic"]
        self.mode_menu = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=mode_options,
            state='readonly',
            font=(THEME.typography.font_family, THEME.typography.size_base),
            width=30
        )
        self.mode_menu.pack(fill=tk.X)

        threshold_frame = tk.Frame(section, bg=THEME.colors.bg_secondary)
        threshold_frame.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        FieldLabel(threshold_frame, text="Detection Threshold").pack(anchor='w', pady=(0, THEME.spacing.pad_xs))

        self.threshold_var = tk.DoubleVar(value=STANDARD_MODE.detection_threshold)
        self.threshold_entry = Entry(threshold_frame, textvariable=self.threshold_var)
        self.threshold_entry.pack(fill=tk.X, ipady=THEME.spacing.pad_xs)

        mask_frame = tk.Frame(section, bg=THEME.colors.bg_secondary)
        mask_frame.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        FieldLabel(mask_frame, text="Mask Scale").pack(anchor='w', pady=(0, THEME.spacing.pad_xs))

        self.mask_scale_var = tk.DoubleVar(value=STANDARD_MODE.mask_scale)
        self.mask_scale_entry = Entry(mask_frame, textvariable=self.mask_scale_var)
        self.mask_scale_entry.pack(fill=tk.X, ipady=THEME.spacing.pad_xs)

        Separator(section).pack(fill=tk.X, pady=THEME.spacing.pad_base)

        check_frame = tk.Frame(section, bg=THEME.colors.bg_secondary)
        check_frame.pack(fill=tk.X, pady=(0, THEME.spacing.pad_xs))

        self.multi_pass_var = tk.BooleanVar(value=STANDARD_MODE.multi_pass_enabled)
        self.multi_pass_check = tk.Checkbutton(
            check_frame,
            text="Multi-Pass Detection",
            variable=self.multi_pass_var,
            bg=THEME.colors.bg_secondary,
            fg=THEME.colors.text_primary,
            selectcolor=THEME.colors.bg_tertiary,
            activebackground=THEME.colors.bg_secondary,
            activeforeground=THEME.colors.text_primary,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            anchor='w'
        )
        self.multi_pass_check.pack(fill=tk.X)

        audio_frame = tk.Frame(section, bg=THEME.colors.bg_secondary)
        audio_frame.pack(fill=tk.X)

        self.keep_audio_var = tk.BooleanVar(value=STANDARD_MODE.keep_audio)
        self.keep_audio_check = tk.Checkbutton(
            audio_frame,
            text="Keep Audio Track",
            variable=self.keep_audio_var,
            bg=THEME.colors.bg_secondary,
            fg=THEME.colors.text_primary,
            selectcolor=THEME.colors.bg_tertiary,
            activebackground=THEME.colors.bg_secondary,
            activeforeground=THEME.colors.text_primary,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            anchor='w'
        )
        self.keep_audio_check.pack(fill=tk.X)

    def _create_hipaa_section(self, parent):
        """Create HIPAA mode section."""
        section = tk.Frame(parent, bg=THEME.colors.bg_secondary)
        section.pack(fill=tk.BOTH, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        header = tk.Frame(section, bg=THEME.colors.bg_secondary)
        header.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        hipaa_label = tk.Label(
            header,
            text="HIPAA Mode",
            fg=THEME.colors.hipaa,
            bg=THEME.colors.bg_secondary,
            font=(THEME.typography.font_family, THEME.typography.size_lg, THEME.typography.weight_bold)
        )
        hipaa_label.pack(side=tk.LEFT)

        self.hipaa_mode_var = tk.BooleanVar(value=True)
        self.hipaa_mode_check = tk.Checkbutton(
            header,
            variable=self.hipaa_mode_var,
            bg=THEME.colors.bg_secondary,
            fg=THEME.colors.hipaa,
            selectcolor=THEME.colors.bg_tertiary,
            activebackground=THEME.colors.bg_secondary,
            activeforeground=THEME.colors.hipaa,
            command=self._on_hipaa_mode_toggle
        )
        self.hipaa_mode_check.pack(side=tk.RIGHT)

        warning = tk.Label(
            section,
            text="Ultra-sensitive detection for medical compliance",
            fg=THEME.colors.text_tertiary,
            bg=THEME.colors.bg_secondary,
            font=(THEME.typography.font_family, THEME.typography.size_xs),
            anchor='w'
        )
        warning.pack(anchor='w')

        self._on_hipaa_mode_toggle()

    def _create_actions_section(self, parent):
        """Create action buttons section."""
        section = tk.Frame(parent, bg=THEME.colors.bg_secondary)
        section.pack(fill=tk.BOTH, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        self.start_button = Button(
            section,
            text="Start Processing",
            variant="primary",
            command=self._start_processing,
            state=tk.DISABLED
        )
        self.start_button.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

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
        section = tk.Frame(parent, bg=THEME.colors.bg_secondary)
        section.pack(fill=tk.BOTH, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        SectionLabel(section, text="Progress").pack(anchor='w', pady=(0, THEME.spacing.pad_base))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            section,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style='Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, THEME.spacing.pad_base))

        self.progress_label = Label(section, text="Ready", variant="secondary")
        self.progress_label.pack(anchor='w')

    def _create_right_pane(self, parent):
        """Create right pane with video preview."""
        pane = tk.Frame(parent, bg=THEME.colors.bg_primary)

        title = Label(pane, text="Preview", variant="heading")
        title.pack(anchor='w', pady=(0, THEME.spacing.pad_xl))

        self.frame_viewer = FrameComparisonViewer(pane)

        self.video_player = VideoPlayer(pane)

        return pane

    def _setup_drag_drop(self):
        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind('<<Drop>>', self._on_drop)

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
                self.mode_menu.config(state='disabled')
                self.threshold_entry.config(state='disabled')
                self.mask_scale_entry.config(state='disabled')
                self.multi_pass_check.config(state='disabled')
                self.keep_audio_check.config(state='disabled')
        else:
            self.mode_menu.config(state='readonly')
            self.threshold_entry.config(state='normal')
            self.mask_scale_entry.config(state='normal')
            self.multi_pass_check.config(state='normal')
            self.keep_audio_check.config(state='normal')

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

        self.file_label.config(text=f"{file_path.name}")
        self.start_button.config(state=tk.NORMAL)
        self.drop_zone.set_text(f"Selected: {file_path.name}")

    def _start_processing(self):
        if not self.input_path:
            return

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.config(text="Starting...")

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
        self.progress_label.config(text="Cancelling...")
        self.cancel_button.config(state=tk.DISABLED)

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
        self.progress_label.config(text=status_text)

        if progress_data.original_frame.size > 0:
            self.frame_viewer.update_frames(
                progress_data.original_frame,
                progress_data.anonymized_frame
            )

        if progress_data.frame_number >= progress_data.total_frames:
            self._processing_complete()

    def _processing_complete(self):
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Complete!")

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
