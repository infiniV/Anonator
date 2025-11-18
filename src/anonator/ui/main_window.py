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
from anonator.core.config import HIPAA_MODE, STANDARD_MODE, get_mode_config


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Anonator - Video Face Anonymization")
        self.root.geometry("1000x700")
        self.root.configure(bg='#000000')

        self.processor = VideoProcessor(progress_callback=self._on_progress)
        self.progress_queue = queue.Queue()

        self.input_path = None
        self.output_path = None
        self.is_processing = False

        self._setup_ui()
        self._setup_drag_drop()
        self._poll_queue()

    def _setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#000000')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            main_frame,
            text="ANONATOR",
            font=('Arial', 24, 'bold'),
            fg='#FFFFFF',
            bg='#000000'
        )
        title_label.pack(pady=(0, 20))

        self.drop_zone = tk.Frame(
            main_frame,
            bg='#000000',
            highlightbackground='#FFFFFF',
            highlightthickness=2,
            height=100
        )
        self.drop_zone.pack(fill=tk.X, pady=(0, 20))
        self.drop_zone.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_zone,
            text="Drop video file here or click to browse",
            font=('Arial', 12),
            fg='#FFFFFF',
            bg='#000000'
        )
        self.drop_label.pack(expand=True)
        self.drop_label.bind('<Button-1>', lambda e: self._browse_file())

        self.file_label = tk.Label(
            main_frame,
            text="No file selected",
            font=('Arial', 10),
            fg='#FFFFFF',
            bg='#000000'
        )
        self.file_label.pack(pady=(0, 20))

        settings_frame = tk.Frame(main_frame, bg='#000000')
        settings_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            settings_frame,
            text="Mode:",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        ).grid(row=0, column=0, sticky=tk.W, padx=5)

        self.mode_var = tk.StringVar(value=STANDARD_MODE.anonymization_mode)
        mode_options = ["blur", "solid", "mosaic"]
        self.mode_menu = ttk.Combobox(
            settings_frame,
            textvariable=self.mode_var,
            values=mode_options,
            state='readonly',
            width=15
        )
        self.mode_menu.grid(row=0, column=1, padx=5)

        tk.Label(
            settings_frame,
            text="Threshold:",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        ).grid(row=0, column=2, sticky=tk.W, padx=5)

        self.threshold_var = tk.DoubleVar(value=STANDARD_MODE.detection_threshold)
        self.threshold_entry = tk.Entry(
            settings_frame,
            textvariable=self.threshold_var,
            width=10
        )
        self.threshold_entry.grid(row=0, column=3, padx=5)

        settings_frame2 = tk.Frame(main_frame, bg='#000000')
        settings_frame2.pack(fill=tk.X, pady=(0, 20))

        tk.Label(
            settings_frame2,
            text="Mask Scale:",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        ).grid(row=0, column=0, sticky=tk.W, padx=5)

        self.mask_scale_var = tk.DoubleVar(value=STANDARD_MODE.mask_scale)
        self.mask_scale_entry = tk.Entry(
            settings_frame2,
            textvariable=self.mask_scale_var,
            width=10
        )
        self.mask_scale_entry.grid(row=0, column=1, padx=5)

        tk.Label(
            settings_frame2,
            text="Multi-Pass:",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        ).grid(row=0, column=2, sticky=tk.W, padx=5)

        self.multi_pass_var = tk.BooleanVar(value=STANDARD_MODE.multi_pass_enabled)
        self.multi_pass_check = tk.Checkbutton(
            settings_frame2,
            variable=self.multi_pass_var,
            bg='#000000',
            fg='#FFFFFF',
            selectcolor='#000000',
            activebackground='#000000',
            activeforeground='#FFFFFF'
        )
        self.multi_pass_check.grid(row=0, column=3, padx=5)

        tk.Label(
            settings_frame2,
            text="HIPAA Mode:",
            fg='#FF0000',
            bg='#000000',
            font=('Arial', 10, 'bold')
        ).grid(row=0, column=4, sticky=tk.W, padx=15)

        self.hipaa_mode_var = tk.BooleanVar(value=True)
        self.hipaa_mode_check = tk.Checkbutton(
            settings_frame2,
            variable=self.hipaa_mode_var,
            bg='#000000',
            fg='#FF0000',
            selectcolor='#000000',
            activebackground='#000000',
            activeforeground='#FF0000',
            command=self._on_hipaa_mode_toggle
        )
        self.hipaa_mode_check.grid(row=0, column=5, padx=5)

        hipaa_warning = tk.Label(
            settings_frame2,
            text="(Ultra-sensitive detection for medical compliance)",
            fg='#888888',
            bg='#000000',
            font=('Arial', 8)
        )
        hipaa_warning.grid(row=0, column=6, sticky=tk.W, padx=5)

        tk.Label(
            settings_frame,
            text="Keep Audio:",
            fg='#FFFFFF',
            bg='#000000',
            font=('Arial', 10)
        ).grid(row=0, column=4, sticky=tk.W, padx=5)

        self.keep_audio_var = tk.BooleanVar(value=STANDARD_MODE.keep_audio)
        self.keep_audio_check = tk.Checkbutton(
            settings_frame,
            variable=self.keep_audio_var,
            bg='#000000',
            fg='#FFFFFF',
            selectcolor='#000000',
            activebackground='#000000',
            activeforeground='#FFFFFF'
        )
        self.keep_audio_check.grid(row=0, column=5, padx=5)

        self._on_hipaa_mode_toggle()

        self.frame_viewer = FrameComparisonViewer(main_frame)

        progress_frame = tk.Frame(main_frame, bg='#000000')
        progress_frame.pack(fill=tk.X, pady=(20, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        self.progress_label = tk.Label(
            progress_frame,
            text="Ready",
            font=('Arial', 10),
            fg='#FFFFFF',
            bg='#000000'
        )
        self.progress_label.pack()

        button_frame = tk.Frame(main_frame, bg='#000000')
        button_frame.pack(pady=(20, 0))

        self.start_button = tk.Button(
            button_frame,
            text="Start Processing",
            command=self._start_processing,
            bg='#FFFFFF',
            fg='#000000',
            font=('Arial', 12, 'bold'),
            width=20,
            height=2,
            state=tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_processing,
            bg='#FFFFFF',
            fg='#000000',
            font=('Arial', 12),
            width=10,
            height=2,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)

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

        self.file_label.config(text=f"Input: {file_path.name}")
        self.start_button.config(state=tk.NORMAL)
        self.drop_label.config(text=f"Selected: {file_path.name}")

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
        messagebox.showinfo("Success", f"Video saved to:\n{self.output_path}")
