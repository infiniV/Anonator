"""Anonator Launcher - Progressive installation and app launcher."""

import sys
import os
import json
import logging
import threading
import subprocess
from pathlib import Path
from typing import Optional, List

import customtkinter as ctk
from tkinter import messagebox

from .config import (
    VERSION, APP_NAME, APP_DATA_DIR, VENV_DIR, PYTHON_DIR, MODELS_DIR,
    CACHE_DIR, CONFIG_FILE, LOG_FILE, PYTHON_VERSION, UV_VERSION, UV_URLS,
    MODEL_REGISTRY, CORE_DEPENDENCIES, CUSTOM_GIT_DEPENDENCIES, TORCH_CPU, TORCH_GPU
)
from .utils import (
    download_file, extract_archive, detect_cuda, get_uv_path,
    get_venv_python_path, run_command, format_size, ensure_directory
)
from .model_manager import ModelManager

# Import theme from main app (will use fallback if not available during packaging)
try:
    from anonator.ui.theme import THEME
except ImportError:
    # Fallback theme for standalone launcher
    from dataclasses import dataclass

    @dataclass
    class ColorPalette:
        bg_primary: str = "#2a2024"
        bg_secondary: str = "#392f35"
        bg_tertiary: str = "#30272c"
        bg_hover: str = "#463a41"
        border_primary: str = "#463a41"
        border_secondary: str = "#392f35"
        text_primary: str = "#f2e9e4"
        text_secondary: str = "#d7c6bc"
        accent_primary: str = "#ff7e5f"
        accent_hover: str = "#e66a4f"
        button_secondary_bg: str = "#463a41"
        button_secondary_hover: str = "#554a50"

    @dataclass
    class Typography:
        font_family: str = "Segoe UI"
        font_family_mono: str = "Consolas"
        size_sm: int = 12
        size_base: int = 14
        size_lg: int = 16
        size_2xl: int = 28

    @dataclass
    class Spacing:
        pad_base: int = 12
        pad_lg: int = 20
        radius_base: int = 8
        radius_lg: int = 12

    class Theme:
        def __init__(self):
            self.colors = ColorPalette()
            self.typography = Typography()
            self.spacing = Spacing()

    THEME = Theme()

# Setup logging
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AnonatorLauncher(ctk.CTk):
    """Main launcher window with installation wizard and app launcher."""

    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} Launcher v{VERSION}")
        self.geometry("800x700")
        self.resizable(False, False)
        self.configure(fg_color=THEME.colors.bg_primary)

        # State
        self.is_installed = False
        self.is_installing = False
        self.installation_cancelled = False
        self.selected_models: List[str] = []
        self.has_gpu = False

        # Model manager (initialized later)
        self.model_manager: Optional[ModelManager] = None

        # UI Elements (created in _setup_ui)
        self.header_label = None
        self.status_label = None
        self.progress_bar = None
        self.log_text = None
        self.model_checkboxes = {}
        self.model_frame = None
        self.install_button = None
        self.launch_button = None
        self.cancel_button = None

        # Setup UI
        self._setup_ui()

        # Check installation status
        self.after(500, self._check_installation)

    def _setup_ui(self):
        """Create all UI elements."""
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=THEME.spacing.pad_lg, pady=THEME.spacing.pad_lg)

        # Header
        self.header_label = ctk.CTkLabel(
            main_frame,
            text=f"{APP_NAME} Setup Wizard",
            font=(THEME.typography.font_family, THEME.typography.size_2xl, "bold"),
            text_color=THEME.colors.text_primary
        )
        self.header_label.pack(pady=(0, THEME.spacing.pad_base))

        # Status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Checking installation...",
            font=(THEME.typography.font_family, THEME.typography.size_base),
            text_color=THEME.colors.text_secondary
        )
        self.status_label.pack(pady=THEME.spacing.pad_base)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            main_frame,
            width=700,
            height=20,
            corner_radius=THEME.spacing.radius_base,
            fg_color=THEME.colors.bg_tertiary,
            progress_color=THEME.colors.accent_primary
        )
        self.progress_bar.pack(pady=THEME.spacing.pad_base)
        self.progress_bar.set(0)

        # Log/Details area
        log_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_secondary,
            border_width=1,
            border_color=THEME.colors.border_primary
        )
        log_frame.pack(fill="both", expand=True, pady=THEME.spacing.pad_base)

        log_label = ctk.CTkLabel(
            log_frame,
            text="Installation Log",
            font=(THEME.typography.font_family, THEME.typography.size_lg, "bold"),
            text_color=THEME.colors.text_primary
        )
        log_label.pack(anchor="w", padx=THEME.spacing.pad_base, pady=(THEME.spacing.pad_base, 5))

        self.log_text = ctk.CTkTextbox(
            log_frame,
            width=700,
            height=250,
            font=(THEME.typography.font_family_mono, THEME.typography.size_sm),
            fg_color=THEME.colors.bg_tertiary,
            text_color=THEME.colors.text_secondary,
            corner_radius=THEME.spacing.radius_base
        )
        self.log_text.pack(padx=THEME.spacing.pad_base, pady=(0, THEME.spacing.pad_base))

        # Model selection frame (hidden initially)
        self.model_frame = ctk.CTkFrame(
            main_frame,
            corner_radius=THEME.spacing.radius_lg,
            fg_color=THEME.colors.bg_secondary,
            border_width=1,
            border_color=THEME.colors.border_primary
        )
        self._create_model_selection()

        # Button frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=THEME.spacing.pad_base)

        self.install_button = ctk.CTkButton(
            button_frame,
            text="Install",
            width=200,
            height=40,
            font=(THEME.typography.font_family, THEME.typography.size_base, "bold"),
            command=self._start_installation,
            state="disabled",
            corner_radius=THEME.spacing.radius_base,
            fg_color=THEME.colors.accent_primary,
            hover_color=THEME.colors.accent_hover,
            text_color="#ffffff"
        )
        self.install_button.pack(side="left", padx=5)

        self.launch_button = ctk.CTkButton(
            button_frame,
            text="Launch Anonator",
            width=200,
            height=40,
            font=(THEME.typography.font_family, THEME.typography.size_base, "bold"),
            command=self._launch_app,
            state="disabled",
            corner_radius=THEME.spacing.radius_base,
            fg_color=THEME.colors.accent_primary,
            hover_color=THEME.colors.accent_hover,
            text_color="#ffffff"
        )
        self.launch_button.pack(side="left", padx=5)

        self.cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            width=120,
            height=40,
            font=(THEME.typography.font_family, THEME.typography.size_base),
            command=self._cancel_installation,
            state="disabled",
            corner_radius=THEME.spacing.radius_base,
            fg_color=THEME.colors.button_secondary_bg,
            hover_color=THEME.colors.button_secondary_hover,
            text_color=THEME.colors.text_primary
        )
        self.cancel_button.pack(side="left", padx=5)

    def _create_model_selection(self):
        """Create model selection checkboxes."""
        label = ctk.CTkLabel(
            self.model_frame,
            text="Select Face Detection Models to Install:",
            font=(THEME.typography.font_family, THEME.typography.size_lg, "bold"),
            text_color=THEME.colors.text_primary
        )
        label.pack(pady=THEME.spacing.pad_base, padx=THEME.spacing.pad_base, anchor="w")

        info_label = ctk.CTkLabel(
            self.model_frame,
            text="Models can be downloaded later if needed. Default: MediaPipe (recommended)",
            font=(THEME.typography.font_family, THEME.typography.size_sm),
            text_color=THEME.colors.text_secondary
        )
        info_label.pack(pady=(0, THEME.spacing.pad_base), padx=THEME.spacing.pad_base, anchor="w")

        # Scrollable frame for models
        model_scroll = ctk.CTkScrollableFrame(
            self.model_frame,
            width=700,
            height=250,
            fg_color=THEME.colors.bg_tertiary,
            corner_radius=THEME.spacing.radius_base
        )
        model_scroll.pack(padx=THEME.spacing.pad_base, pady=(0, THEME.spacing.pad_base), fill="both", expand=True)

        for model_name, model_info in MODEL_REGISTRY.items():
            var = ctk.BooleanVar(value=model_info.get('default', False))
            self.model_checkboxes[model_name] = var

            size_mb = model_info.get('size_mb', 0)
            desc = model_info.get('description', '')

            checkbox = ctk.CTkCheckBox(
                model_scroll,
                text=f"{model_name} ({size_mb} MB) - {desc}",
                variable=var,
                font=(THEME.typography.font_family, THEME.typography.size_sm),
                fg_color=THEME.colors.accent_primary,
                hover_color=THEME.colors.accent_hover,
                border_color=THEME.colors.border_secondary,
                text_color=THEME.colors.text_primary
            )
            checkbox.pack(anchor="w", padx=THEME.spacing.pad_base, pady=3)

    def _log(self, message: str):
        """Add message to log.

        Args:
            message: Message to log
        """
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.update()
        logger.info(message)

    def _update_status(self, message: str, progress: Optional[float] = None):
        """Update status label and progress bar.

        Args:
            message: Status message
            progress: Progress value (0-1) or None to leave unchanged
        """
        self.status_label.configure(text=message)
        if progress is not None:
            self.progress_bar.set(progress)
        self.update()

    def _check_installation(self):
        """Check if Anonator is already installed."""
        self._log("Checking for existing installation...")

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)

                if config.get('installed', False):
                    self._log(f"Found existing installation at {APP_DATA_DIR}")
                    self._log(f"Version: {config.get('version', 'unknown')}")

                    self.is_installed = True
                    self._update_status("Installation found. Ready to launch!", 1.0)
                    self.launch_button.configure(state="normal")
                    self.install_button.configure(state="disabled")

                    # Initialize model manager
                    self.model_manager = ModelManager(MODELS_DIR, VENV_DIR)
                    return

            except Exception as e:
                self._log(f"Error reading config: {e}")

        # No installation found
        self._log("No installation found. First-time setup required.")
        self._update_status("Ready to install", 0)
        self.install_button.configure(state="normal")

        # Show model selection
        self.model_frame.pack(pady=THEME.spacing.pad_base, fill="both", expand=False)

    def _start_installation(self):
        """Start installation in background thread."""
        if self.is_installing:
            return

        # Get selected models
        self.selected_models = [
            name for name, var in self.model_checkboxes.items()
            if var.get()
        ]

        if not self.selected_models:
            messagebox.showwarning(
                "No Models Selected",
                "Please select at least one face detection model to install."
            )
            return

        # Calculate total size
        model_manager_temp = ModelManager(MODELS_DIR)
        total_size_mb = model_manager_temp.get_total_download_size(self.selected_models)

        # Confirm installation
        response = messagebox.askyesno(
            "Confirm Installation",
            f"This will download and install:\n\n"
            f"- UV package manager and Python {PYTHON_VERSION}\n"
            f"- Core dependencies (~500 MB)\n"
            f"- {len(self.selected_models)} face detection model(s) (~{total_size_mb} MB)\n\n"
            f"Installation directory: {APP_DATA_DIR}\n\n"
            f"Continue?"
        )

        if not response:
            return

        # Disable buttons
        self.install_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.is_installing = True
        self.installation_cancelled = False

        # Hide model selection
        self.model_frame.pack_forget()

        # Start installation thread
        thread = threading.Thread(target=self._installation_thread, daemon=True)
        thread.start()

    def _cancel_installation(self):
        """Cancel ongoing installation."""
        self.installation_cancelled = True
        self._log("Installation cancelled by user.")
        self._update_status("Installation cancelled", 0)
        self.cancel_button.configure(state="disabled")

    def _installation_thread(self):
        """Main installation logic (runs in background thread)."""
        try:
            # Create directories
            ensure_directory(APP_DATA_DIR)
            ensure_directory(CACHE_DIR)

            # Step 1: Install UV package manager
            if not self._check_cancelled():
                self._install_uv()

            # Step 2: Create virtual environment with uv
            if not self._check_cancelled():
                self._create_venv()

            # Step 3: Install core dependencies
            if not self._check_cancelled():
                self._install_core_dependencies()

            # Step 4: Install PyTorch
            if not self._check_cancelled():
                self._install_pytorch()

            # Step 5: Install custom dependencies
            if not self._check_cancelled():
                self._install_custom_dependencies()

            # Step 6: Install model dependencies
            if not self._check_cancelled():
                self._install_models()

            # Step 7: Save configuration
            if not self._check_cancelled():
                self._save_config()

            # Installation complete
            if not self.installation_cancelled:
                self._update_status("Installation complete!", 1.0)
                self._log("Anonator is ready to use!")
                self.is_installed = True
                self.launch_button.configure(state="normal")

                messagebox.showinfo(
                    "Installation Complete",
                    "Anonator has been successfully installed!\n\n"
                    "Click 'Launch Anonator' to start the application."
                )

        except Exception as e:
            logger.exception("Installation failed")
            self._log(f"ERROR: Installation failed: {e}")
            self._update_status("Installation failed", 0)

            messagebox.showerror(
                "Installation Failed",
                f"An error occurred during installation:\n\n{e}\n\n"
                f"Check the log for details."
            )

        finally:
            self.is_installing = False
            self.install_button.configure(state="normal" if not self.is_installed else "disabled")
            self.cancel_button.configure(state="disabled")

    def _check_cancelled(self) -> bool:
        """Check if installation was cancelled.

        Returns:
            True if cancelled
        """
        return self.installation_cancelled

    def _install_uv(self):
        """Download and install UV package manager."""
        uv_exe = get_uv_path(APP_DATA_DIR)

        # Check if UV is already installed
        if uv_exe.exists():
            self._log(f"UV v{UV_VERSION} already installed")
            return

        self._update_status("Downloading UV package manager...", 0.05)
        self._log(f"Downloading UV v{UV_VERSION}...")

        uv_url = UV_URLS.get(sys.platform)
        if not uv_url:
            raise RuntimeError(f"UV not supported on platform: {sys.platform}")

        uv_archive = CACHE_DIR / f"uv-{UV_VERSION}.{'zip' if sys.platform == 'win32' else 'tar.gz'}"

        # Only download if not cached
        if not uv_archive.exists():
            def progress(downloaded, total):
                if total > 0:
                    pct = downloaded / total
                    self._update_status(
                        f"Downloading UV... {format_size(downloaded)} / {format_size(total)}",
                        0.05 + (pct * 0.10)
                    )

            download_file(uv_url, uv_archive, progress)
        else:
            self._log(f"Using cached UV archive: {uv_archive}")

        self._update_status("Extracting UV...", 0.15)
        self._log("Extracting UV...")

        extract_archive(uv_archive, APP_DATA_DIR)
        # Keep the archive cached for future use

        self._log("UV installed successfully")

    def _create_venv(self):
        """Create virtual environment with uv."""
        self._update_status("Creating virtual environment...", 0.20)
        self._log("Creating virtual environment with uv...")

        uv_exe = get_uv_path(APP_DATA_DIR)

        # Create venv with uv (automatically installs Python if needed)
        run_command([str(uv_exe), "venv", str(VENV_DIR), "--python", PYTHON_VERSION])

        self._log("Virtual environment created")

    def _install_core_dependencies(self):
        """Install core dependencies with uv."""
        self._update_status("Installing core dependencies...", 0.30)
        self._log("Installing core dependencies...")

        uv_exe = get_uv_path(APP_DATA_DIR)

        # Install core dependencies
        total = len(CORE_DEPENDENCIES)
        for i, dep in enumerate(CORE_DEPENDENCIES):
            if self._check_cancelled():
                return

            progress = 0.30 + ((i + 1) / total * 0.20)
            self._update_status(f"Installing {dep}...", progress)
            self._log(f"Installing {dep}...")

            run_command([str(uv_exe), "pip", "install", "--python", str(VENV_DIR), dep])

        self._log("Core dependencies installed")

    def _install_pytorch(self):
        """Install PyTorch (CPU or GPU based on detection) with uv."""
        self._update_status("Detecting GPU...", 0.50)
        self._log("Detecting GPU...")

        self.has_gpu = detect_cuda()

        if self.has_gpu:
            self._log("CUDA GPU detected! Installing PyTorch with GPU support...")
            torch_deps = TORCH_GPU
        else:
            self._log("No CUDA GPU detected. Installing CPU-only PyTorch...")
            torch_deps = TORCH_CPU

        self._update_status("Installing PyTorch (this may take a few minutes)...", 0.55)

        uv_exe = get_uv_path(APP_DATA_DIR)

        run_command([str(uv_exe), "pip", "install", "--python", str(VENV_DIR)] + torch_deps, timeout=600)

        self._log("PyTorch installed successfully")

    def _install_custom_dependencies(self):
        """Install custom git dependencies and the app itself."""
        self._update_status("Installing custom dependencies...", 0.60)
        self._log("Installing custom dependencies...")

        uv_exe = get_uv_path(APP_DATA_DIR)

        # First, install setuptools (required for building git packages with --no-build-isolation)
        self._log("Installing setuptools...")
        run_command([str(uv_exe), "pip", "install", "--python", str(VENV_DIR), "setuptools"], timeout=300)

        # Install custom git packages with --no-build-isolation
        for dep in CUSTOM_GIT_DEPENDENCIES:
            if self._check_cancelled():
                return

            self._log(f"Installing {dep} (this may take a few minutes)...")
            run_command([str(uv_exe), "pip", "install", "--python", str(VENV_DIR), "--no-build-isolation", dep], timeout=600)

        # Install the app itself in editable mode
        # Find the app source directory
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exe_dir = Path(sys.executable).parent
            app_source_dir = exe_dir.parent / "anonator_app"
        else:
            # Running in development mode
            app_source_dir = Path(__file__).parent.parent

        if app_source_dir.exists() and (app_source_dir / "pyproject.toml").exists():
            self._log(f"Installing Anonator app from {app_source_dir}...")
            run_command([str(uv_exe), "pip", "install", "--python", str(VENV_DIR), "-e", str(app_source_dir)], timeout=600)
            self._log("Anonator app installed successfully")
        else:
            self._log(f"Warning: Could not find app source at {app_source_dir}")

        self._log("Custom dependencies installed")

    def _install_models(self):
        """Install selected model dependencies."""
        self._update_status("Installing model dependencies...", 0.75)
        self._log(f"Installing {len(self.selected_models)} model(s)...")

        # Initialize model manager
        self.model_manager = ModelManager(MODELS_DIR, VENV_DIR)

        def progress_callback(message, progress):
            if self._check_cancelled():
                return
            # Map 0-1 progress to 0.70-0.95 range
            overall_progress = 0.70 + (progress * 0.25)
            self._update_status(message, overall_progress)
            self._log(message)

        results = self.model_manager.install_models_batch(
            self.selected_models,
            progress_callback
        )

        # Log results
        for model_name, success in results.items():
            if success:
                self._log(f"✓ {model_name} installed successfully")
            else:
                self._log(f"✗ {model_name} installation failed")

    def _save_config(self):
        """Save installation configuration."""
        self._update_status("Saving configuration...", 0.95)
        self._log("Saving configuration...")

        config = {
            'installed': True,
            'version': VERSION,
            'python_version': PYTHON_VERSION,
            'app_data_dir': str(APP_DATA_DIR),
            'venv_dir': str(VENV_DIR),
            'python_dir': str(PYTHON_DIR),
            'models_dir': str(MODELS_DIR),
            'has_gpu': self.has_gpu,
            'selected_models': self.selected_models,
        }

        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        self._log("Configuration saved")

    def _launch_app(self):
        """Launch the Anonator application."""
        if not self.is_installed:
            messagebox.showerror("Error", "Anonator is not installed yet.")
            return

        self._log("Launching Anonator...")

        try:
            # Get Python from venv
            python_exe = get_venv_python_path(VENV_DIR)
            self._log(f"Python executable: {python_exe}")

            if not python_exe.exists():
                raise FileNotFoundError(f"Python executable not found at: {python_exe}")

            # Path to main app script
            # Check if running from PyInstaller bundle
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                # Look for anonator_app relative to executable directory
                exe_dir = Path(sys.executable).parent
                self._log(f"Running from PyInstaller bundle. Executable dir: {exe_dir}")

                app_main = exe_dir.parent / "anonator_app" / "src" / "anonator" / "main.py"
                self._log(f"Trying path 1: {app_main}")

                if not app_main.exists():
                    # Try without parent (if in root of distribution)
                    app_main = exe_dir / "anonator_app" / "src" / "anonator" / "main.py"
                    self._log(f"Trying path 2: {app_main}")
            else:
                # Running in development mode
                self._log(f"Running in development mode. __file__: {__file__}")
                app_main = Path(__file__).parent.parent / "src" / "anonator" / "main.py"
                self._log(f"Trying path 1: {app_main}")

                if not app_main.exists():
                    # Try alternative location
                    app_main = Path(__file__).parent / "anonator_app" / "main.py"
                    self._log(f"Trying path 2: {app_main}")

            if not app_main.exists():
                error_msg = (
                    f"Cannot find main.py at {app_main}\n\n"
                    "Please ensure you downloaded the full Anonator setup package, "
                    "not just the launcher. The anonator_app folder with the application "
                    "source code should be in the same directory as the launcher."
                )
                raise FileNotFoundError(error_msg)

            self._log(f"Found app at: {app_main}")
            self._log(f"Starting: {python_exe} {app_main}")

            # Set working directory to src folder (parent of anonator)
            # This allows imports like "from anonator.ui.main_window import MainWindow" to work
            working_dir = app_main.parent.parent
            self._log(f"Working directory: {working_dir}")

            # Launch app in new process
            # Check for debug mode (shows console for error diagnosis)
            debug_mode = os.environ.get('ANONATOR_DEBUG', '0') == '1'

            startupinfo = None
            creationflags = 0
            capture_output = True

            if not debug_mode and sys.platform == "win32":
                # Hide console window in normal mode
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                creationflags = subprocess.CREATE_NO_WINDOW
            else:
                # Debug mode: show console window
                capture_output = False
                self._log("Debug mode: Console window will be visible")

            # Set up clean environment for the app
            env = os.environ.copy()

            # Set PYTHONPATH so Python can find the anonator module
            env['PYTHONPATH'] = str(working_dir)
            self._log(f"PYTHONPATH set to: {working_dir}")

            # Remove any launcher-specific paths that could interfere
            # This prevents Tcl/Tk version conflicts with the launcher's bundled files
            if getattr(sys, 'frozen', False):
                # Remove the launcher's _internal directory from PATH if present
                launcher_internal = str(Path(sys.executable).parent / "_internal")
                if 'PATH' in env:
                    path_entries = env['PATH'].split(os.pathsep)
                    path_entries = [p for p in path_entries if launcher_internal not in p]
                    env['PATH'] = os.pathsep.join(path_entries)

                # Clear Tcl/Tk environment variables that might point to launcher's bundled Tcl
                for key in list(env.keys()):
                    if key.startswith('TCL_') or key.startswith('TK_'):
                        del env[key]

                self._log("Cleaned launcher-specific environment variables")

            # Create log files for app output
            app_log_dir = APP_DATA_DIR / "app_logs"
            app_log_dir.mkdir(exist_ok=True)

            app_stdout_log = app_log_dir / "app_stdout.log"
            app_stderr_log = app_log_dir / "app_stderr.log"

            self._log(f"App output will be logged to: {app_log_dir}")

            if capture_output:
                # Open log files
                stdout_file = open(app_stdout_log, 'w')
                stderr_file = open(app_stderr_log, 'w')

                proc = subprocess.Popen(
                    [str(python_exe), str(app_main)],
                    cwd=working_dir,
                    env=env,
                    startupinfo=startupinfo,
                    creationflags=creationflags,
                    stdout=stdout_file,
                    stderr=stderr_file
                )

                # Store file handles to close later
                proc._stdout_file = stdout_file
                proc._stderr_file = stderr_file
            else:
                proc = subprocess.Popen(
                    [str(python_exe), str(app_main)],
                    cwd=working_dir,
                    env=env,
                    startupinfo=startupinfo,
                    creationflags=creationflags
                )

            self._log(f"Process started with PID: {proc.pid}")

            # Wait a moment to check if process crashes immediately
            import time
            time.sleep(2.0)  # Wait longer to catch startup errors

            if proc.poll() is not None:
                # Process already exited
                error_msg = f"Application exited immediately with code {proc.returncode}\n\n"

                if capture_output:
                    # Close file handles and read logs
                    if hasattr(proc, '_stdout_file'):
                        proc._stdout_file.close()
                        proc._stderr_file.close()

                    # Read the log files
                    if app_stderr_log.exists():
                        with open(app_stderr_log, 'r') as f:
                            stderr_content = f.read()
                            if stderr_content:
                                error_msg += f"Error output:\n{stderr_content}"

                    if app_stdout_log.exists():
                        with open(app_stdout_log, 'r') as f:
                            stdout_content = f.read()
                            if stdout_content:
                                error_msg += f"\n\nStandard output:\n{stdout_content}"

                    error_msg += f"\n\nFull logs available at:\n{app_stdout_log}\n{app_stderr_log}"
                else:
                    error_msg += "Run with ANONATOR_DEBUG=1 environment variable to see console output."

                self._log(error_msg)
                raise RuntimeError(error_msg)

            # Process is running - inform user about log location
            self._log(f"Note: App output is being logged to {app_log_dir}")

            # Close launcher
            self._log("Anonator launched successfully. Closing launcher...")
            self.after(1000, self.destroy)

        except Exception as e:
            logger.exception("Failed to launch app")
            self._log(f"ERROR: {e}")
            messagebox.showerror(
                "Launch Failed",
                f"Failed to launch Anonator:\n\n{e}\n\n"
                f"Check the installation log above for details."
            )


def main():
    """Entry point for launcher."""
    try:
        # Set appearance to match main app
        ctk.set_appearance_mode("dark")
        # Don't set color theme - we use custom THEME colors

        # Create and run launcher
        app = AnonatorLauncher()
        app.mainloop()
    except Exception as e:
        logger.exception("Fatal error in launcher")
        # Show error dialog even if UI fails
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Launcher Error",
                f"Failed to start launcher:\n\n{e}\n\n"
                f"Check log file at:\n{LOG_FILE}"
            )
            root.destroy()
        except:
            # Last resort: print to console
            print(f"FATAL ERROR: {e}")
            print(f"Check log file at: {LOG_FILE}")
            input("Press Enter to exit...")


if __name__ == "__main__":
    main()
