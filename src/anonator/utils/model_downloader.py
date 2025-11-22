"""On-demand model downloader for Anonator UI."""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Callable
import threading

logger = logging.getLogger(__name__)

# Try to import launcher model manager
try:
    # Add launcher to path if needed
    launcher_dir = Path(__file__).parent.parent.parent.parent / "launcher"

    if launcher_dir.exists():
        sys.path.insert(0, str(launcher_dir.parent))
        from launcher.model_manager import ModelManager
        from launcher.config import MODELS_DIR, VENV_DIR
        MODEL_MANAGER_AVAILABLE = True
    else:
        MODEL_MANAGER_AVAILABLE = False
        logger.warning("Launcher model manager not available")

except ImportError as e:
    MODEL_MANAGER_AVAILABLE = False
    logger.warning(f"Could not import model manager: {e}")


class ModelDownloadDialog:
    """Show dialog for downloading model dependencies."""

    def __init__(self, model_name: str, parent_window=None):
        """Initialize download dialog.

        Args:
            model_name: Name of model to download
            parent_window: Parent tkinter window
        """
        self.model_name = model_name
        self.parent = parent_window
        self.dialog = None
        self.progress_var = None
        self.status_label = None
        self.completed = False
        self.success = False

    def show(self, progress_callback: Optional[Callable] = None) -> bool:
        """Show download dialog and start download.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            True if download successful
        """
        try:
            import customtkinter as ctk
            from tkinter import messagebox

            # Create dialog
            self.dialog = ctk.CTkToplevel(self.parent) if self.parent else ctk.CTk()
            self.dialog.title(f"Downloading {self.model_name}")
            self.dialog.geometry("500x200")
            self.dialog.resizable(False, False)
            self.dialog.transient(self.parent)
            self.dialog.grab_set()

            # Status label
            self.status_label = ctk.CTkLabel(
                self.dialog,
                text=f"Downloading dependencies for {self.model_name}...",
                font=("Helvetica", 14)
            )
            self.status_label.pack(pady=20)

            # Progress bar
            self.progress_var = ctk.CTkProgressBar(self.dialog, width=400)
            self.progress_var.pack(pady=20)
            self.progress_var.set(0)

            # Details label
            details_label = ctk.CTkLabel(
                self.dialog,
                text="This may take a few minutes...",
                font=("Helvetica", 11),
                text_color="gray"
            )
            details_label.pack(pady=10)

            # Start download in thread
            def download_thread():
                try:
                    def update_progress(message, progress):
                        self.status_label.configure(text=message)
                        self.progress_var.set(progress)
                        self.dialog.update()

                    self.success = download_model_dependencies(
                        self.model_name,
                        update_progress
                    )

                    if self.success:
                        self.status_label.configure(text=f"{self.model_name} ready!")
                        self.progress_var.set(1.0)
                        self.dialog.after(1000, self.dialog.destroy)
                    else:
                        messagebox.showerror(
                            "Download Failed",
                            f"Failed to download {self.model_name}.\n"
                            "Please check your internet connection and try again.",
                            parent=self.dialog
                        )
                        self.dialog.destroy()

                except Exception as e:
                    logger.exception(f"Download error: {e}")
                    messagebox.showerror(
                        "Error",
                        f"Error downloading model: {e}",
                        parent=self.dialog
                    )
                    self.dialog.destroy()

                self.completed = True

            thread = threading.Thread(target=download_thread, daemon=True)
            thread.start()

            # Show dialog (blocks until closed)
            self.dialog.mainloop()

            return self.success

        except Exception as e:
            logger.error(f"Failed to show download dialog: {e}")
            return False


def is_model_installed(model_name: str) -> bool:
    """Check if model dependencies are installed.

    Args:
        model_name: Name of the model

    Returns:
        True if model is ready to use
    """
    if not MODEL_MANAGER_AVAILABLE:
        # Assume installed if manager not available (standalone mode)
        return True

    try:
        manager = ModelManager(MODELS_DIR, VENV_DIR)
        return manager.is_model_installed(model_name)
    except Exception as e:
        logger.warning(f"Could not check model status: {e}")
        return True  # Assume installed to avoid blocking


def download_model_dependencies(
    model_name: str,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> bool:
    """Download and install model dependencies.

    Args:
        model_name: Name of the model
        progress_callback: Optional callback(message, progress)

    Returns:
        True if successful
    """
    if not MODEL_MANAGER_AVAILABLE:
        logger.warning("Model manager not available")
        return False

    try:
        logger.info(f"Downloading dependencies for {model_name}")

        manager = ModelManager(MODELS_DIR, VENV_DIR)
        success = manager.install_model_dependencies(model_name, progress_callback)

        if success:
            logger.info(f"{model_name} dependencies installed successfully")
        else:
            logger.error(f"Failed to install {model_name} dependencies")

        return success

    except Exception as e:
        logger.exception(f"Error downloading model: {e}")
        return False


def ensure_model_available(
    model_name: str,
    parent_window=None,
    auto_download: bool = True
) -> bool:
    """Ensure model is available, download if needed.

    Args:
        model_name: Name of the model
        parent_window: Parent window for dialogs
        auto_download: If True, automatically download if not installed

    Returns:
        True if model is available
    """
    # Check if already installed
    if is_model_installed(model_name):
        return True

    if not auto_download:
        return False

    # Not installed - download
    logger.info(f"Model {model_name} not installed. Downloading...")

    dialog = ModelDownloadDialog(model_name, parent_window)
    success = dialog.show()

    return success
