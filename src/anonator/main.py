import customtkinter as ctk
import logging
from pathlib import Path

try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

from anonator.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    # Set CustomTkinter appearance mode
    ctk.set_appearance_mode("dark")

    if DND_AVAILABLE:
        # Note: TkinterDnD is not compatible with CustomTkinter
        # Using CustomTkinter instead
        root = ctk.CTk()
        logging.warning("Drag-and-drop with CustomTkinter not yet implemented")
    else:
        root = ctk.CTk()
        logging.warning("tkinterdnd2 not available, drag-and-drop disabled")

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
