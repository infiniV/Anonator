import tkinter as tk
import logging

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
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        logging.warning("tkinterdnd2 not available, drag-and-drop disabled")

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
