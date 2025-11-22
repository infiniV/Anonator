"""Entry point for Anonator Launcher executable."""

import sys
from pathlib import Path

# Add launcher package to path if running as frozen app
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    bundle_dir = Path(sys._MEIPASS)
else:
    # Running as script
    bundle_dir = Path(__file__).parent

# Import and run launcher
from launcher import main

if __name__ == "__main__":
    main()
