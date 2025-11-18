"""
PyInstaller runtime hook for PyTorch face_detection package.
This sets up the torch hub cache directory for the frozen application.
"""
import os
import sys
from pathlib import Path

# When frozen, redirect torch hub cache to bundled models
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    bundle_dir = Path(sys._MEIPASS)
    torch_hub_dir = bundle_dir / 'torch_hub_checkpoints'

    # Set environment variable to point to bundled models
    if torch_hub_dir.exists():
        os.environ['TORCH_HOME'] = str(bundle_dir)
        # Create a fake .cache structure
        cache_dir = bundle_dir / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Symlink or copy model files to expected location
        for model_file in torch_hub_dir.glob('*'):
            if model_file.is_file():
                target = cache_dir / model_file.name
                if not target.exists():
                    try:
                        # Try to create symlink (fast)
                        target.symlink_to(model_file)
                    except (OSError, NotImplementedError):
                        # Fall back to copy if symlinks not supported
                        import shutil
                        shutil.copy2(model_file, target)
