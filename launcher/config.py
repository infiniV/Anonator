"""Configuration and constants for Anonator launcher."""

import sys
from pathlib import Path
import os

# Version
VERSION = "1.0.0"
APP_NAME = "Anonator"

# Paths
if sys.platform == "win32":
    APP_DATA_DIR = Path(os.getenv("APPDATA")) / APP_NAME
else:
    APP_DATA_DIR = Path.home() / f".{APP_NAME.lower()}"

VENV_DIR = APP_DATA_DIR / "venv"
PYTHON_DIR = APP_DATA_DIR / "python"
MODELS_DIR = APP_DATA_DIR / "models"
CACHE_DIR = APP_DATA_DIR / "cache"
CONFIG_FILE = APP_DATA_DIR / "config.json"
LOG_FILE = APP_DATA_DIR / "launcher.log"

# Python version
PYTHON_VERSION = "3.11.9"

# Python embedded download URLs
PYTHON_URLS = {
    "win32": f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip",
    "linux": None,  # Use system Python on Linux
    "darwin": None,  # Use system Python on macOS
}

# Get-pip URL
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# UV (fast Python package installer) version and URL
UV_VERSION = "0.9.11"
UV_URLS = {
    "win32": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip",
    "linux": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz",
    "darwin": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-apple-darwin.tar.gz",
}

# Model registry with download information
MODEL_REGISTRY = {
    "MediaPipe": {
        "display_name": "MediaPipe",
        "size_mb": 20,
        "description": "Fast, CPU-friendly detector",
        "dependencies": ["mediapipe>=0.10.21"],
        "default": True,
        "auto_download": True,  # Model downloads automatically on first use
    },
    "MTCNN": {
        "display_name": "MTCNN",
        "size_mb": 2,
        "description": "Accurate, moderate speed",
        "dependencies": ["facenet-pytorch>=2.6.0"],
        "default": False,
        "auto_download": True,  # PyTorch hub downloads automatically
    },
    "SCRFD-2.5GF": {
        "display_name": "SCRFD-2.5GF",
        "size_mb": 10,
        "description": "Balanced GPU model",
        "dependencies": ["insightface>=0.7.3", "onnxruntime-gpu>=1.20.2"],
        "default": False,
        "auto_download": True,  # InsightFace downloads from their repo
    },
    "SCRFD-10GF": {
        "display_name": "SCRFD-10GF",
        "size_mb": 50,
        "description": "High accuracy GPU model",
        "dependencies": ["insightface>=0.7.3", "onnxruntime-gpu>=1.20.2"],
        "default": False,
        "auto_download": True,
    },
    "SCRFD-34GF": {
        "display_name": "SCRFD-34GF",
        "size_mb": 100,
        "description": "Maximum accuracy GPU model",
        "dependencies": ["insightface>=0.7.3", "onnxruntime-gpu>=1.20.2"],
        "default": False,
        "auto_download": True,
    },
    "YOLOv8-Face": {
        "display_name": "YOLOv8-Face",
        "size_mb": 6,
        "description": "Fast GPU detection",
        "dependencies": ["ultralytics>=8.3.230"],
        "default": False,
        "auto_download": True,  # Downloads from ultralytics
    },
    "YOLO11-Face": {
        "display_name": "YOLO11-Face",
        "size_mb": 10,
        "description": "Latest YOLO model",
        "dependencies": ["ultralytics>=8.3.230"],
        "default": False,
        "auto_download": True,
    },
    "RetinaFace-MobileNet": {
        "display_name": "RetinaFace-MobileNet",
        "size_mb": 27,
        "description": "Mobile-optimized detector",
        "dependencies": ["retinaface-pytorch>=0.0.7"],
        "default": False,
        "auto_download": True,
    },
}

# Core dependencies (always installed)
CORE_DEPENDENCIES = [
    "customtkinter>=5.2.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "numpy<2",
    "tkinterdnd2>=0.3.0",
    "imageio>=2.25.0",
    "imageio-ffmpeg>=0.4.6",
    "requests>=2.31.0",
]

# Custom git dependencies (installed with --no-build-isolation)
CUSTOM_GIT_DEPENDENCIES = [
    "git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git",
]

# PyTorch installation (CPU vs GPU)
TORCH_CPU = [
    "torch",
    "torchvision",
    "--index-url",
    "https://download.pytorch.org/whl/cpu"
]

TORCH_GPU = [
    "torch",
    "torchvision",
    "--index-url",
    "https://download.pytorch.org/whl/cu121"
]

# Update check URL (update this to your actual server)
UPDATE_CHECK_URL = "https://api.github.com/repos/YOUR_USERNAME/anonator/releases/latest"

# UI Theme
THEME = {
    "fg_color": ["#f0f0f0", "#1a1a1a"],
    "bg_color": ["#ffffff", "#0d0d0d"],
    "accent": ["#3b8ed0", "#1f6aa5"],
    "text": ["#000000", "#ffffff"],
}
