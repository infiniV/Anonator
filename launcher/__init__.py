"""Anonator Launcher Package - Progressive installation system."""

from .launcher import AnonatorLauncher, main
from .model_manager import ModelManager
from .config import VERSION, APP_NAME

__version__ = VERSION
__all__ = ['AnonatorLauncher', 'ModelManager', 'main', 'VERSION', 'APP_NAME']
