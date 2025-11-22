"""Model manager for on-demand model downloads and dependency installation."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
import subprocess

from .config import MODEL_REGISTRY, MODELS_DIR
from .utils import get_pip_path, run_command, ensure_directory

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages face detection model downloads and dependencies."""

    def __init__(self, models_dir: Path = MODELS_DIR, venv_dir: Optional[Path] = None):
        """Initialize model manager.

        Args:
            models_dir: Directory to store model files
            venv_dir: Virtual environment directory (for pip)
        """
        self.models_dir = ensure_directory(models_dir)
        self.venv_dir = venv_dir
        self.registry_file = models_dir / "registry.json"
        self.installed_models = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict]:
        """Load registry of installed models.

        Returns:
            Dictionary of installed models with metadata
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save registry of installed models."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.installed_models, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def is_model_installed(self, model_name: str) -> bool:
        """Check if model and its dependencies are installed.

        Args:
            model_name: Name of the model

        Returns:
            True if model is ready to use
        """
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model: {model_name}")
            return False

        # Check if marked as installed in registry
        if model_name in self.installed_models:
            return self.installed_models[model_name].get('installed', False)

        return False

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information from registry.

        Args:
            model_name: Name of the model

        Returns:
            Model info dictionary or None
        """
        return MODEL_REGISTRY.get(model_name)

    def get_available_models(self) -> List[str]:
        """Get list of all available models.

        Returns:
            List of model names
        """
        return list(MODEL_REGISTRY.keys())

    def get_default_models(self) -> List[str]:
        """Get list of default models to install.

        Returns:
            List of model names marked as default
        """
        return [
            name for name, info in MODEL_REGISTRY.items()
            if info.get('default', False)
        ]

    def install_model_dependencies(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """Install dependencies for a model.

        Args:
            model_name: Name of the model
            progress_callback: Optional callback(message, progress)

        Returns:
            True if successful

        Raises:
            ValueError: If model not found or venv not configured
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        if not self.venv_dir:
            raise ValueError("Virtual environment not configured")

        model_info = MODEL_REGISTRY[model_name]
        dependencies = model_info.get('dependencies', [])

        if not dependencies:
            logger.info(f"No dependencies for {model_name}")
            self._mark_installed(model_name)
            return True

        logger.info(f"Installing dependencies for {model_name}: {dependencies}")

        pip_path = get_pip_path(self.venv_dir)

        try:
            for i, dep in enumerate(dependencies):
                if progress_callback:
                    progress = (i + 1) / len(dependencies)
                    progress_callback(f"Installing {dep}...", progress)

                logger.info(f"Installing {dep}")

                # Use pip to install
                run_command([
                    str(pip_path),
                    "install",
                    "--no-warn-script-location",
                    dep
                ])

            self._mark_installed(model_name)
            logger.info(f"Successfully installed dependencies for {model_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies for {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Failed to install {model_name}", 0)
            return False

    def _mark_installed(self, model_name: str):
        """Mark model as installed in registry.

        Args:
            model_name: Name of the model
        """
        self.installed_models[model_name] = {
            'installed': True,
            'name': model_name,
            'info': MODEL_REGISTRY.get(model_name, {})
        }
        self._save_registry()

    def uninstall_model(self, model_name: str) -> bool:
        """Uninstall model (remove from registry, dependencies remain).

        Args:
            model_name: Name of the model

        Returns:
            True if successful
        """
        if model_name in self.installed_models:
            del self.installed_models[model_name]
            self._save_registry()
            logger.info(f"Uninstalled {model_name}")
            return True
        return False

    def get_total_download_size(self, model_names: List[str]) -> int:
        """Calculate total download size for models.

        Args:
            model_names: List of model names

        Returns:
            Total size in MB
        """
        total_mb = 0
        for name in model_names:
            if name in MODEL_REGISTRY:
                total_mb += MODEL_REGISTRY[name].get('size_mb', 0)
        return total_mb

    def install_models_batch(
        self,
        model_names: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, bool]:
        """Install multiple models.

        Args:
            model_names: List of model names to install
            progress_callback: Optional callback(message, progress)

        Returns:
            Dictionary mapping model names to success status
        """
        results = {}
        total = len(model_names)

        for i, model_name in enumerate(model_names):
            overall_progress = i / total

            if progress_callback:
                progress_callback(f"Installing {model_name}...", overall_progress)

            def model_progress(msg, prog):
                if progress_callback:
                    # Combine overall and model-specific progress
                    combined = overall_progress + (prog / total)
                    progress_callback(msg, combined)

            success = self.install_model_dependencies(model_name, model_progress)
            results[model_name] = success

        if progress_callback:
            progress_callback("All models installed", 1.0)

        return results

    def get_installation_status(self) -> Dict[str, Dict]:
        """Get status of all models.

        Returns:
            Dictionary with model names and their installation status
        """
        status = {}
        for model_name in MODEL_REGISTRY:
            status[model_name] = {
                'installed': self.is_model_installed(model_name),
                'info': MODEL_REGISTRY[model_name]
            }
        return status
