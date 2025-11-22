"""Utility functions for downloads, verification, and system checks."""

import urllib.request
import urllib.error
import hashlib
import subprocess
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


def download_file(
    url: str,
    destination: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 8192
) -> Path:
    """Download a file with progress reporting.

    Args:
        url: URL to download from
        destination: Path to save file
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        chunk_size: Size of chunks to download

    Returns:
        Path to downloaded file

    Raises:
        urllib.error.URLError: If download fails
    """
    logger.info(f"Downloading {url} to {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(destination, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress_callback(downloaded, total_size)

        logger.info(f"Download complete: {destination}")
        return destination

    except urllib.error.URLError as e:
        logger.error(f"Download failed: {e}")
        if destination.exists():
            destination.unlink()
        raise


def verify_checksum(file_path: Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify file checksum.

    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        True if checksum matches
    """
    logger.info(f"Verifying {file_path} with {algorithm}")

    hash_obj = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_obj.update(chunk)

    actual_hash = hash_obj.hexdigest()
    matches = actual_hash == expected_hash

    if not matches:
        logger.warning(f"Checksum mismatch! Expected: {expected_hash}, Got: {actual_hash}")

    return matches


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract zip or tar archive.

    Args:
        archive_path: Path to archive file
        destination: Directory to extract to

    Returns:
        Path to extraction directory
    """
    logger.info(f"Extracting {archive_path} to {destination}")

    destination.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    logger.info("Extraction complete")
    return destination


def detect_cuda() -> bool:
    """Detect if NVIDIA CUDA is available.

    Returns:
        True if CUDA GPU is detected
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
            text=True
        )

        if result.returncode == 0:
            logger.info("CUDA GPU detected")
            return True
        else:
            logger.info("No CUDA GPU detected")
            return False

    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.info("nvidia-smi not found, assuming no GPU")
        return False


def get_pip_path(venv_dir: Path) -> Path:
    """Get path to pip executable in virtual environment.

    Args:
        venv_dir: Path to virtual environment

    Returns:
        Path to pip executable
    """
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "pip.exe"
    else:
        return venv_dir / "bin" / "pip"


def get_python_path(python_dir: Path) -> Path:
    """Get path to Python executable.

    Args:
        python_dir: Path to Python installation

    Returns:
        Path to Python executable
    """
    if sys.platform == "win32":
        return python_dir / "python.exe"
    else:
        return python_dir / "bin" / "python"


def get_venv_python_path(venv_dir: Path) -> Path:
    """Get path to Python executable in virtual environment.

    Args:
        venv_dir: Path to virtual environment

    Returns:
        Path to Python executable
    """
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def run_command(
    cmd: list,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return result.

    Args:
        cmd: Command and arguments as list
        cwd: Working directory
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess result

    Raises:
        subprocess.CalledProcessError: If command fails
    """
    logger.info(f"Running command: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
        check=True
    )

    return result


def get_installed_size(path: Path) -> int:
    """Get total size of directory in bytes.

    Args:
        path: Directory path

    Returns:
        Size in bytes
    """
    if not path.exists():
        return 0

    total_size = 0
    for item in path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size

    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def cleanup_directory(path: Path, confirm: bool = True) -> bool:
    """Remove directory and all contents.

    Args:
        path: Directory to remove
        confirm: Whether directory must exist

    Returns:
        True if removed, False if didn't exist
    """
    if not path.exists():
        return False

    logger.info(f"Removing directory: {path}")
    shutil.rmtree(path)
    return True


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
