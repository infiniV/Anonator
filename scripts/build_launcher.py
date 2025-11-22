"""Build script for Anonator Launcher distribution."""

import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

# Paths
SCRIPTS_DIR = Path(__file__).parent
ROOT_DIR = SCRIPTS_DIR.parent  # Project root (parent of scripts/)
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"
LAUNCHER_DIR = ROOT_DIR / "launcher"
SRC_DIR = ROOT_DIR / "src"
OUTPUT_DIR = ROOT_DIR / "release"

# Output names
LAUNCHER_NAME = "AnonatorLauncher"
PACKAGE_NAME = "AnonatorSetup"


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous builds...")

    for directory in [DIST_DIR, BUILD_DIR, OUTPUT_DIR]:
        if directory.exists():
            try:
                shutil.rmtree(directory, ignore_errors=True)
                print(f"  Removed {directory}")
            except Exception as e:
                print(f"  Warning: Could not fully remove {directory}: {e}")

    print("Clean complete.\n")


def build_launcher():
    """Build launcher executable with PyInstaller."""
    print("Building launcher executable...")

    # Ensure pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "packaging"], check=True)

    # Run PyInstaller
    spec_file = SCRIPTS_DIR / "launcher.spec"
    subprocess.run([
        "pyinstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ], check=True)

    launcher_dist = DIST_DIR / LAUNCHER_NAME

    if not launcher_dist.exists():
        raise FileNotFoundError(f"Launcher build failed: {launcher_dist} not found")

    print(f"Launcher built successfully: {launcher_dist}\n")
    return launcher_dist


def package_app_source():
    """Package Anonator app source code."""
    print("Packaging app source code...")

    app_package_dir = OUTPUT_DIR / "anonator_app"
    app_package_dir.mkdir(parents=True, exist_ok=True)

    # Copy source files
    if SRC_DIR.exists():
        shutil.copytree(SRC_DIR, app_package_dir / "src", dirs_exist_ok=True)
        print(f"  Copied {SRC_DIR} -> {app_package_dir / 'src'}")
    else:
        print(f"  WARNING: Source directory not found: {SRC_DIR}")

    # Copy tests (optional, for validation)
    tests_dir = ROOT_DIR / "tests"
    if tests_dir.exists():
        shutil.copytree(tests_dir, app_package_dir / "tests", dirs_exist_ok=True)
        print(f"  Copied {tests_dir} -> {app_package_dir / 'tests'}")

    # Copy pyproject.toml
    pyproject = ROOT_DIR / "pyproject.toml"
    if pyproject.exists():
        shutil.copy(pyproject, app_package_dir / "pyproject.toml")
        print(f"  Copied {pyproject}")

    print(f"App source packaged to {app_package_dir}\n")
    return app_package_dir


def create_distribution():
    """Create final distribution package."""
    print("Creating distribution package...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build launcher
    launcher_dist = build_launcher()

    # 2. Package app source
    app_package = package_app_source()

    # 3. Copy launcher to output
    launcher_output = OUTPUT_DIR / LAUNCHER_NAME
    if launcher_output.exists():
        shutil.rmtree(launcher_output)

    shutil.copytree(launcher_dist, launcher_output)
    print(f"Copied launcher to {launcher_output}")

    # 4. Create final zip package
    final_zip = OUTPUT_DIR / f"{PACKAGE_NAME}.zip"

    if final_zip.exists():
        final_zip.unlink()

    print(f"Creating distribution archive: {final_zip}")

    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add launcher executable and DLLs
        for file in launcher_output.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(OUTPUT_DIR)
                zf.write(file, arcname)
                print(f"  Added {arcname}")

        # Add app source
        for file in app_package.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(OUTPUT_DIR)
                zf.write(file, arcname)
                print(f"  Added {arcname}")

        # Add README
        create_readme(OUTPUT_DIR / "README.txt")
        zf.write(OUTPUT_DIR / "README.txt", "README.txt")
        print("  Added README.txt")

    print(f"\nDistribution package created: {final_zip}")

    # Print size
    size_mb = final_zip.stat().st_size / (1024 * 1024)
    print(f"Package size: {size_mb:.1f} MB")

    # 5. Also create standalone launcher-only zip (for updates)
    launcher_only_zip = OUTPUT_DIR / f"{LAUNCHER_NAME}_Only.zip"

    with zipfile.ZipFile(launcher_only_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in launcher_output.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(launcher_output)
                zf.write(file, arcname)

    print(f"Launcher-only package: {launcher_only_zip}")
    size_mb = launcher_only_zip.stat().st_size / (1024 * 1024)
    print(f"Launcher size: {size_mb:.1f} MB")


def create_readme(path: Path):
    """Create README file for distribution."""
    readme_content = """
# Anonator Setup

Thank you for downloading Anonator!

## Installation

1. Extract this ZIP file to a folder
2. Run `AnonatorLauncher.exe` (Windows) or `AnonatorLauncher` (Linux/macOS)
3. Follow the on-screen setup wizard:
   - Select which face detection models to install
   - Wait for installation to complete (5-10 minutes)
4. Click "Launch Anonator" when ready

## First-Time Setup

The launcher will:
- Download UV package manager (~10MB)
- Install Python 3.11.9 automatically via UV
- Create a virtual environment
- Install dependencies (~500MB-1GB)
- Download selected face detection models

All files are installed to:
- Windows: %APPDATA%\\Anonator
- Linux/macOS: ~/.anonator

## Models

The following face detection models are available:
- MediaPipe (20MB) - Fast, CPU-friendly (recommended)
- MTCNN (2MB) - Accurate, moderate speed
- SCRFD-2.5GF (10MB) - Balanced GPU model
- SCRFD-10GF (50MB) - High accuracy GPU model
- YOLOv8-Face (6MB) - Fast GPU detection
- YOLO11-Face (10MB) - Latest YOLO model
- RetinaFace-MobileNet (27MB) - Mobile-optimized

You can install additional models later from within the application.

## GPU Support

If you have an NVIDIA GPU with CUDA support, the launcher will
automatically install GPU-accelerated PyTorch for faster processing.

## Troubleshooting

If installation fails:
1. Check the installation log in the launcher window
2. Ensure you have internet connection
3. Check antivirus isn't blocking downloads
4. Try running as administrator (Windows)

For more help, visit: https://github.com/YOUR_USERNAME/anonator

## Uninstallation

To uninstall:
1. Delete the installation directory (%APPDATA%\\Anonator or ~/.anonator)
2. Delete the launcher folder

---

Anonator v1.0.0
GPU-accelerated video face anonymization
"""

    with open(path, 'w') as f:
        f.write(readme_content)


def main():
    """Main build process."""
    print("=" * 60)
    print("Anonator Launcher Build Script")
    print("=" * 60)
    print()

    # Clean previous builds
    clean_build()

    # Create distribution
    create_distribution()

    print()
    print("=" * 60)
    print("Build complete!")
    print("=" * 60)
    print()
    print(f"Distribution package: {OUTPUT_DIR / f'{PACKAGE_NAME}.zip'}")
    print(f"Launcher only: {OUTPUT_DIR / f'{LAUNCHER_NAME}_Only.zip'}")
    print()
    print("Ready for distribution!")


if __name__ == "__main__":
    main()
