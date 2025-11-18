"""
Build script for creating Anonator Windows executable.

This script handles the complete build process using PyInstaller,
including cleanup and verification.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def clean_build_folders():
    """Remove previous build artifacts."""
    print("Cleaning previous build artifacts...")
    folders_to_remove = ['build', 'dist']

    for folder in folders_to_remove:
        folder_path = Path(folder)
        if folder_path.exists():
            print(f"  Removing {folder}/")
            shutil.rmtree(folder_path)

    print("Clean complete.")

def run_pyinstaller():
    """Run PyInstaller with the spec file."""
    print("\nBuilding executable with PyInstaller...")
    print("This may take 5-10 minutes...")
    print("-" * 80)

    result = subprocess.run(
        ['pyinstaller', 'anonator.spec', '--clean'],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print("\nERROR: PyInstaller build failed!")
        sys.exit(1)

    print("-" * 80)
    print("Build complete!")

def verify_executable():
    """Check if the executable was created successfully."""
    exe_path = Path('dist/Anonator.exe')

    if not exe_path.exists():
        print("\nERROR: Anonator.exe was not created!")
        return False

    exe_size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"\nExecutable created successfully!")
    print(f"  Location: {exe_path.absolute()}")
    print(f"  Size: {exe_size_mb:.1f} MB")

    return True

def main():
    print("=" * 80)
    print("ANONATOR - WINDOWS EXECUTABLE BUILD")
    print("=" * 80)
    print()

    # Step 1: Clean
    clean_build_folders()

    # Step 2: Build
    run_pyinstaller()

    # Step 3: Verify
    if verify_executable():
        print("\nBUILD SUCCESS!")
        print("You can now distribute dist/Anonator.exe")
    else:
        print("\nBUILD FAILED!")
        sys.exit(1)

if __name__ == '__main__':
    main()
