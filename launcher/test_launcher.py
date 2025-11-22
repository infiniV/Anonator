"""Test script for Anonator launcher (development mode)."""

import sys
from pathlib import Path

# Add launcher to path
launcher_dir = Path(__file__).parent / "launcher"
sys.path.insert(0, str(launcher_dir.parent))

def test_imports():
    """Test that all launcher imports work."""
    print("Testing imports...")

    try:
        from launcher import config
        print("[OK] Config module OK")
    except ImportError as e:
        print(f"[FAIL] Config module FAILED: {e}")
        return False

    try:
        from launcher import utils
        print("[OK] Utils module OK")
    except ImportError as e:
        print(f"[FAIL] Utils module FAILED: {e}")
        return False

    try:
        from launcher import model_manager
        print("[OK] Model manager OK")
    except ImportError as e:
        print(f"[FAIL] Model manager FAILED: {e}")
        return False

    try:
        from launcher import updater
        print("[OK] Updater module OK")
    except ImportError as e:
        print(f"[FAIL] Updater module FAILED: {e}")
        return False

    try:
        from launcher import launcher
        print("[OK] Launcher module OK")
    except ImportError as e:
        print(f"[FAIL] Launcher module FAILED: {e}")
        return False

    return True


def test_model_manager():
    """Test model manager functionality."""
    print("\nTesting model manager...")

    from launcher.model_manager import ModelManager
    from launcher.config import MODELS_DIR

    manager = ModelManager(MODELS_DIR)

    # Get available models
    models = manager.get_available_models()
    print(f"[OK] Found {len(models)} models: {', '.join(models)}")

    # Get default models
    defaults = manager.get_default_models()
    print(f"[OK] Default models: {', '.join(defaults)}")

    # Get installation status
    status = manager.get_installation_status()
    print(f"[OK] Installation status retrieved for {len(status)} models")

    return True


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")

    from launcher.utils import format_size, detect_cuda

    # Test size formatting
    sizes = [1024, 1024*1024, 1024*1024*1024]
    for size in sizes:
        formatted = format_size(size)
        print(f"[OK] {size} bytes = {formatted}")

    # Test GPU detection
    has_gpu = detect_cuda()
    print(f"[OK] GPU detected: {has_gpu}")

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    from launcher.config import (
        VERSION, APP_NAME, APP_DATA_DIR, MODEL_REGISTRY,
        CORE_DEPENDENCIES, PYTHON_VERSION
    )

    print(f"[OK] App name: {APP_NAME}")
    print(f"[OK] Version: {VERSION}")
    print(f"[OK] Python version: {PYTHON_VERSION}")
    print(f"[OK] App data dir: {APP_DATA_DIR}")
    print(f"[OK] Models: {len(MODEL_REGISTRY)} available")
    print(f"[OK] Core dependencies: {len(CORE_DEPENDENCIES)} packages")

    return True


def run_launcher():
    """Run the launcher GUI."""
    print("\nLaunching GUI...")

    try:
        from launcher import main
        main()
    except Exception as e:
        print(f"[FAIL] Failed to launch: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Main test runner."""
    print("=" * 60)
    print("Anonator Launcher Test Suite")
    print("=" * 60)
    print()

    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Utilities", test_utils),
        ("Model Manager", test_model_manager),
    ]

    failed = []

    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print('='*60)

        try:
            success = test_func()
            if not success:
                failed.append(name)
        except Exception as e:
            print(f"[FAIL] {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    if failed:
        print(f"\n[FAIL] {len(failed)} test(s) FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nPlease fix errors before building.")
        return 1
    else:
        print("\n[PASS] All tests PASSED!")
        print("\nLauncher is ready to build or run.")

        # Ask to run launcher
        response = input("\nRun launcher GUI now? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            run_launcher()

        return 0


if __name__ == "__main__":
    sys.exit(main())
