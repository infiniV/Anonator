# Makefile Commands Reference

Quick reference for all available Makefile commands in Anonator.

## Cleaning Commands

### `make clean`
**Clean all build, test, and release artifacts**

Runs: `clean-build`, `clean-pyc`, `clean-test`, `clean-release`

```bash
make clean
```

**What it removes:**
- `build/` - PyInstaller build files
- `dist/` - PyInstaller output
- `release/` - Distribution packages
- `installer_output/` - Installer files
- `__pycache__/` - Python cache directories
- `*.pyc`, `*.pyo` - Compiled Python files
- `.pytest_cache/` - Pytest cache
- `.coverage` - Coverage data
- `htmlcov/` - HTML coverage reports
- `.benchmarks/` - Benchmark data

### `make clean-all`
**Alias for `make clean`**

```bash
make clean-all
```

### `make clean-build`
**Remove build artifacts only**

```bash
make clean-build
```

Removes:
- `build/`
- `dist/`

### `make clean-release`
**Remove release and installer outputs**

```bash
make clean-release
```

Removes:
- `release/`
- `installer_output/`

### `make clean-pyc`
**Remove Python file artifacts**

```bash
make clean-pyc
```

Removes:
- `__pycache__/` directories
- `*.pyc`, `*.pyo` files
- Backup files (`*~`)

### `make clean-test`
**Remove test and coverage artifacts**

```bash
make clean-test
```

Removes:
- `.pytest_cache/`
- `.coverage`
- `htmlcov/`
- `.benchmarks/`

## Build Commands

### `make build`
**Build main application executable**

```bash
make build
```

Output: `dist/Anonator/`

Automatically runs `clean-build` first.

### `make build-launcher`
**Build progressive launcher distribution**

```bash
make build-launcher
```

Output:
- `release/AnonatorSetup.zip` (26 MB)
- `release/AnonatorLauncher_Only.zip` (17 MB)

### `make build-onefile`
**Build single-file executable**

```bash
make build-onefile
```

Output: `dist/Anonator.exe`

### `make build-debug`
**Build executable with console for debugging**

```bash
make build-debug
```

Output: `dist/Anonator/` (with console window)

## Testing Commands

### `make test`
**Run all tests**

```bash
make test
```

### `make test-launcher`
**Test launcher system**

```bash
make test-launcher
```

### `make test-unit`
**Run unit tests only**

```bash
make test-unit
```

### `make test-integration`
**Run integration tests**

```bash
make test-integration
```

### `make test-benchmark`
**Run benchmark tests**

```bash
make test-benchmark
```

### `make test-cov`
**Run tests with coverage report**

```bash
make test-cov
```

Output: `htmlcov/index.html`

### `make test-fast`
**Run tests in parallel**

```bash
make test-fast
```

Uses `pytest -n auto` for parallel execution.

### `make test-data`
**Prepare test data**

```bash
make test-data
```

Extracts frames from test videos.

## Development Commands

### `make install`
**Install production dependencies**

```bash
make install
```

Installs:
- PyTorch (CUDA)
- Core dependencies
- Production packages

### `make install-dev`
**Install development dependencies**

```bash
make install-dev
```

Installs everything from `install` plus:
- pytest and plugins
- Testing tools
- Development utilities

### `make install-all`
**Install all dependencies**

```bash
make install-all
```

Equivalent to `install` + `install-dev`.

### `make dev`
**Setup complete development environment**

```bash
make dev
```

Runs: `install-dev` + `test-data`

## Running Commands

### `make run`
**Run the application UI**

```bash
make run
```

Launches `src/anonator/main.py`

## Code Quality Commands

### `make lint`
**Check code style**

```bash
make lint
```

Requires: `ruff` or `flake8`

### `make format`
**Format code**

```bash
make format
```

Requires: `black`

### `make check`
**Run tests and linting**

```bash
make check
```

Runs: `test` + `lint`

## Information Commands

### `make info`
**Show project information**

```bash
make info
```

Displays:
- Project name and paths
- Python version
- Installed packages

### `make help`
**Show help message**

```bash
make help
```

Lists all available commands with descriptions.

## Complete Pipeline Commands

### `make all`
**Clean, install, test, and build everything**

```bash
make all
```

Runs: `clean` + `install-dev` + `test` + `build`

Complete build pipeline for release.

## Common Workflows

### Starting Development
```bash
make dev
make test
```

### Before Committing
```bash
make clean
make check
```

### Building Release
```bash
make clean
make build-launcher
make build
```

### Full Rebuild
```bash
make clean-all
make install-dev
make test
make build
```

### Quick Test
```bash
make test-fast
```

## Environment Variables

Makefile uses these variables:
- `PROJECT_NAME` - Anonator
- `PYTHON` - python
- `VENV_PYTHON` - .venv/Scripts/python.exe
- `SRC_DIR` - src/anonator
- `TESTS_DIR` - tests
- `BUILD_DIR` - build
- `DIST_DIR` - dist

## Platform Support

Commands work on:
- ✅ Windows (primary)
- ✅ Linux (via WSL or native)
- ✅ macOS (with adjustments)

The Makefile detects OS and adjusts commands accordingly.

## Tips

### Speed up builds
```bash
# Clean only what's needed
make clean-build

# Run tests in parallel
make test-fast
```

### Check what will be cleaned
```bash
# List files before cleaning
ls -la build/ dist/ release/

# Then clean
make clean
```

### Combine commands
```bash
# Clean and build
make clean build

# Test and lint
make check
```

### Silent mode
```bash
# Suppress output (for scripts)
make clean > /dev/null 2>&1
```

## Default Target

Running `make` without arguments shows help:

```bash
make
# Same as: make help
```

## Summary

Most used commands:

```bash
make clean          # Clean everything
make build-launcher # Build launcher
make test           # Run tests
make dev            # Setup dev environment
make help           # Show all commands
```

For complete command list:
```bash
make help
```
