# Makefile Usage

## Quick Start

```bash
make help          # Show all available commands
make test          # Run tests
make build         # Build executable
make run           # Run UI
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install production dependencies |
| `make install-dev` | Install dev dependencies + testing tools |
| `make test` | Run all tests (unit + integration) |
| `make test-unit` | Run unit tests only (fast, ~10s) |
| `make test-integration` | Run integration tests (~40s) |
| `make test-cov` | Run tests with coverage report |
| `make test-fast` | Run tests in parallel (faster) |
| `make test-data` | Prepare test data from videos |
| `make run` | Start the application UI |
| `make build` | Build executable with PyInstaller |
| `make clean` | Clean all build/test artifacts |
| `make dev` | Setup complete dev environment |
| `make info` | Show project information |
| `make all` | Clean, install, test, and build |

## Common Workflows

**First time setup:**
```bash
make dev           # Install deps + prepare test data
make test          # Verify everything works
```

**Daily development:**
```bash
make test-unit     # Fast tests while coding
make test          # Full tests before commit
```

**Build release:**
```bash
make clean
make test
make build
```

**Run application:**
```bash
make run
```

## Requirements

- GNU Make (Git Bash includes it)
- Python 3.9+ with venv at `.venv`
- CUDA 12 (optional, for GPU acceleration)

## Notes

- All commands use `.venv/Scripts/python.exe`
- Tests output to console with `-v` flag
- Coverage reports go to `htmlcov/index.html`
- Built executables go to `dist/Anonator/`
