# Makefile for Anonator Face Detection & Anonymization System

# Project configuration
PROJECT_NAME = Anonator
PYTHON = python
VENV_PYTHON = .venv/Scripts/python.exe
SRC_DIR = src/anonator
TESTS_DIR = tests
BUILD_DIR = build
DIST_DIR = dist
RELEASE_DIR = release

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
	RM = cmd /c del /q
	RMDIR = cmd /c rmdir /s /q
	MKDIR = cmd /c mkdir
	SEP = \\
else
	RM = rm -f
	RMDIR = rm -rf
	MKDIR = mkdir -p
	SEP = /
endif

# ANSI color codes (for Unix-like terminals)
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m

.PHONY: help
help: ## Show this help message
	@echo "$(GREEN)$(PROJECT_NAME) - Makefile Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

.PHONY: install
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	uv pip install --upgrade pip
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	uv pip install --no-build-isolation git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
	uv pip install -e .
	@echo "$(GREEN)Production dependencies installed!$(NC)"

.PHONY: install-dev
install-dev: ## Install development dependencies (includes testing tools)
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	uv pip install --upgrade pip
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	uv pip install --no-build-isolation git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
	uv pip install -e .
	uv pip install pytest pytest-cov pytest-mock pytest-benchmark pytest-timeout pytest-xdist hypothesis
	@echo "$(GREEN)Development dependencies installed!$(NC)"

.PHONY: install-all
install-all: install install-dev ## Install all dependencies (production + development)
	@echo "$(GREEN)All dependencies installed!$(NC)"

.PHONY: test
test: ## Run all tests (unit + integration)
	@echo "$(GREEN)Running tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/ -v
	@echo "$(GREEN)Tests completed!$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only (fast)
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(NC)"

.PHONY: test-benchmark
test-benchmark: ## Run benchmark tests (requires pytest-benchmark)
	@echo "$(GREEN)Running benchmark tests...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/benchmarks/ --benchmark-only
	@echo "$(GREEN)Benchmark tests completed!$(NC)"

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

.PHONY: test-fast
test-fast: ## Run tests in parallel (faster)
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	$(VENV_PYTHON) -m pytest $(TESTS_DIR)/ -n auto -v
	@echo "$(GREEN)Tests completed!$(NC)"

.PHONY: test-data
test-data: ## Prepare test data (extract frames from videos)
	@echo "$(GREEN)Preparing test data...$(NC)"
	$(VENV_PYTHON) $(TESTS_DIR)/prepare_test_data.py
	@echo "$(GREEN)Test data prepared!$(NC)"

.PHONY: run
run: ## Run the application UI
	@echo "$(GREEN)Starting $(PROJECT_NAME) UI...$(NC)"
	$(VENV_PYTHON) $(SRC_DIR)/main.py

.PHONY: build
build: build-launcher ## Build progressive launcher distribution
	@echo "$(GREEN)Build complete!$(NC)"
	@echo "$(YELLOW)Output: release/AnonatorSetup.zip (26 MB)$(NC)"
	@echo "$(YELLOW)Users download only 26MB, then install what they need$(NC)"

.PHONY: build-launcher
build-launcher: ## Build progressive launcher (same as 'make build')
	@echo "$(GREEN)Building progressive launcher...$(NC)"
	@echo "$(YELLOW)Output will be 26 MB (includes launcher + app source)$(NC)"
	$(PYTHON) scripts/build_launcher.py
	@echo "$(GREEN)Launcher built! Output: release/AnonatorSetup.zip$(NC)"

.PHONY: test-launcher
test-launcher: ## Test launcher system
	@echo "$(GREEN)Testing launcher...$(NC)"
	$(PYTHON) launcher/test_launcher.py
	@echo "$(GREEN)Launcher tests completed!$(NC)"

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-release ## Clean all build, test, and Python artifacts
	@echo "$(GREEN)Cleaned all artifacts!$(NC)"

.PHONY: clean-all
clean-all: clean ## Alias for clean (remove all generated files)
	@echo "$(GREEN)Complete cleanup done!$(NC)"

.PHONY: clean-build
clean-build: ## Remove build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('$(BUILD_DIR)'), pathlib.Path('$(DIST_DIR)')]]"
	@echo "$(GREEN)Build artifacts cleaned!$(NC)"

.PHONY: clean-release
clean-release: ## Remove release and installer outputs
	@echo "$(YELLOW)Cleaning release artifacts...$(NC)"
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('release'), pathlib.Path('installer_output')]]"
	@echo "$(GREEN)Release artifacts cleaned!$(NC)"

.PHONY: clean-pyc
clean-pyc: ## Remove Python file artifacts
	@echo "$(YELLOW)Cleaning Python artifacts...$(NC)"
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyo')]"
	@echo "$(GREEN)Python artifacts cleaned!$(NC)"

.PHONY: clean-test
clean-test: ## Remove test and coverage artifacts
	@echo "$(YELLOW)Cleaning test artifacts...$(NC)"
	@$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [pathlib.Path('.pytest_cache'), pathlib.Path('htmlcov'), pathlib.Path('.benchmarks')]]; pathlib.Path('.coverage').unlink(missing_ok=True)"
	@echo "$(GREEN)Test artifacts cleaned!$(NC)"

.PHONY: lint
lint: ## Check code style (requires ruff or flake8)
	@echo "$(GREEN)Checking code style...$(NC)"
	@$(VENV_PYTHON) -m ruff check $(SRC_DIR) || echo "$(YELLOW)Ruff not installed. Install with: pip install ruff$(NC)"

.PHONY: format
format: ## Format code with black (requires black)
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(VENV_PYTHON) -m black $(SRC_DIR) $(TESTS_DIR) || echo "$(YELLOW)Black not installed. Install with: pip install black$(NC)"

.PHONY: check
check: test lint ## Run tests and linting
	@echo "$(GREEN)All checks passed!$(NC)"

.PHONY: dev
dev: install-dev test-data ## Setup complete development environment
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "$(YELLOW)Run 'make test' to verify setup$(NC)"

.PHONY: info
info: ## Show project information
	@echo "$(GREEN)Project Information:$(NC)"
	@echo "  Name:         $(PROJECT_NAME)"
	@echo "  Python:       $(VENV_PYTHON)"
	@echo "  Source:       $(SRC_DIR)"
	@echo "  Tests:        $(TESTS_DIR)"
	@echo "  Build Dir:    $(BUILD_DIR)"
	@echo "  Dist Dir:     $(DIST_DIR)"
	@echo ""
	@echo "$(YELLOW)Python Version:$(NC)"
	@$(VENV_PYTHON) --version
	@echo ""
	@echo "$(YELLOW)Installed Packages:$(NC)"
	@$(VENV_PYTHON) -m pip list | findstr "pytest torch face-detection opencv"

.PHONY: all
all: clean install-dev test build ## Clean, install, test, and build progressive launcher
	@echo "$(GREEN)Complete build pipeline finished!$(NC)"
	@echo "$(YELLOW)Distribution ready: release/AnonatorSetup.zip$(NC)"

# Default target
.DEFAULT_GOAL := help
