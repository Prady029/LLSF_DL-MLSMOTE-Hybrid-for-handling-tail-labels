.PHONY: help install install-dev test test-verbose clean lint format type-check all-checks setup-dev run-demo

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install package in current environment"
	@echo "  install-dev   - Install package with development dependencies"
	@echo "  setup-dev     - Set up development environment"
	@echo "  test          - Run test suite"
	@echo "  test-verbose  - Run test suite with verbose output"
	@echo "  lint          - Run linting checks (flake8)"
	@echo "  format        - Format code with black and isort"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  all-checks    - Run all quality checks (lint, format, type-check)"
	@echo "  run-demo      - Run demo evaluation"
	@echo "  clean         - Clean up build artifacts and cache files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

setup-dev:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo "Development environment ready! Activate with: source .venv/bin/activate"

# Testing
test:
	cd python_src && python evaluate.py --test

test-verbose:
	cd python_src && python evaluate.py --test

test-pytest:
	pytest tests/ -v

# Code quality
lint:
	flake8 python_src/ tests/

format:
	black python_src/ tests/
	isort python_src/ tests/

type-check:
	mypy python_src/

all-checks: lint type-check
	@echo "All quality checks completed"

# Demo and examples
run-demo:
	cd python_src && python evaluate.py --quick demo minority

run-demo-full:
	cd python_src && python evaluate.py --quick demo both

# Cleanup
clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Build and distribution
build:
	python -m build

upload-test:
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	python -m twine upload dist/*
