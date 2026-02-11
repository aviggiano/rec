PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev lint format format-check typecheck test

install:
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e ".[dev]"

lint:
	ruff check src tests

format:
	black src tests
	ruff check src tests --fix

format-check:
	black --check src tests

typecheck:
	mypy src

test:
	pytest
