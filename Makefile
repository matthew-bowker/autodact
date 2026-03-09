.PHONY: venv install run test clean build-mac build-win

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

run:
	$(PYTHON) -m src.main

test:
	$(PYTHON) -m pytest tests/ -v

build-mac:
	$(PYTHON) -m PyInstaller autodact_macos.spec

build-win:
	$(PYTHON) -m PyInstaller autodact_windows.spec

clean:
	rm -rf $(VENV) __pycache__ src/__pycache__ tests/__pycache__
	rm -rf dist/ build/
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
