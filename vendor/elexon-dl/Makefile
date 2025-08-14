.PHONY: venv lint typecheck test build clean

VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

venv:
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e .[dev,parquet]

lint:
	$(VENV)/bin/ruff check .

typecheck:
	$(PY) -m mypy src

test:
	$(VENV)/bin/pytest

build:
	$(PY) -m build

clean:
	rm -rf dist .build *.egg-info .pytest_cache .mypy_cache .ruff_cache
