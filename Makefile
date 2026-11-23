.PHONY: install test lint format check audit

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format --check .

check: lint test

audit:
	python scripts/inventory_dataset.py
