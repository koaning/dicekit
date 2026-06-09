build:
	uv run python nbs/build.py

install:
	uv venv
	uv pip install -e . marimo pytest

uv-install:
	uv pip install -e . marimo

pypi: clean test
	uv build
	uv publish

test:
	uv run pytest

clean:
	rm -rf dist nbs/__pycache__ dicekit/__pycache__
