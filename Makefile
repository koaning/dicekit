build:
	uv run mobuild export nbs/__init__.py dicekit/__init__.py
	uv run mobuild export nbs/learn.py dicekit/learn.py

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

typecheck:
	uv run basedpyright dicekit

clean:
	rm -rf dist nbs/__pycache__ dicekit/__pycache__
