build:
	uv run python nbs/build.py
	uvx marimo export html-wasm --mode edit -o docs nbs/__init__.py --force

install: 
	python -m pip install uv
	python -m pip install pytest
	uv venv
	uv pip install -e . marimo

uv-install: 
	uv pip install -e . marimo

pypi:
	uv build
	uv publish

test:
	uv run pytest