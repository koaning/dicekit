# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

**dicekit** is a Python library for modeling dice and urn draws with arbitrary probability distributions. Development is notebook-driven: the canonical implementation lives in `nbs/__init__.py`, exported to `dicekit/__init__.py` via `nbs/build.py`. There is no web API, database, or Docker stack.

### Prerequisites

- **Python 3.10–3.12** (see `requires-python` in `pyproject.toml`)
- **uv** package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh` if not on `PATH`)

### Common commands

See `Makefile` and `.github/workflows/tests.yml` for the canonical workflow:

| Task | Command |
|------|---------|
| Install dev deps | `make install` |
| Run tests | `make test` |
| Rebuild lib + docs | `make build` |
| Interactive notebook | `uv run marimo edit nbs/__init__.py --headless` |

There is no dedicated linter in this repo; CI runs **pytest** only.

### Marimo notebook server

When starting marimo on `nbs/__init__.py`, it may prompt:

> Run in a sandboxed venv containing this notebook's dependencies? [Y/n]

Answer **`n`** when dependencies are already installed via `make install` (non-interactive: pipe `printf 'n\n'` or send `n` in tmux). Default URL: **http://localhost:2718** (token appended to URL when `--token` is enabled).

### Static docs (optional)

Pre-built WASM docs are in `docs/`. Serve locally with `python -m http.server 8080 -d docs` for docs-only validation.

### Testing notes

- Automated verification is `uv run pytest` (15 tests in `tests/test_basics.py`).
- No external services or network calls are required for tests.
