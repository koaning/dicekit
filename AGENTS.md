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
| Interactive notebook | `uvx marimo -y edit nbs/__init__.py --headless` |

There is no dedicated linter in this repo; CI runs **pytest** only.

### Marimo notebook server

Use the global **`-y` / `--yes`** flag for non-interactive startup (same pattern as `make build`, which runs `uvx marimo -y export ...`):

```bash
uvx marimo -y edit nbs/__init__.py --headless
```

`-y` auto-accepts the sandbox prompt and starts the server without tmux keystrokes or piping answers. Default URL: **http://localhost:2718** (token appended when `--token` is enabled).

**Note:** `uv run marimo -y edit ...` can fail here because the sandbox re-invokes the project marimo and hits a version conflict. Prefer **`uvx marimo -y edit`** instead.

### Static docs (optional)

Pre-built WASM docs are in `docs/`. Serve locally with `python -m http.server 8080 -d docs` for docs-only validation.

### Testing notes

- Automated verification is `uv run pytest` (15 tests in `tests/test_basics.py`).
- No external services or network calls are required for tests.
