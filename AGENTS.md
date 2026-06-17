# Agent Instructions

## Source Of Truth

- `nbs/__init__.py` is the source notebook for exported package code.
- `dicekit/__init__.py` is generated from exported cells in `nbs/__init__.py`.
- Do not hand-edit only `dicekit/__init__.py` for package behavior changes. Update `nbs/__init__.py`, then run `make build`.

## Validation

- Run `make build` after changing exported notebook code.
- Run `uv run pytest -q` before publishing changes.
- Run `uv build` for release-prep changes.

## Release Prep

- Check the latest published version on PyPI before bumping `pyproject.toml`.
- Keep `uv.lock` in sync with `pyproject.toml` by running `uv lock`.
- Use the next available patch version for compatible fixes and performance improvements.
