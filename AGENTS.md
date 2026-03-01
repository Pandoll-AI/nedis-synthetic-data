# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python source code, split into `core`, `analysis`, `vectorized`, `population`, `clinical`, `temporal`, `privacy`, `validation`, and `allocation` packages.
- `scripts/`: Runnable maintenance and workflow entry points (generation, setup, validation, comparison dashboards, and reporting).
- `tests/`: Automated test suite (current pytest-focused tests live here).
- `docs/`, `plan/`, `validator/`: design references, planning notes, and helper documentation.
- `config/`: runtime YAML config (`generation_params.yaml`).
- `templates/`: HTML templates for Flask report views.
- `outputs/`, `cache/`, `*.duckdb` are build/runtime artifacts and should not be treated as source inputs.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` – create an isolated environment.
- `pip install -r requirements.txt` – install dependencies.
- `python scripts/setup_database.py` – initialize and migrate a working project database.
- `python scripts/generate.py --help` – inspect generator options.
- `python scripts/generate.py --source-database nedis_data.duckdb --database nedis_synth_2018.duckdb --year 2018` – run the primary synthetic-generation pipeline.
- `python scripts/test_privacy_simple.py` – run the lightweight privacy smoke pipeline.
- `python scripts/test_enhanced_generation.py --db nedis_data.duckdb --test-configs` – multi-config privacy/generator test flow.
- `python scripts/compare_original_vs_synthetic.py --database nedis_synth_2018.duckdb --export-only` – render and export a comparison HTML report.
- `pytest tests/` – execute test suite; prefer `pytest tests/test_privacy_modules.py` while iterating.

## Coding Style & Naming Conventions
- Python with 4-space indentation and UTF-8 files.
- Naming: `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Use docstrings for public functions/classes; prefer type hints on public APIs.
- Keep generated artifacts and large in-memory tables out of function signatures unless necessary.
- Formatting/lint tools used by project dependencies: `black`, `flake8`, `mypy`.

## Testing Guidelines
- Framework: `pytest`.
- Test naming convention: files `test_*.py`, test functions `test_*`.
- Aim for deterministic tests when randomness is involved (`seed` or fixed inputs).
- Add/extend tests for each changed module, especially privacy/transformation logic.
- Include smoke-level validation in script-based checks when full pipelines are too heavy for CI.

## Commit & Pull Request Guidelines
- Recent history in `.git/logs/HEAD` shows conventional-style prefixes (`feat:`, `docs:`) and plain `commit:`.
- Use concise imperative subjects; prefer `type(scope): message` format (e.g., `feat(privacy): ...`).
- PR description should include:
  - change summary and rationale
  - touched files/commands run
  - validation output (`pytest ...`, `scripts/*` results)
  - impact on data/config files

## Security & Configuration Tips
- Run and test with synthetic or sample data where possible.
- Never commit database files, logs, caches, or generated outputs unless explicitly sanitized.
- Environment overrides supported in `src/core/config.py`: `NEDIS_DB_PATH`, `NEDIS_LOG_LEVEL`, `NEDIS_TARGET_RECORDS`.
