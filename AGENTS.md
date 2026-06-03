# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package for running Codex through a Telegram bot bridge.
Application code lives in `src/hermes_telegram/`:

- `cli.py` wires configuration, Telegram polling, Codex execution, and session storage.
- `config.py` parses `.env` settings and validates runtime paths.
- `gateway.py` handles Telegram commands, authorization, model selection, and message flow.
- `codex_runner.py`, `telegram_api.py`, `session_store.py`, and `model_catalog.py` isolate external process, API, state, and model-catalog behavior.

Tests live in `tests/` and mirror the module responsibilities. Deployment assets are under `deploy/systemd/`. Runtime files such as `.env` and `.hermes-telegram/` are local-only and should not be committed.

## Build, Test, and Development Commands

- `PYTHONPATH=src python3 -m hermes_telegram.cli --env-file .env`: run the bridge directly from the checkout.
- `python3 -m pip install -e .`: install the package in editable mode.
- `hermes-telegram --env-file .env`: run via the console script after editable install.
- `PYTHONPATH=src python3 -m unittest discover -s tests`: run the full test suite.
- `python3 -m build`: build a package artifact if the `build` module is installed.

## Coding Style & Naming Conventions

Use Python 3.10+ with 4-space indentation, `from __future__ import annotations`, and type hints for public helpers and dataclasses. Keep modules focused and dependency-light; this project currently has no runtime dependencies. Use `snake_case` for functions, variables, and module names, and `PascalCase` for classes and dataclasses. Prefer explicit small helpers over broad inline parsing in gateway or CLI code.

## Testing Guidelines

Tests use the standard `unittest` framework. Name files `tests/test_<module>.py` and test classes `<Feature>Tests`. Keep external systems mocked or isolated; tests should not call Telegram, run real Codex, or require a real `.env`. Add focused tests for config parsing, authorization, callback handling, session persistence, and command construction when changing those paths.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add Telegram model picker` and `Fix stale Codex model preferences`. Keep commits scoped to one behavior change and include tests or a clear reason tests were not run. Pull requests should describe the runtime impact, list validation commands, and mention any configuration or systemd changes. Include screenshots only for Telegram UI-visible command or button changes.

## Security & Configuration Tips

Never commit bot tokens, user IDs from private chats, `.env`, or `.hermes-telegram/` state. When editing systemd service files, keep absolute paths and `CODEX_COMMAND` behavior clear because user services may not inherit shell-managed paths.
