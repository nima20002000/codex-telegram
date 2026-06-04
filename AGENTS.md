# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package for running Codex through a Telegram bot bridge.
Application code lives in `src/codex_telegram/`:

- `cli.py` wires configuration, Telegram polling, Codex execution, and session storage.
- `config.py` parses `.env` settings and validates runtime paths.
- `gateway.py` handles Telegram commands, authorization, model selection, and message flow.
- `codex_runner.py`, `telegram_api.py`, `session_store.py`, and `model_catalog.py` isolate external process, API, state, and model-catalog behavior.

Tests live in `tests/` and mirror the module responsibilities. Deployment assets are under `deploy/systemd/`. Runtime files such as `.env` and `.codex-telegram/` are local-only and should not be committed.

## Build, Test, and Development Commands

- `PYTHONPATH=src python3 -m codex_telegram.cli --env-file .env`: run the bridge directly from the checkout.
- `python3 -m pip install -e .`: install the package in editable mode.
- `codex-telegram --env-file .env`: run via the console script after editable install.
- `PYTHONPATH=src python3 -m unittest discover -s tests`: run the full test suite.
- `python3 -m build`: build a package artifact if the `build` module is installed.

## Coding Style & Naming Conventions

Use Python 3.10+ with 4-space indentation, `from __future__ import annotations`, and type hints for public helpers and dataclasses. Keep modules focused and dependency-light; this project currently has no runtime dependencies. Use `snake_case` for functions, variables, and module names, and `PascalCase` for classes and dataclasses. Prefer explicit small helpers over broad inline parsing in gateway or CLI code.

## Testing Guidelines

Tests use the standard `unittest` framework. Name files `tests/test_<module>.py` and test classes `<Feature>Tests`. Keep external systems mocked or isolated; tests should not call Telegram, run real Codex, or require a real `.env`. Add focused tests for config parsing, authorization, callback handling, session persistence, and command construction when changing those paths.

## Manual Telegram E2E Testing

Manual Telegram E2E checks are opt-in and use local-only ignored state. Do not
commit the venv, Telegram session files, `.env`, `telegram-cred.md`, or anything
under `.codex-telegram/`.

The local admin MTProto session is stored at
`.codex-telegram/e2e/admin-account.session`. If that file exists, reuse it; do
not start a new Telegram login flow. A reusable setup is:

```bash
python3 -m venv /tmp/codex-telegram-e2e-venv
/tmp/codex-telegram-e2e-venv/bin/python -m pip install --upgrade pip telethon
mkdir -p .codex-telegram/e2e
chmod 700 .codex-telegram .codex-telegram/e2e
```

Read `App api_id` and `App api_hash` from the ignored `telegram-cred.md`, then
open Telethon with session path `.codex-telegram/e2e/admin-account`. Check
authorization with `await client.is_user_authorized()` before doing anything
else. If it returns `False`, stop and ask the user for the login code; do not
guess or print sensitive values.

For the current private forum-group E2E, verify the live bot first with Bot API
`getMe` using the ignored installed `.env` token, then use the admin MTProto
session to send a harmless marker message in the group and poll for the bot's
reply. Keep messages explicitly non-mutating, for example: ask the bot to reply
with a unique marker and not modify files.

Use the committed preflight harness when possible instead of rewriting one-off
Telethon scripts:

```bash
/tmp/codex-telegram-e2e-venv/bin/python scripts/telegram-e2e-preflight.py \
  --env-file $HOME/.local/share/codex-telegram/.env \
  --credentials telegram-cred.md \
  --session .codex-telegram/e2e/admin-account \
  --group "Codex Telegram E2E" \
  --expected-service-workdir $HOME/.local/share/codex-telegram \
  --expected-service-branch feature/add-feature \
  --marker
```

The harness uses Bot API `getMe` only; it never calls `getUpdates`. It redacts
tokens, API hashes, user IDs, and private chat IDs from output. For release
checks, also pass `--expected-service-commit <short-sha>` when the service must
be proven to run a specific checkout. If the service is intentionally stopped
because a foreground bridge is running from this checkout, pass
`--skip-service-check` and record that choice in the Linear evidence.

## Commit & Pull Request Guidelines

Recent commits use short imperative subjects, for example `Add Telegram model picker` and `Fix stale Codex model preferences`. Keep commits scoped to one behavior change and include tests or a clear reason tests were not run. Pull requests should describe the runtime impact, list validation commands, and mention any configuration or systemd changes. Include screenshots only for Telegram UI-visible command or button changes.

## Security & Configuration Tips

Never commit bot tokens, user IDs from private chats, `.env`, or `.codex-telegram/` state. When editing systemd service files, keep absolute paths and `CODEX_COMMAND` behavior clear because user services may not inherit shell-managed paths.
