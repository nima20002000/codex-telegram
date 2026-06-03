# Hermes Telegram

Standalone Telegram bridge for running Codex from a Telegram bot chat.

This repo extracts the useful Telegram gateway shape from Hermes into a small local service:

- reads a Telegram bot token from `.env`
- polls Telegram Bot API with `getUpdates`
- gates access by Telegram user IDs and/or chat IDs
- keeps compact per-chat local history under `.hermes-telegram/`
- runs `codex exec` in the configured local workspace
- replies to Telegram with the final Codex response

## Setup

```bash
cd $HOME/Desktop/hermes-telegram
cp .env.example .env
```

Edit `.env`:

```bash
TELEGRAM_BOT_TOKEN=123456:from-botfather
TELEGRAM_ALLOWED_USERS=123456789
CODEX_WORKDIR=$HOME/Desktop
HERMES_TELEGRAM_STATE_DIR=$HOME/Desktop/hermes-telegram/.hermes-telegram
```

Then run:

```bash
PYTHONPATH=src python3 -m hermes_telegram.cli --env-file .env
```

Or install the package in editable mode and use the console script:

```bash
python3 -m pip install -e .
hermes-telegram --env-file .env
```

## Run On Startup

Install the user service:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/hermes-telegram.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-telegram.service
loginctl enable-linger "$USER"
```

The committed unit assumes the checkout stays at `$HOME/Desktop/hermes-telegram`.
It reads the ignored `.env`, runs Codex in `CODEX_WORKDIR`, and restarts the bridge if
the process exits.

## Commands

- `/reset`
- `/models`
- `/workspace`
- `/sandbox`

Any other text message is sent to `codex exec`.

`/workspace` opens an inline folder browser rooted at `CODEX_WORKDIR`. Use folder
buttons to move into child directories, then press `Start session` to make later
messages run in that directory. The selected workspace is stored per Telegram
chat.

Model choices come from the local `codex debug models` catalog when available.
The selected model and thinking amount are stored per Telegram chat under the
bridge state directory and are applied to later agent messages from that chat,
including new workspace sessions after `/reset`.

`/sandbox` lets a chat choose between `Constrained` and `YOLO`. Constrained runs
Codex with `--sandbox workspace-write`. YOLO runs Codex with
`--dangerously-bypass-approvals-and-sandbox`, which disables approval prompts and
sandboxing. The selected sandbox mode is stored per Telegram chat and stays in
effect for later messages and sessions.

## Environment

Required:

- `TELEGRAM_BOT_TOKEN`: bot token from BotFather

Recommended:

- `TELEGRAM_ALLOWED_USERS`: comma-separated numeric Telegram user IDs
- `TELEGRAM_ALLOWED_CHATS`: optional comma-separated chat IDs
- `CODEX_WORKDIR`: directory where Codex should work

Codex options:

- `CODEX_COMMAND`: defaults to `codex`; use an absolute path when running under systemd if your shell gets Codex from `nvm`, `asdf`, or another shell-managed path
- `CODEX_MODEL`: optional model override
- `CODEX_PROFILE`: optional Codex config profile
- `CODEX_SANDBOX`: defaults to `workspace-write`
- `CODEX_EXTRA_ARGS`: extra shell-split args inserted after `codex exec`
- `CODEX_TIMEOUT_SECONDS`: defaults to `1800`

Telegram options:

- `TELEGRAM_POLL_TIMEOUT_SECONDS`: defaults to `30`
- `TELEGRAM_REQUEST_TIMEOUT_SECONDS`: defaults to `45`
- `MAX_TELEGRAM_RESPONSE_CHARS`: defaults to `12000`

Bridge options:

- `SESSION_HISTORY_TURNS`: defaults to `8`
- `HERMES_TELEGRAM_STATE_DIR`: defaults to `$CODEX_WORKDIR/.hermes-telegram`

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Notes

This bridge intentionally does not vendor the full `~/.hermes/hermes-agent` codebase. It extracts the Telegram-to-agent control path into a focused local repo. The agent side is Codex CLI, so the bot can work in whatever `CODEX_WORKDIR` points at.
