# Codex Telegram

MIT-licensed local Telegram bridge for running Codex from a bot chat.

This tool has only been tested on Ubuntu with the Codex CLI.

This repo provides a small local Telegram service for driving Codex:

- reads a Telegram bot token from `.env`
- polls Telegram Bot API with `getUpdates`
- gates access by Telegram user IDs and/or chat IDs
- keeps compact per-chat local history under `.codex-telegram/`
- runs `codex exec` in the configured local workspace
- replies to Telegram with the final Codex response

## Requirements

- Ubuntu with systemd user services
- Python 3.10+
- Codex CLI available to the service through `CODEX_COMMAND`
- Telegram bot token from BotFather: https://t.me/BotFather
- Numeric Telegram user ID from userinfobot: https://t.me/userinfobot

## Install

Before running the installer:

1. Open BotFather at https://t.me/BotFather, create or select a bot, and copy
   the bot API token.
2. Open userinfobot at https://t.me/userinfobot and copy your numeric Telegram
   user ID.
3. Run the installer. If `.env` already contains those values, the installer
   keeps them and does not ask again.

```bash
cd /home/example/codex-telegram
bash scripts/install.sh
```

The installer:

- installs the package with `python3 -m pip install -e .`
- falls back to an ignored repo-local `.venv` if system `pip` is unavailable or refuses the editable install
- creates `.env` from `.env.example` only when `.env` is missing
- preserves existing `.env` values, including bot token, allowed users, Codex command, sandbox, and workdir
- resolves `codex` to an absolute `CODEX_COMMAND` path for newly created env files
- prompts only for missing `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS`
- installs and restarts the user service at `~/.config/systemd/user/codex-telegram.service`
- enables linger so the user service can run after reboot without an interactive login

For a one-off foreground run without systemd:

```bash
PYTHONPATH=src python3 -m codex_telegram.cli --env-file .env
```

Or after installation:

```bash
codex-telegram --env-file .env
```

## Verify

Check service state and recent logs:

```bash
systemctl --user status codex-telegram.service --no-pager
journalctl --user -u codex-telegram.service --since '2 minutes ago' --no-pager
```

From the allowed Telegram user, send `/status`, `/models`, `/workspace`, and
`/sandbox`. Replies should arrive and logs should not show authorization or
runtime errors.

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

## Disable Or Uninstall

Disable the service:

```bash
systemctl --user disable --now codex-telegram.service
```

Remove the package installation from the active Python environment:

```bash
python3 -m pip uninstall codex-telegram
```

Local runtime state remains in `.env` and `.codex-telegram/` unless you remove
those files yourself.

## Environment

Required:

- `TELEGRAM_BOT_TOKEN`: bot token from BotFather at https://t.me/BotFather

Recommended:

- `TELEGRAM_ALLOWED_USERS`: comma-separated numeric Telegram user IDs from
  userinfobot at https://t.me/userinfobot
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
- `CODEX_TELEGRAM_STATE_DIR`: defaults to `$CODEX_WORKDIR/.codex-telegram`

## Security

`.env` is local secret state and must not be committed. It contains the Telegram
bot token and allowed user IDs. The repository ignores `.env`, `.codex-telegram/`,
Python caches, build output, and local backup files. Do not paste bot tokens or
Telegram user IDs into issues, logs, commits, or documentation.

YOLO mode is intentionally dangerous. It maps to Codex
`--dangerously-bypass-approvals-and-sandbox`, skipping approval prompts and
running without sandboxing. Use it only for chats and workspaces you trust.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Notes

This bridge keeps the Telegram-to-Codex control path in a focused local repo. The agent side is Codex CLI, so the bot can work in whatever `CODEX_WORKDIR` points at.
