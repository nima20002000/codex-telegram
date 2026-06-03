from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .codex_runner import CodexRunner
from .config import Settings
from .gateway import HermesTelegramGateway
from .session_store import SessionStore
from .telegram_api import TelegramAPI, TelegramAPIError


BOT_COMMANDS = [
    ("/reset", "Reset"),
    ("/models", "Models"),
    ("/workspace", "Workspace"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone Hermes Telegram bridge.")
    parser.add_argument("--env-file", default=".env", help="Path to env file. Default: .env")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    env_file = Path(args.env_file).expanduser().resolve()
    settings = Settings.from_env(env_file=env_file, default_workdir=Path.cwd())
    settings.validate()
    telegram = TelegramAPI(
        settings.bot_token,
        request_timeout_seconds=settings.telegram_request_timeout_seconds,
    )
    try:
        telegram.set_bot_commands(BOT_COMMANDS)
    except TelegramAPIError:
        logging.warning("Failed to update Telegram bot commands", exc_info=True)

    gateway = HermesTelegramGateway(
        settings=settings,
        telegram=telegram,
        codex=CodexRunner(settings),
        sessions=SessionStore(settings.state_dir, history_turns=settings.session_history_turns),
    )
    gateway.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
