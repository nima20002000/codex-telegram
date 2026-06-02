from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .codex_runner import CodexRunner
from .config import Settings
from .session_store import SessionStore
from .telegram_api import IncomingMessage, TelegramAPI, parse_message_update

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Codex, reached through a Telegram bot bridge.
Work in the configured repository and answer concisely for a chat interface.
When changing files, make the real change in the local workspace and verify it.
Do not ask the Telegram user to copy files that already exist on this machine.
"""


@dataclass(frozen=True)
class GatewayStatus:
    workdir: str
    allowed_users: int
    allowed_chats: int


class HermesTelegramGateway:
    def __init__(
        self,
        *,
        settings: Settings,
        telegram: TelegramAPI,
        codex: CodexRunner,
        sessions: SessionStore,
    ):
        self._settings = settings
        self._telegram = telegram
        self._codex = codex
        self._sessions = sessions
        self._offset: int | None = None

    def status(self) -> GatewayStatus:
        return GatewayStatus(
            workdir=str(self._settings.codex_workdir),
            allowed_users=len(self._settings.allowed_users),
            allowed_chats=len(self._settings.allowed_chats),
        )

    def _is_authorized(self, message: IncomingMessage) -> bool:
        if self._settings.allowed_chats and message.chat_id not in self._settings.allowed_chats:
            return False
        if self._settings.allowed_users and message.user_id not in self._settings.allowed_users:
            return False
        return True

    def _build_codex_prompt(self, message: IncomingMessage) -> str:
        history = self._sessions.render_recent(message.chat_id)
        identity = (
            f"Telegram user_id={message.user_id or 'unknown'} "
            f"username={message.username or 'unknown'} chat_id={message.chat_id}"
        )
        parts = [SYSTEM_PROMPT.strip(), identity]
        if history:
            parts.append(history)
        parts.append(f"Current Telegram message:\n{message.text}")
        return "\n\n".join(parts)

    def _command_response(self, message: IncomingMessage) -> str | None:
        command = message.text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in {"/start", "/help"}:
            return (
                "Hermes Telegram bridge is online.\n\n"
                "Send a normal message to run Codex in the configured workspace.\n"
                "/status shows the current workspace.\n"
                "/reset clears this chat's local bridge history."
            )
        if command == "/status":
            status = self.status()
            return (
                f"Workspace: {status.workdir}\n"
                f"Allowed users configured: {status.allowed_users}\n"
                f"Allowed chats configured: {status.allowed_chats}"
            )
        if command == "/reset":
            self._sessions.reset(message.chat_id)
            return "This chat's bridge history was reset."
        return None

    def handle_message(self, message: IncomingMessage) -> None:
        if not self._is_authorized(message):
            logger.warning("Unauthorized Telegram message from user=%s chat=%s", message.user_id, message.chat_id)
            return

        command_response = self._command_response(message)
        if command_response is not None:
            self._telegram.send_message(
                message.chat_id,
                command_response,
                reply_to_message_id=message.message_id,
            )
            return

        prompt = self._build_codex_prompt(message)
        self._sessions.append(message.chat_id, "user", message.text)
        try:
            self._telegram.send_chat_action(message.chat_id)
        except Exception:
            logger.warning("Failed to send Telegram typing indicator", exc_info=True)
        result = self._codex.run(prompt)
        response = result.text.strip()
        if len(response) > self._settings.max_telegram_response_chars:
            response = response[: self._settings.max_telegram_response_chars].rstrip()
            response += "\n\n[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]"
        self._sessions.append(message.chat_id, "assistant", response)
        self._telegram.send_message(
            message.chat_id,
            response,
            reply_to_message_id=message.message_id,
        )

    def handle_update(self, update: dict) -> None:
        message = parse_message_update(update)
        if message is None:
            return
        if self._sessions.was_processed(message.chat_id, message.message_id, message.update_id):
            logger.info(
                "Skipping already-processed Telegram message chat=%s message_id=%s update_id=%s",
                message.chat_id,
                message.message_id,
                message.update_id,
            )
            return
        self._sessions.mark_processed(message.chat_id, message.message_id, message.update_id)
        self.handle_message(message)

    def poll_once(self) -> None:
        updates = self._telegram.get_updates(
            offset=self._offset,
            timeout=self._settings.telegram_poll_timeout_seconds,
        )
        for update in updates:
            update_id = int(update["update_id"])
            try:
                self.handle_update(update)
            finally:
                self._offset = update_id + 1

    def run_forever(self) -> None:
        logger.info("Starting Telegram polling for Codex workspace %s", self._settings.codex_workdir)
        while True:
            try:
                self.poll_once()
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Gateway loop error")
                time.sleep(3)
