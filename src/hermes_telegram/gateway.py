from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .codex_runner import CodexRunner
from .config import Settings
from .model_catalog import CodexModelCatalog, ModelChoice
from .session_store import ChatModelPreference, SessionStore
from .telegram_api import IncomingCallback, IncomingMessage, TelegramAPI, parse_callback_update, parse_message_update

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
    model: str
    reasoning_effort: str


class HermesTelegramGateway:
    def __init__(
        self,
        *,
        settings: Settings,
        telegram: TelegramAPI,
        codex: CodexRunner,
        model_catalog: CodexModelCatalog | None = None,
        sessions: SessionStore,
    ):
        self._settings = settings
        self._telegram = telegram
        self._codex = codex
        self._model_catalog = model_catalog or CodexModelCatalog(settings.codex_command)
        self._sessions = sessions
        self._offset: int | None = None

    def _active_model_preference(self, chat_id: str | None) -> ChatModelPreference | None:
        if chat_id is None:
            return None
        preference = self._sessions.load_model_preference(chat_id)
        if preference is None:
            return None
        model = self._model_catalog.get_model(preference.model)
        if not self._model_catalog.is_authoritative():
            logger.warning(
                "Keeping model preference after non-authoritative model lookup chat=%s model=%s effort=%s",
                chat_id,
                preference.model,
                preference.reasoning_effort,
            )
            return preference
        if model is not None and preference.reasoning_effort in model.reasoning_efforts:
            return preference
        logger.warning(
            "Ignoring unavailable model preference chat=%s model=%s effort=%s",
            chat_id,
            preference.model,
            preference.reasoning_effort,
        )
        self._sessions.clear_model_preference(chat_id)
        return None

    def status(self, chat_id: str | None = None) -> GatewayStatus:
        preference = self._active_model_preference(chat_id)
        return GatewayStatus(
            workdir=str(self._settings.codex_workdir),
            allowed_users=len(self._settings.allowed_users),
            allowed_chats=len(self._settings.allowed_chats),
            model=(preference.model if preference else self._settings.codex_model or "default"),
            reasoning_effort=(
                preference.reasoning_effort
                if preference
                else self._settings.codex_extra_args and "custom"
                or "config/default"
            ),
        )

    def _is_authorized(self, incoming: IncomingMessage | IncomingCallback) -> bool:
        if self._settings.allowed_chats and incoming.chat_id not in self._settings.allowed_chats:
            return False
        if self._settings.allowed_users and incoming.user_id not in self._settings.allowed_users:
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
                "/models selects the Codex model and thinking amount.\n"
                "/reset clears this chat's local bridge history."
            )
        if command == "/status":
            status = self.status(message.chat_id)
            return (
                f"Workspace: {status.workdir}\n"
                f"Allowed users configured: {status.allowed_users}\n"
                f"Allowed chats configured: {status.allowed_chats}\n"
                f"Model: {status.model}\n"
                f"Thinking: {status.reasoning_effort}"
            )
        if command == "/reset":
            self._sessions.reset(message.chat_id)
            return "This chat's bridge history was reset."
        return None

    @staticmethod
    def _button(text: str, callback_data: str) -> dict[str, str]:
        return {"text": text, "callback_data": callback_data}

    def _model_keyboard(self) -> dict[str, list[list[dict[str, str]]]]:
        rows = []
        for model in self._model_catalog.list_models():
            rows.append([self._button(model.display_name, f"model:{model.slug}")])
        return {"inline_keyboard": rows}

    def _effort_keyboard(self, model: ModelChoice) -> dict[str, list[list[dict[str, str]]]]:
        labels = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "X High"}
        row = [
            self._button(labels.get(effort, effort.title()), f"effort:{model.slug}:{effort}")
            for effort in model.reasoning_efforts
        ]
        return {"inline_keyboard": [row]}

    def _send_model_picker(self, message: IncomingMessage) -> None:
        self._telegram.send_message(
            message.chat_id,
            "Choose a Codex model:",
            reply_to_message_id=message.message_id,
            reply_markup=self._model_keyboard(),
        )

    def handle_message(self, message: IncomingMessage) -> None:
        if not self._is_authorized(message):
            logger.warning("Unauthorized Telegram message from user=%s chat=%s", message.user_id, message.chat_id)
            return

        command = message.text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command == "/models":
            self._send_model_picker(message)
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
        preference = self._active_model_preference(message.chat_id)
        result = self._codex.run(
            prompt,
            model=preference.model if preference else None,
            reasoning_effort=preference.reasoning_effort if preference else None,
        )
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

    def _reply_to_callback(
        self,
        callback: IncomingCallback,
        text: str,
        *,
        reply_markup: dict | None = None,
    ) -> None:
        if callback.message_id is None:
            self._telegram.send_message(callback.chat_id, text, reply_markup=reply_markup)
            return
        self._telegram.edit_message_text(
            callback.chat_id,
            callback.message_id,
            text,
            reply_markup=reply_markup,
        )

    def handle_callback(self, callback: IncomingCallback) -> None:
        if not self._is_authorized(callback):
            logger.warning("Unauthorized Telegram callback from user=%s chat=%s", callback.user_id, callback.chat_id)
            self._telegram.answer_callback_query(callback.callback_query_id, text="Not authorized.")
            return

        if callback.data.startswith("model:"):
            slug = callback.data.split(":", 1)[1]
            model = self._model_catalog.get_model(slug)
            if model is None:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Model is not available.")
                self._reply_to_callback(callback, "That model is no longer available. Send /models again.")
                return
            self._telegram.answer_callback_query(callback.callback_query_id)
            self._reply_to_callback(
                callback,
                f"Choose thinking amount for {model.display_name}:",
                reply_markup=self._effort_keyboard(model),
            )
            return

        if callback.data.startswith("effort:"):
            parts = callback.data.split(":", 2)
            if len(parts) != 3:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Unknown action.")
                return
            _, slug, effort = parts
            model = self._model_catalog.get_model(slug)
            if model is None or effort not in model.reasoning_efforts:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Selection is not available.")
                self._reply_to_callback(callback, "That selection is no longer available. Send /models again.")
                return
            self._sessions.save_model_preference(callback.chat_id, model=model.slug, reasoning_effort=effort)
            self._telegram.answer_callback_query(callback.callback_query_id, text="Model selected.")
            self._reply_to_callback(
                callback,
                f"Selected {model.display_name} with {effort} thinking.\n\nSend a message to talk to the agent.",
            )
            return

        self._telegram.answer_callback_query(callback.callback_query_id, text="Unknown action.")

    def handle_update(self, update: dict) -> None:
        message = parse_message_update(update)
        if message is None:
            callback = parse_callback_update(update)
            if callback is not None:
                self.handle_callback(callback)
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
