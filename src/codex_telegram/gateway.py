from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .codex_runner import EMPTY_CODEX_RESPONSE, CodexRunner
from .config import Settings
from .model_catalog import CodexModelCatalog, ModelChoice
from .session_store import ChatModelPreference, SessionStore, TopicSession
from .telegram_api import (
    IncomingCallback,
    IncomingMessage,
    TelegramAPI,
    TelegramAPIError,
    parse_callback_update,
    parse_message_update,
)

logger = logging.getLogger(__name__)

AUTO_COMPACT_HISTORY_CHARS = 24000
REASONING_EFFORT_RANK = {
    "minimal": 0,
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "xhigh": 4,
}


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
    sandbox: str
    fast_mode: str


@dataclass(frozen=True)
class SessionProvisioningRequest:
    workspace_path: Path
    workspace_relative: str
    model: ModelChoice
    reasoning_effort: str
    sandbox_mode: str
    topic_name: str


@dataclass(frozen=True)
class TopicLifecycleRequest:
    action: str
    target: str
    new_name: str | None = None


class CodexTelegramGateway:
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
        self._active_session_keys: set[str] = set()

    @staticmethod
    def _session_key(incoming: IncomingMessage | IncomingCallback) -> str:
        if incoming.message_thread_id is None:
            return incoming.chat_id
        return f"{incoming.chat_id}:thread:{incoming.message_thread_id}"

    def _workspace_path(self, relative_path: str) -> Path | None:
        root = self._settings.codex_workdir.resolve()
        candidate = root if relative_path in {"", "."} else (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if not candidate.is_dir():
            return None
        return candidate

    def _workspace_relative_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self._settings.codex_workdir.resolve())
        return "" if rel == Path(".") else rel.as_posix()

    def _active_workdir(self, chat_id: str | None) -> Path:
        if chat_id is None:
            return self._settings.codex_workdir
        relative_path = self._sessions.load_active_workspace(chat_id)
        path = self._workspace_path(relative_path)
        if path is None:
            self._sessions.clear_active_workspace(chat_id)
            return self._settings.codex_workdir
        return path

    def _active_sandbox_mode(self, chat_id: str | None) -> str | None:
        if chat_id is None:
            return None
        return self._sessions.load_sandbox_mode(chat_id)

    def _sandbox_status_label(self, chat_id: str | None) -> str:
        mode = self._active_sandbox_mode(chat_id)
        if mode is None:
            return f"configured ({self._settings.codex_sandbox})"
        return mode

    def _child_workspaces(self, path: Path) -> list[Path]:
        root = self._settings.codex_workdir.resolve()
        state_dir = self._settings.state_dir.resolve()
        children: list[Path] = []
        try:
            candidates = path.iterdir()
        except OSError:
            return []
        for child in candidates:
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                resolved = child.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved == state_dir:
                continue
            children.append(child)
        return sorted(children, key=lambda child: child.name.lower())

    def _workspace_keyboard(self, path: Path) -> dict[str, list[list[dict[str, str]]]]:
        current_relative = self._workspace_relative_path(path)
        current_token = self._sessions.remember_workspace_token(current_relative)
        rows = [[self._button("Start session", f"ws:s:{current_token}")]]
        for child in self._child_workspaces(path):
            child_token = self._sessions.remember_workspace_token(self._workspace_relative_path(child))
            rows.append([self._button(child.name, f"ws:o:{child_token}")])
        if current_relative:
            parent_token = self._sessions.remember_workspace_token(self._workspace_relative_path(path.parent))
            rows.append([self._button("Back", f"ws:o:{parent_token}")])
        return {"inline_keyboard": rows}

    def _workspace_text(self, path: Path) -> str:
        return f"Workspace:\n{path}"

    @staticmethod
    def _normalize_lookup(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _clean_workspace_phrase(value: str) -> str:
        cleaned = value.strip().strip("'\"`")
        for suffix in (" folder", " directory", " repo", " repository", " workspace"):
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                break
        return cleaned

    def _resolve_requested_workspace(self, value: str) -> tuple[Path, str] | None:
        cleaned = self._clean_workspace_phrase(value)
        if not cleaned:
            return None
        path = self._workspace_path(cleaned)
        if path is not None:
            return path, self._workspace_relative_path(path)

        root = self._settings.codex_workdir.resolve()
        for child in self._child_workspaces(root):
            if child.name.lower() == cleaned.lower():
                return child.resolve(), self._workspace_relative_path(child)
        return None

    def _resolve_requested_model(self, text: str) -> ModelChoice | None:
        normalized_text = self._normalize_lookup(text)
        matches: dict[str, tuple[int, ModelChoice]] = {}
        for model in self._model_catalog.list_models():
            names = {model.slug, model.display_name}
            for name in names:
                normalized_name = self._normalize_lookup(name)
                if normalized_name and normalized_name in normalized_text:
                    current = matches.get(model.slug)
                    if current is None or len(normalized_name) > current[0]:
                        matches[model.slug] = (len(normalized_name), model)
                    break
        if not matches:
            return None
        best_length = max(length for length, _ in matches.values())
        best_matches = [model for length, model in matches.values() if length == best_length]
        if len(best_matches) != 1:
            return None
        return best_matches[0]

    @staticmethod
    def _extract_reasoning_effort(text: str) -> str | None:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        if re.search(r"\bx\s*high\b|\bxhigh\b", normalized_text):
            return "xhigh"
        for effort in ("high", "medium", "low"):
            if re.search(rf"\b{effort}\b", normalized_text):
                return effort
        return None

    @staticmethod
    def _extract_sandbox_mode(text: str) -> str | None:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        if re.search(r"\byolo\b|\bdanger full access\b|\bbypass\b", normalized_text):
            return "yolo"
        if re.search(r"\bconstrained\b|\bworkspace write\b|\bsafe\b", normalized_text):
            return "constrained"
        return None

    def _extract_workspace_phrase(self, text: str) -> str | None:
        match = re.search(
            r"\bin\s+(.+?)(?:\s+with\b|\s+using\b|\s+on\b|\s+for\b|$)",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        return self._clean_workspace_phrase(match.group(1))

    def _topic_name_for(self, request: SessionProvisioningRequest) -> str:
        workspace = request.workspace_relative or request.workspace_path.name
        name = f"{workspace} | {request.model.slug} {request.reasoning_effort} | {request.sandbox_mode}"
        return name[:128]

    def _parse_session_provisioning_request(self, text: str) -> tuple[SessionProvisioningRequest | None, str | None]:
        lowered = text.lower()
        if "session" not in lowered and "agent" not in lowered:
            return None, (
                "General chat is for creating and managing Codex sessions. "
                "Try: make me a session in kitia folder with gpt 5.5 high in yolo mode"
            )

        workspace_phrase = self._extract_workspace_phrase(text)
        if workspace_phrase is None:
            return None, "I need a workspace. Example: make me a session in kitia folder with gpt 5.5 high in yolo mode"
        workspace = self._resolve_requested_workspace(workspace_phrase)
        if workspace is None:
            return None, f"I could not find a workspace named `{workspace_phrase}` under {self._settings.codex_workdir}."
        workspace_path, workspace_relative = workspace

        model = self._resolve_requested_model(text)
        if model is None:
            return None, "I need exactly one available model name, for example `gpt 5.5`."

        reasoning_effort = self._extract_reasoning_effort(text)
        if reasoning_effort is None:
            return None, "I need a thinking effort: low, medium, high, or xhigh."
        if reasoning_effort not in model.reasoning_efforts:
            return None, f"`{model.slug}` does not support `{reasoning_effort}` thinking."

        sandbox_mode = self._extract_sandbox_mode(text)
        if sandbox_mode is None:
            return None, "I need a sandbox mode: constrained or yolo."

        placeholder = SessionProvisioningRequest(
            workspace_path=workspace_path,
            workspace_relative=workspace_relative,
            model=model,
            reasoning_effort=reasoning_effort,
            sandbox_mode=sandbox_mode,
            topic_name="",
        )
        return (
            SessionProvisioningRequest(
                workspace_path=workspace_path,
                workspace_relative=workspace_relative,
                model=model,
                reasoning_effort=reasoning_effort,
                sandbox_mode=sandbox_mode,
                topic_name=self._topic_name_for(placeholder),
            ),
            None,
        )

    def _is_general_forum_message(self, message: IncomingMessage) -> bool:
        return message.chat_type == "supergroup" and message.message_thread_id is None

    @staticmethod
    def _parse_group_rename_request(text: str) -> str | None:
        match = re.fullmatch(r"rename\s+group\s+to\s+(.+)", text.strip(), flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).strip().strip("'\"`")

    @staticmethod
    def _is_metadata_report_request(text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return normalized in {
            "report metadata",
            "show metadata",
            "group metadata",
            "topic metadata",
            "session metadata",
            "list topics",
            "show topics",
        }

    def _handle_group_rename_request(self, message: IncomingMessage, title: str) -> bool:
        if not 1 <= len(title) <= 128:
            self._telegram.send_message(
                message.chat_id,
                "Group titles must be 1 to 128 characters.",
                reply_to_message_id=message.message_id,
            )
            return True
        try:
            self._telegram.set_chat_title(message.chat_id, title)
        except TelegramAPIError:
            logger.warning("Failed to rename Telegram group chat=%s", message.chat_id, exc_info=True)
            self._telegram.send_message(
                message.chat_id,
                "I could not rename the group. Check the bot's group admin permissions.",
                reply_to_message_id=message.message_id,
            )
            return True
        self._telegram.send_message(
            message.chat_id,
            f"Renamed group to `{title}`.",
            reply_to_message_id=message.message_id,
        )
        return True

    def _handle_metadata_report_request(self, message: IncomingMessage) -> bool:
        try:
            chat = self._telegram.get_chat(message.chat_id)
        except TelegramAPIError:
            logger.warning("Failed to fetch Telegram group metadata chat=%s", message.chat_id, exc_info=True)
            self._telegram.send_message(
                message.chat_id,
                "I could not read the group metadata. Check the bot's group admin permissions.",
                reply_to_message_id=message.message_id,
            )
            return True

        forum = "yes" if chat.is_forum else "no"
        lines = [
            "Group metadata:",
            f"Title: {chat.title or 'unknown'}",
            f"Type: {chat.chat_type or 'unknown'}",
            f"Forum topics enabled: {forum}",
            "",
            f"Recorded topic sessions: {len(self._sessions.list_topic_sessions(message.chat_id))}",
        ]
        for session in self._sessions.list_topic_sessions(message.chat_id):
            status = "closed" if session.is_closed else "open"
            workspace = session.workspace or "."
            lines.append(
                f"- {session.topic_name} [{status}] "
                f"workspace={workspace} model={session.model} "
                f"thinking={session.reasoning_effort} sandbox={session.sandbox_mode} "
                f"fast={'on' if session.fast_mode else 'off'}"
            )
        self._telegram.send_message(
            message.chat_id,
            "\n".join(lines),
            reply_to_message_id=message.message_id,
        )
        return True

    @staticmethod
    def _clean_topic_phrase(value: str) -> str:
        return value.strip().strip("'\"`")

    def _parse_topic_lifecycle_request(self, text: str) -> TopicLifecycleRequest | None:
        stripped = text.strip()
        rename = re.fullmatch(r"rename\s+topic\s+(.+?)\s+to\s+(.+)", stripped, flags=re.IGNORECASE)
        if rename is not None:
            return TopicLifecycleRequest(
                action="rename",
                target=self._clean_topic_phrase(rename.group(1)),
                new_name=self._clean_topic_phrase(rename.group(2)),
            )

        lifecycle = re.fullmatch(
            r"(close|reopen|delete|remove)\s+topic\s+(.+)",
            stripped,
            flags=re.IGNORECASE,
        )
        if lifecycle is None:
            return None
        action = lifecycle.group(1).lower()
        if action == "remove":
            action = "delete"
        return TopicLifecycleRequest(action=action, target=self._clean_topic_phrase(lifecycle.group(2)))

    def _resolve_topic_session(self, chat_id: str, target: str) -> tuple[TopicSession | None, str | None]:
        cleaned = self._clean_topic_phrase(target)
        if not cleaned:
            return None, "I need a topic name or thread id."
        if self._normalize_lookup(cleaned) in {"general", "generalchat", "all", "alltopic", "alltopics"}:
            return None, "I cannot target General chat or all topics with a session lifecycle command."

        sessions = self._sessions.list_topic_sessions(chat_id)
        if not sessions:
            return None, "I do not have any topic-backed sessions recorded for this chat."

        thread_id = cleaned.removeprefix("#")
        if thread_id.isdigit():
            matches = [session for session in sessions if session.message_thread_id == int(thread_id)]
            if len(matches) == 1:
                return matches[0], None
            return None, f"I could not find a recorded topic with thread id `{cleaned}`."

        normalized_target = self._normalize_lookup(cleaned)
        exact = [session for session in sessions if self._normalize_lookup(session.topic_name) == normalized_target]
        if len(exact) == 1:
            return exact[0], None
        if len(exact) > 1:
            return None, f"`{cleaned}` matches more than one topic. Use the thread id instead."

        partial = [session for session in sessions if normalized_target in self._normalize_lookup(session.topic_name)]
        if len(partial) == 1:
            return partial[0], None
        if len(partial) > 1:
            names = ", ".join(f"`{session.topic_name}`" for session in partial[:5])
            return None, f"`{cleaned}` is ambiguous. Matching topics: {names}."
        return None, f"I could not find a recorded topic named `{cleaned}`."

    def _handle_topic_lifecycle_request(
        self,
        message: IncomingMessage,
        request: TopicLifecycleRequest,
    ) -> bool:
        session, error = self._resolve_topic_session(message.chat_id, request.target)
        if session is None:
            self._telegram.send_message(
                message.chat_id,
                error or "I could not find that topic session.",
                reply_to_message_id=message.message_id,
            )
            return True

        try:
            if request.action == "rename":
                new_name = self._clean_topic_phrase(request.new_name or "")
                if not 1 <= len(new_name) <= 128:
                    self._telegram.send_message(
                        message.chat_id,
                        "Topic names must be 1 to 128 characters.",
                        reply_to_message_id=message.message_id,
                    )
                    return True
                self._telegram.edit_forum_topic(
                    message.chat_id,
                    session.message_thread_id,
                    name=new_name,
                )
                self._sessions.update_topic_session_name(session.session_key, new_name)
                self._telegram.send_message(
                    message.chat_id,
                    f"Renamed topic `{session.topic_name}` to `{new_name}`.",
                    reply_to_message_id=message.message_id,
                )
                return True

            if request.action == "close":
                self._telegram.close_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.set_topic_session_closed(session.session_key, True)
                self._telegram.send_message(
                    message.chat_id,
                    f"Closed topic `{session.topic_name}` and marked its session closed.",
                    reply_to_message_id=message.message_id,
                )
                return True

            if request.action == "reopen":
                self._telegram.reopen_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.set_topic_session_closed(session.session_key, False)
                self._telegram.send_message(
                    message.chat_id,
                    f"Reopened topic `{session.topic_name}`.",
                    reply_to_message_id=message.message_id,
                )
                return True

            if request.action == "delete":
                self._telegram.delete_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.remove_topic_session(session.session_key)
                self._telegram.send_message(
                    message.chat_id,
                    f"Deleted topic `{session.topic_name}` and removed only its bridge session state.",
                    reply_to_message_id=message.message_id,
                )
                return True
        except TelegramAPIError:
            logger.warning(
                "Failed Telegram topic lifecycle action=%s chat=%s thread=%s",
                request.action,
                message.chat_id,
                session.message_thread_id,
                exc_info=True,
            )
            self._telegram.send_message(
                message.chat_id,
                f"I could not {request.action} topic `{session.topic_name}`. "
                "Check the bot's topic admin permissions and whether the topic still exists.",
                reply_to_message_id=message.message_id,
            )
            return True

        self._telegram.send_message(
            message.chat_id,
            "I did not recognize that topic lifecycle action.",
            reply_to_message_id=message.message_id,
        )
        return True

    def _handle_general_forum_message(self, message: IncomingMessage) -> bool:
        group_title = self._parse_group_rename_request(message.text)
        if group_title is not None:
            return self._handle_group_rename_request(message, group_title)

        if self._is_metadata_report_request(message.text):
            return self._handle_metadata_report_request(message)

        lifecycle_request = self._parse_topic_lifecycle_request(message.text)
        if lifecycle_request is not None:
            return self._handle_topic_lifecycle_request(message, lifecycle_request)

        request, error = self._parse_session_provisioning_request(message.text)
        if request is None:
            self._telegram.send_message(
                message.chat_id,
                error or "I could not understand that session request.",
                reply_to_message_id=message.message_id,
            )
            return True

        try:
            topic = self._telegram.create_forum_topic(message.chat_id, request.topic_name)
        except TelegramAPIError:
            logger.warning("Failed to create Telegram forum topic chat=%s", message.chat_id, exc_info=True)
            self._telegram.send_message(
                message.chat_id,
                "I could not create a forum topic for that session. "
                "Check that topics are enabled and the bot can manage topics.",
                reply_to_message_id=message.message_id,
            )
            return True
        topic_key = f"{message.chat_id}:thread:{topic.message_thread_id}"
        self._sessions.save_active_workspace(topic_key, request.workspace_relative)
        self._sessions.save_model_preference(
            topic_key,
            model=request.model.slug,
            reasoning_effort=request.reasoning_effort,
        )
        self._sessions.save_sandbox_mode(topic_key, request.sandbox_mode)
        self._sessions.reset(topic_key)
        self._sessions.save_topic_session(
            chat_id=message.chat_id,
            message_thread_id=topic.message_thread_id,
            session_key=topic_key,
            topic_name=topic.name,
            workspace=request.workspace_relative,
            model=request.model.slug,
            reasoning_effort=request.reasoning_effort,
            sandbox_mode=request.sandbox_mode,
        )

        ready_text = (
            "Session ready.\n"
            f"Workspace: {request.workspace_path}\n"
            f"Model: {request.model.slug}\n"
            f"Thinking: {request.reasoning_effort}\n"
            f"Sandbox: {request.sandbox_mode}\n\n"
            "Send messages in this topic to talk to this agent."
        )
        self._telegram.send_message(
            message.chat_id,
            ready_text,
            message_thread_id=topic.message_thread_id,
        )
        self._telegram.send_message(
            message.chat_id,
            f"Created topic `{topic.name}` and configured the Codex session there.",
            reply_to_message_id=message.message_id,
        )
        return True

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

    def _fast_reasoning_effort(self, model_slug: str | None) -> str:
        if model_slug:
            model = self._model_catalog.get_model(model_slug)
            if model is not None:
                ranked_efforts = [
                    effort for effort in model.reasoning_efforts if effort in REASONING_EFFORT_RANK
                ]
                if ranked_efforts:
                    return min(ranked_efforts, key=lambda effort: REASONING_EFFORT_RANK[effort])
                if model.reasoning_efforts:
                    return model.reasoning_efforts[0]
        return "low"

    def _effective_model_preference(self, chat_id: str | None) -> ChatModelPreference | None:
        preference = self._active_model_preference(chat_id)
        if chat_id is None or not self._sessions.load_fast_mode(chat_id):
            return preference
        model = preference.model if preference is not None else self._settings.codex_model
        return ChatModelPreference(
            model=model,
            reasoning_effort=self._fast_reasoning_effort(model or None),
        )

    def status(self, chat_id: str | None = None) -> GatewayStatus:
        preference = self._effective_model_preference(chat_id)
        fast_mode = self._sessions.load_fast_mode(chat_id) if chat_id is not None else False
        return GatewayStatus(
            workdir=str(self._active_workdir(chat_id)),
            allowed_users=len(self._settings.allowed_users),
            allowed_chats=len(self._settings.allowed_chats),
            model=(preference.model if preference else self._settings.codex_model or "default"),
            reasoning_effort=(
                preference.reasoning_effort
                if preference
                else self._settings.codex_extra_args and "custom"
                or "config/default"
            ),
            sandbox=self._sandbox_status_label(chat_id),
            fast_mode="on" if fast_mode else "off",
        )

    def _is_authorized(self, incoming: IncomingMessage | IncomingCallback) -> bool:
        if self._settings.allowed_chats and incoming.chat_id not in self._settings.allowed_chats:
            return False
        if self._settings.allowed_users and incoming.user_id not in self._settings.allowed_users:
            return False
        return True

    def _build_codex_prompt(self, message: IncomingMessage) -> str:
        session_key = self._session_key(message)
        history = self._sessions.render_recent(session_key)
        compact_summary = self._compact_summary_for_session(session_key)
        topic = (
            f" message_thread_id={message.message_thread_id}"
            if message.message_thread_id is not None
            else ""
        )
        identity = (
            f"Telegram user_id={message.user_id or 'unknown'} "
            f"username={message.username or 'unknown'} chat_id={message.chat_id} "
            f"chat_type={message.chat_type}{topic} "
            f"workspace={self._active_workdir(session_key)} "
            f"sandbox={self._sandbox_status_label(session_key)}"
        )
        parts = [SYSTEM_PROMPT.strip(), identity]
        if compact_summary:
            parts.append(f"Compacted Telegram conversation context:\n{compact_summary}")
        if history:
            parts.append(history)
        parts.append(f"Current Telegram message:\n{message.text}")
        return "\n\n".join(parts)

    def _compact_summary_for_session(self, session_key: str) -> str:
        session = self._sessions.load_topic_session(session_key)
        if session is None:
            return ""
        summary = session.compact_metadata.get("summary")
        return summary if isinstance(summary, str) else ""

    def _command_response(self, message: IncomingMessage) -> str | None:
        command = message.text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in {"/start", "/help"}:
            return "/reset\n/compact\n/fast\n/models\n/workspace\n/sandbox"
        if command == "/status":
            status = self.status(self._session_key(message))
            return (
                f"Workspace: {status.workdir}\n"
                f"Allowed users configured: {status.allowed_users}\n"
                f"Allowed chats configured: {status.allowed_chats}\n"
                f"Model: {status.model}\n"
                f"Thinking: {status.reasoning_effort}\n"
                f"Sandbox: {status.sandbox}\n"
                f"Fast mode: {status.fast_mode}"
            )
        if command == "/reset":
            session_key = self._session_key(message)
            self._sessions.reset(session_key)
            self._sessions.clear_compact_metadata(session_key)
            return "This chat's bridge history was reset."
        return None

    def _compact_session(
        self,
        message: IncomingMessage,
        *,
        auto: bool,
        reply_to_message_id: int | None,
    ) -> bool:
        session_key = self._session_key(message)
        topic_session = self._sessions.load_topic_session(session_key)
        if topic_session is None:
            if not auto:
                self._telegram.send_message(
                    message.chat_id,
                    "Run /compact inside a recorded Codex session topic.",
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message.message_thread_id,
                )
            return False
        if session_key in self._active_session_keys:
            self._telegram.send_message(
                message.chat_id,
                "This topic already has a Codex run in progress. Try /compact after it finishes.",
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message.message_thread_id,
            )
            return False

        history = self._sessions.render_recent(session_key)
        turns = self._sessions.load(session_key)
        if not history.strip():
            if not auto:
                self._telegram.send_message(
                    message.chat_id,
                    "There is no conversation context to compact in this topic.",
                    reply_to_message_id=reply_to_message_id,
                    message_thread_id=message.message_thread_id,
                )
            return False

        self._telegram.send_message(
            message.chat_id,
            "conversation compact started",
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message.message_thread_id,
        )
        preference = self._effective_model_preference(session_key)
        self._active_session_keys.add(session_key)
        try:
            result = self._codex.compact(
                history,
                existing_summary=self._compact_summary_for_session(session_key),
                model=preference.model if preference else None,
                reasoning_effort=preference.reasoning_effort if preference else None,
                workdir=self._active_workdir(session_key),
                sandbox_mode="read-only",
            )
        finally:
            self._active_session_keys.discard(session_key)

        summary = result.text.strip()
        if result.returncode != 0 or not summary or summary == EMPTY_CODEX_RESPONSE:
            self._telegram.send_message(
                message.chat_id,
                "conversation compact failed. The existing topic context was left unchanged.",
                reply_to_message_id=reply_to_message_id,
                message_thread_id=message.message_thread_id,
            )
            return False

        self._sessions.save_compact_metadata(
            session_key,
            summary=summary,
            source_char_count=len(history),
            turns_compacted=len(turns),
            auto=auto,
        )
        self._sessions.reset(session_key)
        self._telegram.send_message(
            message.chat_id,
            "conversation compact finished",
            reply_to_message_id=reply_to_message_id,
            message_thread_id=message.message_thread_id,
        )
        return True

    def _handle_compact_command(self, message: IncomingMessage) -> None:
        if message.message_thread_id is None:
            self._telegram.send_message(
                message.chat_id,
                "Run /compact inside a Codex session topic. General chat compaction does not compact topic sessions.",
                reply_to_message_id=message.message_id,
            )
            return
        self._compact_session(message, auto=False, reply_to_message_id=message.message_id)

    def _handle_fast_command(self, message: IncomingMessage) -> None:
        if message.message_thread_id is None:
            self._telegram.send_message(
                message.chat_id,
                "Run /fast inside a Codex session topic, or use `/fast status` there to inspect that topic.",
                reply_to_message_id=message.message_id,
            )
            return
        session_key = self._session_key(message)
        if self._sessions.load_topic_session(session_key) is None:
            self._telegram.send_message(
                message.chat_id,
                "Run /fast inside a recorded Codex session topic.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        parts = message.text.split(maxsplit=1)
        action = parts[1].strip().lower() if len(parts) > 1 else "on"
        if action in {"status", "state"}:
            self._telegram.send_message(
                message.chat_id,
                f"Fast mode: {'on' if self._sessions.load_fast_mode(session_key) else 'off'}",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return
        if action in {"on", "enable", "enabled"}:
            self._sessions.set_fast_mode(session_key, True)
            self._telegram.send_message(
                message.chat_id,
                "Fast mode enabled for this topic. Model and sandbox stay unchanged; reasoning uses the lowest supported effort.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return
        if action in {"off", "disable", "disabled"}:
            self._sessions.set_fast_mode(session_key, False)
            self._telegram.send_message(
                message.chat_id,
                "Fast mode disabled for this topic.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return
        self._telegram.send_message(
            message.chat_id,
            "Use `/fast`, `/fast off`, or `/fast status`.",
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
        )

    def _maybe_auto_compact(self, message: IncomingMessage, session_key: str) -> None:
        if message.message_thread_id is None:
            return
        if self._sessions.history_char_count(session_key) < AUTO_COMPACT_HISTORY_CHARS:
            return
        self._compact_session(message, auto=True, reply_to_message_id=message.message_id)

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

    def _sandbox_keyboard(self) -> dict[str, list[list[dict[str, str]]]]:
        return {
            "inline_keyboard": [
                [self._button("Constrained", "sandbox:constrained")],
                [self._button("YOLO", "sandbox:yolo")],
            ]
        }

    def _send_model_picker(self, message: IncomingMessage) -> None:
        self._telegram.send_message(
            message.chat_id,
            "Choose a Codex model:",
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
            reply_markup=self._model_keyboard(),
        )

    def _send_sandbox_picker(self, message: IncomingMessage) -> None:
        self._telegram.send_message(
            message.chat_id,
            "Choose sandbox mode:",
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
            reply_markup=self._sandbox_keyboard(),
        )

    def _send_workspace_browser(self, message: IncomingMessage) -> None:
        path = self._settings.codex_workdir.resolve()
        self._telegram.send_message(
            message.chat_id,
            self._workspace_text(path),
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
            reply_markup=self._workspace_keyboard(path),
        )

    def handle_message(self, message: IncomingMessage) -> None:
        if not self._is_authorized(message):
            logger.warning("Unauthorized Telegram message from user=%s chat=%s", message.user_id, message.chat_id)
            return

        command = message.text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command == "/models":
            self._send_model_picker(message)
            return
        if command == "/sandbox":
            self._send_sandbox_picker(message)
            return
        if command == "/workspace":
            self._send_workspace_browser(message)
            return
        if command == "/compact":
            self._handle_compact_command(message)
            return
        if command == "/fast":
            self._handle_fast_command(message)
            return

        command_response = self._command_response(message)
        if command_response is not None:
            self._telegram.send_message(
                message.chat_id,
                command_response,
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        if self._is_general_forum_message(message):
            if self._handle_general_forum_message(message):
                return

        session_key = self._session_key(message)
        if session_key in self._active_session_keys:
            self._telegram.send_message(
                message.chat_id,
                "This topic already has a Codex run in progress. Please wait for it to finish.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return
        self._maybe_auto_compact(message, session_key)
        prompt = self._build_codex_prompt(message)
        self._sessions.append(session_key, "user", message.text)
        try:
            self._telegram.send_chat_action(message.chat_id, message_thread_id=message.message_thread_id)
        except Exception:
            logger.warning("Failed to send Telegram typing indicator", exc_info=True)
        preference = self._effective_model_preference(session_key)
        self._active_session_keys.add(session_key)
        try:
            result = self._codex.run(
                prompt,
                model=preference.model if preference and preference.model else None,
                reasoning_effort=preference.reasoning_effort if preference else None,
                workdir=self._active_workdir(session_key),
                sandbox_mode=self._active_sandbox_mode(session_key),
            )
        finally:
            self._active_session_keys.discard(session_key)
        response = result.text.strip()
        if len(response) > self._settings.max_telegram_response_chars:
            response = response[: self._settings.max_telegram_response_chars].rstrip()
            response += "\n\n[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]"
        self._sessions.append(session_key, "assistant", response)
        self._telegram.send_message(
            message.chat_id,
            response,
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
        )

    def _reply_to_callback(
        self,
        callback: IncomingCallback,
        text: str,
        *,
        reply_markup: dict | None = None,
    ) -> None:
        if callback.message_id is None:
            self._telegram.send_message(
                callback.chat_id,
                text,
                message_thread_id=callback.message_thread_id,
                reply_markup=reply_markup,
            )
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

        session_key = self._session_key(callback)
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

        if callback.data.startswith("ws:"):
            parts = callback.data.split(":", 2)
            if len(parts) != 3:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Unknown action.")
                return
            _, action, token = parts
            relative_path = self._sessions.resolve_workspace_token(token)
            path = self._workspace_path(relative_path or "") if relative_path is not None else None
            if path is None:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Workspace is not available.")
                self._reply_to_callback(callback, "Workspace is not available. Send /workspace again.")
                return
            if action == "o":
                self._telegram.answer_callback_query(callback.callback_query_id)
                self._reply_to_callback(
                    callback,
                    self._workspace_text(path),
                    reply_markup=self._workspace_keyboard(path),
                )
                return
            if action == "s":
                self._sessions.save_active_workspace(session_key, self._workspace_relative_path(path))
                self._sessions.reset(session_key)
                self._sessions.clear_compact_metadata(session_key)
                self._telegram.answer_callback_query(callback.callback_query_id, text="Session started.")
                status = self.status(session_key)
                self._reply_to_callback(
                    callback,
                    f"Session workspace:\n{path}\n\n"
                    f"Model: {status.model}\n"
                    f"Thinking: {status.reasoning_effort}\n"
                    f"Sandbox: {status.sandbox}\n"
                    f"Fast mode: {status.fast_mode}",
                )
                return
            self._telegram.answer_callback_query(callback.callback_query_id, text="Unknown action.")
            return

        if callback.data.startswith("sandbox:"):
            mode = callback.data.split(":", 1)[1]
            if mode not in {"constrained", "yolo"}:
                self._telegram.answer_callback_query(callback.callback_query_id, text="Unknown sandbox mode.")
                return
            self._sessions.save_sandbox_mode(session_key, mode)
            self._telegram.answer_callback_query(callback.callback_query_id, text="Sandbox selected.")
            label = "YOLO" if mode == "yolo" else "Constrained"
            self._reply_to_callback(
                callback,
                f"Selected sandbox: {label}\n\nSend a message to talk to the agent.",
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
            self._sessions.save_model_preference(session_key, model=model.slug, reasoning_effort=effort)
            self._sessions.set_fast_mode(session_key, False)
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
        processed_chat_key = self._session_key(message)
        if self._sessions.was_processed(processed_chat_key, message.message_id, message.update_id):
            logger.info(
                "Skipping already-processed Telegram message session=%s message_id=%s update_id=%s",
                processed_chat_key,
                message.message_id,
                message.update_id,
            )
            return
        self._sessions.mark_processed(processed_chat_key, message.message_id, message.update_id)
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
