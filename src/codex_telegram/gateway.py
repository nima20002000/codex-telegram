from __future__ import annotations

import json
import logging
import re
import tempfile
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
PROGRESS_MIN_INTERVAL_SECONDS = 5.0
GENERAL_MEMORY_MAX_CHARS = 500
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
For Telegram replies, summarize code changes instead of pasting code blocks or raw diffs unless explicitly requested and safe.
Never include secrets, tokens, .env values, Telegram session data, or private chat IDs.
"""


SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_-]*(?:token|api[_-]?key|api[_-]?hash|password|secret)[A-Za-z0-9_-]*)\s*[:=]\s*[^,\s]+"
)
BOT_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
PRIVATE_CHAT_ID_RE = re.compile(r"-100\d{6,}")
ENV_PATH_RE = re.compile(r"\S*\.env(?:\.\S+)?")
SESSION_PATH_RE = re.compile(r"\S+\.session\b")
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:home|tmp|var|etc|usr|opt|root|mnt|media|run|dev|proc|srv)"
    r"(?:/[A-Za-z0-9._-]+)*"
)


@dataclass(frozen=True)
class GatewayStatus:
    workdir: str
    allowed_users: int
    allowed_chats: int
    model: str
    reasoning_effort: str
    sandbox: str
    fast_mode: str
    goal: str


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


@dataclass(frozen=True)
class GeneralControllerOutcome:
    handled: bool
    memory_text: str


GENERAL_CONTROLLER_PROMPT = """You are the General-chat controller for a Telegram forum group.
Your job is to understand the user's natural-language request and choose one safe Telegram bridge action.
Return only one JSON object. Do not include Markdown, prose, or code fences.

Available actions:
- create_topic_session: create a new forum topic backed by a topic-scoped Codex agent.
- rename_topic, close_topic, reopen_topic, delete_topic: manage one recorded topic session.
- rename_group: rename the Telegram group.
- report_metadata: show group/session metadata.
- reply: ask for missing information or answer a General-chat management question.

Action schemas:
{"action":"create_topic_session","workspace":"<relative workspace or .>","model":"<model slug>","reasoning_effort":"<one listed effort for the selected model>","sandbox_mode":"constrained|yolo"}
{"action":"rename_topic","target":"<topic name or thread id>","new_name":"<new topic name>"}
{"action":"close_topic","target":"<topic name or thread id>"}
{"action":"reopen_topic","target":"<topic name or thread id>"}
{"action":"delete_topic","target":"<topic name or thread id>"}
{"action":"rename_group","title":"<new group title>"}
{"action":"report_metadata"}
{"action":"reply","text":"<short response>"}

Rules:
- General chat is a controller only. Do not perform coding work from General chat.
- If the user asks to create a topic, session, or agent and gives enough settings, use create_topic_session.
- Treat "topic", "session", and "agent" as equivalent creation language.
- Correct obvious typos and casual wording when the intended workspace/settings are clear.
- The root workspace can be represented as ".".
- If a required setting is missing, return reply with a concise question.
- Use recent General-chat context to understand follow-up questions.
- Use only listed workspaces, models, sandbox modes, and the selected model's listed reasoning efforts.
- The bridge validates that create-session settings are visible in the current user's message.
- Destructive or administrative actions must match an explicit visible user request.
- Never include secrets, tokens, private chat ids, or hidden local state.
"""


def sanitize_progress_text(text: str) -> str | None:
    if "```" in text:
        return None
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("diff --git", "@@", "+++", "---")):
            return None
        if stripped.startswith(("+", "-")) and len(stripped) > 1:
            return None
        lines.append(stripped)
    sanitized = " ".join(lines)
    sanitized = BOT_TOKEN_RE.sub("<redacted-token>", sanitized)
    sanitized = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)
    sanitized = PRIVATE_CHAT_ID_RE.sub("<chat>", sanitized)
    sanitized = ENV_PATH_RE.sub("<env-file>", sanitized)
    sanitized = SESSION_PATH_RE.sub("<session-file>", sanitized)
    sanitized = ABSOLUTE_PATH_RE.sub("<path>", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if not sanitized:
        return None
    if len(sanitized) > 240:
        sanitized = sanitized[:237].rstrip() + "..."
    return sanitized


def sanitize_general_memory_text(text: str) -> str:
    sanitized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    sanitized = BOT_TOKEN_RE.sub("<redacted-token>", sanitized)
    sanitized = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", sanitized)
    sanitized = PRIVATE_CHAT_ID_RE.sub("<chat>", sanitized)
    sanitized = ENV_PATH_RE.sub("<env-file>", sanitized)
    sanitized = SESSION_PATH_RE.sub("<session-file>", sanitized)
    sanitized = ABSOLUTE_PATH_RE.sub("<path>", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if not sanitized:
        return "<empty>"
    if len(sanitized) > GENERAL_MEMORY_MAX_CHARS:
        sanitized = sanitized[: GENERAL_MEMORY_MAX_CHARS - 3].rstrip() + "..."
    return sanitized


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
        self._chat_is_forum_cache: dict[str, bool] = {}

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
        root = self._settings.codex_workdir.resolve()
        path = self._workspace_path(cleaned)
        if path is not None:
            return path, self._workspace_relative_path(path)
        for child in self._child_workspaces(root):
            if child.name.lower() == cleaned.lower():
                return child.resolve(), self._workspace_relative_path(child)
        if self._normalize_lookup(cleaned) in {
            ".",
            "root",
            "workspace",
            "desktop",
            self._normalize_lookup(root.name),
        }:
            return root, ""
        return None

    def _available_workspace_relatives(self, *, limit: int = 200) -> list[str]:
        root = self._settings.codex_workdir.resolve()
        out: list[str] = []
        queue = self._child_workspaces(root)
        index = 0
        while index < len(queue) and len(out) < limit:
            path = queue[index]
            index += 1
            relative = self._workspace_relative_path(path)
            out.append(relative)
            queue.extend(self._child_workspaces(path))
        return out

    def _general_controller_prompt(self, message: IncomingMessage) -> str:
        root = self._settings.codex_workdir.resolve()
        workspace_lines = [f"- . -> {root.name}"]
        for relative in self._available_workspace_relatives():
            workspace_lines.append(f"- {relative}")

        model_lines = [
            f"- {model.slug}: efforts={', '.join(model.reasoning_efforts)}"
            for model in self._model_catalog.list_models()
        ]
        session_lines = []
        for session in self._sessions.list_topic_sessions(message.chat_id):
            status = "closed" if session.is_closed else "open"
            session_lines.append(
                f"- {session.topic_name} [{status}] "
                f"workspace={session.workspace or '.'} model={session.model} "
                f"thinking={session.reasoning_effort} sandbox={session.sandbox_mode}"
            )
        if not session_lines:
            session_lines.append("- none")

        parts = [
            GENERAL_CONTROLLER_PROMPT.strip(),
            f"Root workspace name: {root.name}",
            "Available workspaces:\n" + "\n".join(workspace_lines),
            "Available models:\n" + "\n".join(model_lines),
            "Recorded topic sessions:\n" + "\n".join(session_lines),
        ]
        history = self._sessions.render_recent(message.chat_id)
        if history:
            parts.append("Recent General-chat controller context:\n" + history)
        parts.append(f"Current General-chat message:\n{message.text}")
        return "\n\n".join(parts)

    def _general_reply_memory_text(self, text: str) -> str:
        return "Controller reply: " + sanitize_general_memory_text(self._telegram_response_text(text))

    def _general_action_memory_text(self, action_name: str, **values: str) -> str:
        payload = {
            key: value
            for key, value in {"action": action_name, **values}.items()
            if value
        }
        sanitized_payload = {
            key: sanitize_general_memory_text(value)
            for key, value in payload.items()
        }
        return "Controller action succeeded: " + json.dumps(
            sanitized_payload,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, object] | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    def _run_general_controller(self, message: IncomingMessage) -> dict[str, object] | None:
        preference = self._effective_model_preference(message.chat_id)
        with tempfile.TemporaryDirectory(prefix="codex-telegram-controller-") as tmpdir:
            result = self._codex.run(
                self._general_controller_prompt(message),
                model=preference.model if preference and preference.model else None,
                reasoning_effort=preference.reasoning_effort if preference else None,
                workdir=Path(tmpdir),
                sandbox_mode="read-only",
                extra_args=("--skip-git-repo-check",),
            )
        if result.returncode != 0:
            logger.warning("General controller Codex run failed: %s", result.stderr)
            return None
        return self._extract_json_object(result.text)

    @staticmethod
    def _action_text(action: dict[str, object], key: str) -> str | None:
        value = action.get(key)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _visible_action_allowed(text: str, action_name: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        direct_prefix = r"^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|hey\s+can\s+you\s+|i\s+(?:want|need)\s+(?:you\s+)?to\s+)?"
        patterns = {
            "create_topic_session": (
                direct_prefix
                + r"\b(?:create|make|start|open|setup|set up)\b\s+"
                + r"(?:me\s+)?(?:a\s+|an\s+)?(?:new\s+)?(?:topic|session|agent)\b",
                direct_prefix + r"\bnew\s+(?:topic|session|agent)\b",
                r"^(?:please\s+)?i\s+(?:want|need)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:topic|session|agent)\b",
            ),
            "rename_group": (r"\brename\b.{0,40}\bgroup\b", r"\bchange\b.{0,40}\bgroup name\b"),
            "rename_topic": (r"\brename\b.{0,80}\btopic\b", r"\bchange\b.{0,80}\btopic name\b"),
            "close_topic": (r"\bclose\b.{0,80}\btopic\b",),
            "reopen_topic": (r"\breopen\b.{0,80}\btopic\b",),
            "delete_topic": (r"\bdelete\b.{0,80}\btopic\b", r"\bremove\b.{0,80}\btopic\b"),
            "report_metadata": (
                direct_prefix + r"\breport\b.{0,80}\b(metadata|topics|sessions)\b",
                direct_prefix + r"\bshow\b.{0,80}\b(metadata|topics|sessions)\b",
                direct_prefix + r"\blist\b.{0,80}\b(topics|sessions)\b",
                direct_prefix + r"\b(group|topic|session)\b.{0,20}\bmetadata\b",
            ),
        }
        return any(re.search(pattern, normalized) for pattern in patterns.get(action_name, ()))

    @classmethod
    def _visible_value_present(cls, text: str, value: str, *, exact: bool = False) -> bool:
        cleaned = value.strip().strip("'\"`")
        if cleaned == ".":
            return re.search(r"(^|\s)\.(\s|$)", text) is not None
        if exact:
            if cleaned.isdigit():
                return re.search(rf"(?<!\d){re.escape(cleaned)}(?!\d)", text) is not None
            if re.fullmatch(r"[A-Za-z0-9 ]+", cleaned):
                return cls._visible_token_sequence_present(text, cleaned)
            return cleaned.casefold() in text.casefold()
        return cls._visible_token_sequence_present(text, cleaned)

    @staticmethod
    def _visible_token_sequence_present(text: str, value: str) -> bool:
        value_tokens = re.findall(r"[a-z0-9]+", value.lower())
        text_tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not value_tokens:
            return value.casefold() in text.casefold()
        if len(value_tokens) > len(text_tokens):
            return False
        width = len(value_tokens)
        return any(text_tokens[index : index + width] == value_tokens for index in range(len(text_tokens) - width + 1))

    @staticmethod
    def _clean_visible_admin_value(value: str) -> str:
        return value.strip().strip("'\"`")

    @classmethod
    def _same_visible_admin_value(cls, controller_value: str, visible_value: str) -> bool:
        return cls._clean_visible_admin_value(controller_value).casefold() == cls._clean_visible_admin_value(visible_value).casefold()

    @classmethod
    def _same_visible_topic_target(cls, controller_value: str, visible_value: str) -> bool:
        controller = cls._clean_visible_admin_value(controller_value).removeprefix("#")
        visible = cls._clean_visible_admin_value(visible_value).removeprefix("#")
        return controller.casefold() == visible.casefold()

    @staticmethod
    def _admin_command_prefix() -> str:
        return r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|hey\s+can\s+you\s+)?"

    @classmethod
    def _visible_group_rename_title(cls, text: str) -> str | None:
        prefix = cls._admin_command_prefix()
        patterns = (
            prefix + r"rename\s+(?:the\s+)?group\s+to\s+(.+?)\s*$",
            prefix + r"change\s+(?:the\s+)?group\s+name\s+to\s+(.+?)\s*$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                return cls._clean_visible_admin_value(match.group(1))
        return None

    @classmethod
    def _visible_topic_lifecycle_values(cls, text: str, action_name: str) -> tuple[str | None, str | None]:
        prefix = cls._admin_command_prefix()
        if action_name == "rename_topic":
            patterns = (
                prefix + r"rename\s+(?:the\s+)?topic\s+(.+?)\s+to\s+(.+?)\s*$",
                prefix + r"rename\s+(?:the\s+)?(.+?)\s+topic\s+to\s+(.+?)\s*$",
                prefix + r"change\s+(?:the\s+)?topic\s+(.+?)\s+name\s+to\s+(.+?)\s*$",
            )
            for pattern in patterns:
                match = re.match(pattern, text, flags=re.IGNORECASE)
                if match is not None:
                    return (
                        cls._clean_visible_admin_value(match.group(1)),
                        cls._clean_visible_admin_value(match.group(2)),
                    )
            return None, None

        verbs = {
            "close_topic": "close",
            "reopen_topic": "reopen",
            "delete_topic": "(?:delete|remove)",
        }.get(action_name)
        if verbs is None:
            return None, None
        patterns = (
            prefix + verbs + r"\s+(?:the\s+)?topic\s+(.+?)\s*$",
            prefix + verbs + r"\s+(?:the\s+)?(.+?)\s+topic\s*$",
        )
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                return cls._clean_visible_admin_value(match.group(1)), None
        return None, None

    def _visible_model_choice(self, text: str) -> ModelChoice | None:
        matches: dict[str, tuple[int, ModelChoice]] = {}
        for model in self._model_catalog.list_models():
            names = {model.slug, model.display_name}
            for name in names:
                if self._visible_token_sequence_present(text, name):
                    normalized_name = self._normalize_lookup(name)
                    current = matches.get(model.slug)
                    if current is None or len(normalized_name) > current[0]:
                        matches[model.slug] = (len(normalized_name), model)
                    break
        if not matches:
            return None
        best_length = max(length for length, _ in matches.values())
        best_matches = [model for length, model in matches.values() if length == best_length]
        return best_matches[0] if len(best_matches) == 1 else None

    @staticmethod
    def _visible_reasoning_effort(text: str, supported_efforts: tuple[str, ...] | None = None) -> str | None:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        if re.search(r"\bx\s*high\b|\bxhigh\b", normalized_text):
            return "xhigh"
        efforts = supported_efforts if supported_efforts is not None else tuple(REASONING_EFFORT_RANK)

        unique_efforts = sorted(set(efforts), key=len, reverse=True)
        for effort in unique_efforts:
            if effort == "xhigh":
                continue
            normalized_effort = re.sub(r"[^a-z0-9]+", " ", effort.lower()).strip()
            if not normalized_effort:
                continue
            pattern = r"\b" + r"\s+".join(re.escape(part) for part in normalized_effort.split()) + r"\b"
            if re.search(pattern, normalized_text):
                return effort
        return None

    @staticmethod
    def _visible_sandbox_mode(text: str) -> str | None:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        negated_yolo = re.search(
            r"\b(?:not|no|without|disable|disabled|dont|don t|do not|never|avoid)\s+"
            r"(?:(?:use|using|enable|enabling|turn on|run in|running in|set|setting|choose|choosing|select|selecting)\s+)?"
            r"(?:(?:the|a|an|sandbox|mode)\s+){0,3}"
            r"(?:yolo|danger full access|bypass)\b",
            normalized_text,
        )
        if re.search(r"\bconstrained\b|\bworkspace write\b|\bsafe\b", normalized_text):
            return "constrained"
        if negated_yolo:
            return "constrained"
        if re.search(r"\byolo\b|\bdanger full access\b|\bbypass\b", normalized_text):
            return "yolo"
        return None

    @staticmethod
    def _visible_standalone_alias(text: str, alias: str) -> bool:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            before_index = match.start() - 1
            while before_index >= 0 and text[before_index].isspace():
                before_index -= 1
            after_index = match.end()
            while after_index < len(text) and text[after_index].isspace():
                after_index += 1
            if before_index >= 0 and text[before_index] in {"/", "\\", "-", "_", "."}:
                continue
            if after_index < len(text) and text[after_index] in {"/", "\\", "-", "_", "."}:
                continue
            return True
        return False

    def _visible_root_workspace_present(self, text: str, root: Path) -> bool:
        if re.search(r"(^|\s)\.(\s|$)", text):
            return True
        if any(
            self._visible_standalone_alias(text, alias)
            for alias in ("root", "current workspace", "current folder", "current repo", "workspace root")
        ):
            return True
        return any(
            self._visible_standalone_alias(text, alias)
            for alias in ("desktop", "deaktop", root.name)
            if alias
        )

    def _visible_create_settings_missing(
        self,
        text: str,
        request: SessionProvisioningRequest,
    ) -> list[str]:
        root = self._settings.codex_workdir.resolve()
        missing: list[str] = []

        if request.workspace_relative:
            workspace_aliases = {request.workspace_relative}
            if "/" not in request.workspace_relative:
                workspace_aliases.add(Path(request.workspace_relative).name)
            if not any(self._visible_value_present(text, alias) for alias in workspace_aliases):
                missing.append("workspace")
        else:
            if not self._visible_root_workspace_present(text, root):
                missing.append("workspace")

        visible_model = self._visible_model_choice(text)
        if visible_model is None or visible_model.slug != request.model.slug:
            missing.append("model")

        visible_effort = self._visible_reasoning_effort(text, request.model.reasoning_efforts)
        if visible_effort != request.reasoning_effort:
            missing.append("thinking")

        visible_sandbox = self._visible_sandbox_mode(text)
        if visible_sandbox != request.sandbox_mode:
            missing.append("sandbox")

        return missing

    def _topic_name_for(self, request: SessionProvisioningRequest) -> str:
        workspace = request.workspace_relative or request.workspace_path.name
        name = f"{workspace} | {request.model.slug} {request.reasoning_effort} | {request.sandbox_mode}"
        return name[:128]

    def _general_forum_status(self, message: IncomingMessage) -> bool | None:
        if message.chat_type != "supergroup" or message.message_thread_id is not None:
            return False
        cached = self._chat_is_forum_cache.get(message.chat_id)
        if cached is not None:
            return cached
        try:
            is_forum = self._telegram.get_chat(message.chat_id).is_forum
        except TelegramAPIError:
            logger.warning("Failed to check Telegram forum metadata chat=%s", message.chat_id, exc_info=True)
            return None
        self._chat_is_forum_cache[message.chat_id] = is_forum
        return is_forum

    def _handle_group_rename_request(self, message: IncomingMessage, title: str) -> GeneralControllerOutcome:
        if not 1 <= len(title) <= 128:
            response = "Group titles must be 1 to 128 characters."
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
        try:
            self._telegram.set_chat_title(message.chat_id, title)
        except TelegramAPIError:
            logger.warning("Failed to rename Telegram group chat=%s", message.chat_id, exc_info=True)
            response = "I could not rename the group. Check the bot's group admin permissions."
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
        response = f"Renamed group to `{title}`."
        self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
        return GeneralControllerOutcome(True, self._general_action_memory_text("rename_group", title=title))

    def _handle_metadata_report_request(self, message: IncomingMessage) -> GeneralControllerOutcome:
        try:
            chat = self._telegram.get_chat(message.chat_id)
        except TelegramAPIError:
            logger.warning("Failed to fetch Telegram group metadata chat=%s", message.chat_id, exc_info=True)
            response = "I could not read the group metadata. Check the bot's group admin permissions."
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

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
            goal = session.goal_metadata.get("status") if isinstance(session.goal_metadata, dict) else None
            lines.append(
                f"- {session.topic_name} [{status}] "
                f"workspace={workspace} model={session.model} "
                f"thinking={session.reasoning_effort} sandbox={session.sandbox_mode} "
                f"fast={'on' if session.fast_mode else 'off'} "
                f"goal={goal if isinstance(goal, str) else 'none'}"
            )
        self._telegram.send_message(
            message.chat_id,
            "\n".join(lines),
            reply_to_message_id=message.message_id,
        )
        return GeneralControllerOutcome(True, self._general_action_memory_text("report_metadata"))

    @staticmethod
    def _clean_topic_phrase(value: str) -> str:
        return value.strip().strip("'\"`")

    def _resolve_topic_session(
        self,
        chat_id: str,
        target: str,
        *,
        allow_partial: bool = True,
    ) -> tuple[TopicSession | None, str | None]:
        cleaned = self._clean_topic_phrase(target)
        if not cleaned:
            return None, "I need a topic name or thread id."
        if self._normalize_lookup(cleaned) in {"general", "generalchat", "all", "alltopic", "alltopics"}:
            return None, "I cannot target General chat or all topics with a session lifecycle command."

        sessions = self._sessions.list_topic_sessions(chat_id)
        if not sessions:
            return None, "I do not have any topic-backed sessions recorded for this chat."

        if not allow_partial:
            literal_matches = [
                session
                for session in sessions
                if self._clean_topic_phrase(session.topic_name).casefold() == cleaned.casefold()
            ]
            if len(literal_matches) == 1:
                return literal_matches[0], None
            if len(literal_matches) > 1:
                return None, f"`{cleaned}` matches more than one topic. Use the thread id instead."

        thread_id = cleaned.removeprefix("#")
        if thread_id.isdigit():
            matches = [session for session in sessions if session.message_thread_id == int(thread_id)]
            if len(matches) == 1:
                return matches[0], None
            if not allow_partial:
                return None, f"I could not find an exact recorded topic named `{cleaned}` or thread id `{cleaned}`."
            return None, f"I could not find a recorded topic with thread id `{cleaned}`."

        if not allow_partial:
            return None, f"I could not find an exact recorded topic named `{cleaned}`. Use the full topic title or thread id."

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
        *,
        exact_target: bool = False,
    ) -> GeneralControllerOutcome:
        session, error = self._resolve_topic_session(message.chat_id, request.target, allow_partial=not exact_target)
        if session is None:
            response = error or "I could not find that topic session."
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
            )
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

        try:
            if request.action == "rename":
                new_name = self._clean_topic_phrase(request.new_name or "")
                if not 1 <= len(new_name) <= 128:
                    response = "Topic names must be 1 to 128 characters."
                    self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                    return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
                self._telegram.edit_forum_topic(
                    message.chat_id,
                    session.message_thread_id,
                    name=new_name,
                )
                self._sessions.update_topic_session_name(session.session_key, new_name)
                response = f"Renamed topic `{session.topic_name}` to `{new_name}`."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(
                    True,
                    self._general_action_memory_text("rename_topic", target=session.topic_name, new_name=new_name),
                )

            if request.action == "close":
                self._telegram.close_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.set_topic_session_closed(session.session_key, True)
                response = f"Closed topic `{session.topic_name}` and marked its session closed."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(
                    True,
                    self._general_action_memory_text("close_topic", target=session.topic_name),
                )

            if request.action == "reopen":
                self._telegram.reopen_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.set_topic_session_closed(session.session_key, False)
                response = f"Reopened topic `{session.topic_name}`."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(
                    True,
                    self._general_action_memory_text("reopen_topic", target=session.topic_name),
                )

            if request.action == "delete":
                self._telegram.delete_forum_topic(message.chat_id, session.message_thread_id)
                self._sessions.remove_topic_session(session.session_key)
                response = f"Deleted topic `{session.topic_name}` and removed only its bridge session state."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(
                    True,
                    self._general_action_memory_text("delete_topic", target=session.topic_name),
                )
        except TelegramAPIError:
            logger.warning(
                "Failed Telegram topic lifecycle action=%s chat=%s thread=%s",
                request.action,
                message.chat_id,
                session.message_thread_id,
                exc_info=True,
            )
            response = (
                f"I could not {request.action} topic `{session.topic_name}`. "
                "Check the bot's topic admin permissions and whether the topic still exists."
            )
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

        response = "I did not recognize that topic lifecycle action."
        self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
        return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

    def _topic_agent_intro(
        self,
        *,
        request: SessionProvisioningRequest,
        session_key: str,
    ) -> str:
        prompt = (
            "You are the topic-scoped Codex agent for a Telegram forum topic.\n"
            "Write the first message in this new topic. Keep it short.\n"
            "Say that the session is ready, restate the configured settings below, "
            "and ask the user to send the actual task in this topic.\n"
            f"Workspace: {request.workspace_path}\n"
            f"Model: {request.model.slug}\n"
            f"Thinking: {request.reasoning_effort}\n"
            f"Configured sandbox: {request.sandbox_mode}\n"
            "This introduction is generated read-only, but the saved topic session uses the configured sandbox above.\n"
            "Do not include code, command output, Markdown tables, secrets, or private IDs."
        )
        result = self._codex.run(
            prompt,
            model=request.model.slug,
            reasoning_effort=request.reasoning_effort,
            workdir=request.workspace_path,
            sandbox_mode="read-only",
        )
        text = self._telegram_response_text(result.text)
        if result.returncode != 0 or not text or text == EMPTY_CODEX_RESPONSE:
            text = self._telegram_response_text(
                "Session ready.\n"
                f"Workspace: {request.workspace_path}\n"
                f"Model: {request.model.slug}\n"
                f"Thinking: {request.reasoning_effort}\n"
                f"Sandbox: {request.sandbox_mode}\n\n"
                "Send your task in this topic."
            )
        self._sessions.append(session_key, "assistant", text)
        return text

    def _create_topic_session(
        self,
        message: IncomingMessage,
        request: SessionProvisioningRequest,
    ) -> GeneralControllerOutcome:
        topic_name = request.topic_name.strip()[:128]
        if not topic_name:
            response = "I need a non-empty topic name for that session."
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

        try:
            topic = self._telegram.create_forum_topic(message.chat_id, topic_name)
        except TelegramAPIError:
            logger.warning("Failed to create Telegram forum topic chat=%s", message.chat_id, exc_info=True)
            response = (
                "I could not create a forum topic for that session. "
                "Check that topics are enabled and the bot can manage topics."
            )
            self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
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

        ready_text = self._topic_agent_intro(request=request, session_key=topic_key)
        self._telegram.send_message(
            message.chat_id,
            ready_text,
            message_thread_id=topic.message_thread_id,
        )
        response = f"Created topic `{topic.name}` and configured the Codex session there."
        self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
        return GeneralControllerOutcome(
            True,
            self._general_action_memory_text(
                "create_topic_session",
                workspace=request.workspace_relative or ".",
                model=request.model.slug,
                reasoning_effort=request.reasoning_effort,
                sandbox_mode=request.sandbox_mode,
            ),
        )

    def _controller_create_request(self, action: dict[str, object]) -> tuple[SessionProvisioningRequest | None, str | None]:
        workspace_value = self._action_text(action, "workspace")
        if workspace_value is None:
            return None, "I need a workspace for the new topic session."
        workspace = self._resolve_requested_workspace(workspace_value)
        if workspace is None:
            return None, f"I could not find a workspace named `{workspace_value}` under {self._settings.codex_workdir}."
        workspace_path, workspace_relative = workspace

        model_value = self._action_text(action, "model")
        if model_value is None:
            return None, "I need a model for the new topic session."
        model = self._model_catalog.get_model(model_value)
        if model is None:
            return None, f"`{model_value}` is not an available model."

        reasoning_effort = self._action_text(action, "reasoning_effort")
        if reasoning_effort is None:
            return None, "I need a thinking effort supported by the selected model."
        if reasoning_effort not in model.reasoning_efforts:
            return None, f"`{model.slug}` does not support `{reasoning_effort}` thinking."

        sandbox_mode = self._action_text(action, "sandbox_mode")
        if sandbox_mode not in {"constrained", "yolo"}:
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

    def _handle_general_controller_action(
        self,
        message: IncomingMessage,
        action: dict[str, object],
    ) -> GeneralControllerOutcome:
        action_name = self._action_text(action, "action")
        if action_name == "create_topic_session":
            if not self._visible_action_allowed(message.text, action_name):
                response = (
                    "For safety, ask to create a topic/session/agent explicitly with the workspace, model, "
                    "thinking, and sandbox."
                )
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            request, error = self._controller_create_request(action)
            if request is None:
                response = error or "I could not configure that topic session."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            missing = self._visible_create_settings_missing(message.text, request)
            if missing:
                response = "I need the visible request to include: " + ", ".join(missing) + "."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            return self._create_topic_session(message, request)

        if action_name == "reply":
            response = self._telegram_response_text(
                self._action_text(action, "text") or "I need more detail to manage the group."
            )
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
            )
            return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

        if action_name == "report_metadata":
            if not self._visible_action_allowed(message.text, action_name):
                response = "For safety, ask to show or list group/topic/session metadata explicitly."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            return self._handle_metadata_report_request(message)

        if action_name == "rename_group":
            visible_title = self._visible_group_rename_title(message.text)
            if visible_title is None:
                response = "For safety, ask the group rename explicitly, for example `rename group to <title>`."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            title = self._action_text(action, "title")
            if title is None:
                response = "I need the new group title."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            if not self._same_visible_admin_value(title, visible_title):
                response = "For safety, include the exact new group title in your message."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            return self._handle_group_rename_request(message, visible_title)

        lifecycle_actions = {
            "rename_topic": "rename",
            "close_topic": "close",
            "reopen_topic": "reopen",
            "delete_topic": "delete",
        }
        if action_name in lifecycle_actions:
            visible_target, visible_new_name = self._visible_topic_lifecycle_values(message.text, action_name)
            if visible_target is None:
                response = "For safety, ask the topic admin action explicitly, for example `delete topic <name>`."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            target = self._action_text(action, "target")
            if target is None:
                response = "I need the topic name or thread id."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            if not self._same_visible_topic_target(target, visible_target):
                response = "For safety, include the exact topic name or thread id in your message."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            new_name = self._action_text(action, "new_name")
            if action_name == "rename_topic" and (
                new_name is None
                or visible_new_name is None
                or not self._same_visible_admin_value(new_name, visible_new_name)
            ):
                response = "For safety, include the exact new topic name in your message."
                self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
                return GeneralControllerOutcome(True, self._general_reply_memory_text(response))
            request = TopicLifecycleRequest(
                action=lifecycle_actions[action_name],
                target=visible_target,
                new_name=visible_new_name if action_name == "rename_topic" else new_name,
            )
            return self._handle_topic_lifecycle_request(message, request, exact_target=True)

        response = (
            "I could not decide which group action to take. Ask for a topic/session action with workspace, "
            "model, thinking, and sandbox."
        )
        self._telegram.send_message(message.chat_id, response, reply_to_message_id=message.message_id)
        return GeneralControllerOutcome(True, self._general_reply_memory_text(response))

    def _handle_general_forum_message(self, message: IncomingMessage) -> bool:
        action = self._run_general_controller(message)
        if action is None:
            response = "I could not understand the General-chat controller response. Try again with workspace, model, thinking, and sandbox."
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
            )
            self._sessions.append(message.chat_id, "user", sanitize_general_memory_text(message.text))
            self._sessions.append(message.chat_id, "assistant", self._general_reply_memory_text(response))
            return True
        outcome = self._handle_general_controller_action(message, action)
        self._sessions.append(message.chat_id, "user", sanitize_general_memory_text(message.text))
        self._sessions.append(message.chat_id, "assistant", outcome.memory_text)
        return outcome.handled

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
            goal=self._goal_status_label(chat_id),
        )

    def _telegram_response_text(self, text: str) -> str:
        response = text.strip()
        if len(response) > self._settings.max_telegram_response_chars:
            response = response[: self._settings.max_telegram_response_chars].rstrip()
            response += "\n\n[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]"
        return response

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
        goal_context = self._goal_context_for_session(session_key)
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
        if goal_context:
            parts.append(goal_context)
        if compact_summary:
            parts.append(f"Compacted Telegram conversation context:\n{compact_summary}")
        if history:
            parts.append(history)
        parts.append(f"Current Telegram message:\n{message.text}")
        return "\n\n".join(parts)

    def _goal_metadata_for_session(self, session_key: str | None) -> dict[str, object]:
        if session_key is None:
            return {}
        session = self._sessions.load_topic_session(session_key)
        if session is None:
            return {}
        return session.goal_metadata

    def _goal_status_label(self, session_key: str | None) -> str:
        goal = self._goal_metadata_for_session(session_key)
        status = goal.get("status")
        objective = goal.get("objective")
        if not isinstance(status, str) or not isinstance(objective, str) or not objective:
            return "none"
        return f"{status}: {objective}"

    def _goal_context_for_session(self, session_key: str) -> str:
        goal = self._goal_metadata_for_session(session_key)
        if goal.get("status") != "active":
            return ""
        objective = goal.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            return ""
        lines = [
            "Active Telegram goal:",
            f"Objective: {objective}",
            "Status: active",
            "Continue working toward this goal until it is completed or cleared.",
        ]
        notes = goal.get("notes", [])
        if isinstance(notes, list):
            valid_notes = [note for note in notes if isinstance(note, str) and note.strip()]
            if valid_notes:
                lines.append("Goal notes:")
                lines.extend(f"- {note}" for note in valid_notes[-5:])
        return "\n".join(lines)

    def _compact_summary_for_session(self, session_key: str) -> str:
        session = self._sessions.load_topic_session(session_key)
        if session is None:
            return ""
        summary = session.compact_metadata.get("summary")
        return summary if isinstance(summary, str) else ""

    def _command_response(self, message: IncomingMessage) -> str | None:
        command = message.text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if command in {"/start", "/help"}:
            return "/reset\n/compact\n/fast\n/goal\n/models\n/workspace\n/sandbox"
        if command == "/status":
            status = self.status(self._session_key(message))
            return (
                f"Workspace: {status.workdir}\n"
                f"Allowed users configured: {status.allowed_users}\n"
                f"Allowed chats configured: {status.allowed_chats}\n"
                f"Model: {status.model}\n"
                f"Thinking: {status.reasoning_effort}\n"
                f"Sandbox: {status.sandbox}\n"
                f"Fast mode: {status.fast_mode}\n"
                f"Goal: {status.goal}"
            )
        if command == "/reset":
            session_key = self._session_key(message)
            self._sessions.reset(session_key)
            self._sessions.clear_compact_metadata(session_key)
            if self._goal_metadata_for_session(session_key).get("status") == "active":
                return "This chat's bridge history was reset. Active goal was kept; use /goal clear to remove it."
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
        goal_context = self._goal_context_for_session(session_key)
        conversation_context = "\n\n".join(part for part in (goal_context, history) if part)
        preference = self._effective_model_preference(session_key)
        self._active_session_keys.add(session_key)
        try:
            result = self._codex.compact(
                conversation_context,
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

    def _format_goal_status(self, session_key: str) -> str:
        goal = self._goal_metadata_for_session(session_key)
        objective = goal.get("objective")
        status = goal.get("status")
        if not isinstance(objective, str) or not isinstance(status, str):
            return "Goal: none"
        lines = [f"Goal: {status}", f"Objective: {objective}"]
        notes = goal.get("notes", [])
        if isinstance(notes, list):
            valid_notes = [note for note in notes if isinstance(note, str) and note.strip()]
            if valid_notes:
                lines.append("Notes:")
                lines.extend(f"- {note}" for note in valid_notes[-5:])
        return "\n".join(lines)

    def _handle_goal_command(self, message: IncomingMessage) -> None:
        if message.message_thread_id is None:
            self._telegram.send_message(
                message.chat_id,
                "Run /goal inside a recorded Codex session topic. General chat goals do not target topic sessions.",
                reply_to_message_id=message.message_id,
            )
            return
        session_key = self._session_key(message)
        if self._sessions.load_topic_session(session_key) is None:
            self._telegram.send_message(
                message.chat_id,
                "Run /goal inside a recorded Codex session topic.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        parts = message.text.split(maxsplit=1)
        raw_action = parts[1].strip() if len(parts) > 1 else "status"
        action, _, remainder = raw_action.partition(" ")
        action_lower = action.lower()

        if action_lower in {"status", "state"} or not raw_action:
            self._telegram.send_message(
                message.chat_id,
                self._format_goal_status(session_key),
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        if action_lower in {"clear", "cancel", "remove"}:
            self._sessions.clear_goal(session_key)
            self._telegram.send_message(
                message.chat_id,
                "Goal cleared for this topic.",
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        if action_lower in {"complete", "done"}:
            if self._sessions.complete_goal(session_key):
                response = "Goal marked complete for this topic."
            else:
                response = "There is no active goal to complete in this topic."
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        if action_lower in {"update", "note"}:
            note = remainder.strip()
            if not note:
                response = "Use `/goal update <note>` to add progress or constraints."
            elif self._sessions.append_goal_note(session_key, note):
                response = "Goal updated for this topic."
            else:
                response = "There is no active goal to update in this topic."
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        if action_lower in {"replace", "set"}:
            objective = remainder.strip()
            if not objective:
                response = "Use `/goal set <objective>` to set a topic goal."
            else:
                self._sessions.save_goal(session_key, objective=objective)
                response = "Goal set for this topic."
            self._telegram.send_message(
                message.chat_id,
                response,
                reply_to_message_id=message.message_id,
                message_thread_id=message.message_thread_id,
            )
            return

        active_goal = self._goal_metadata_for_session(session_key).get("status") == "active"
        if active_goal:
            response = (
                "A goal is already active in this topic. Use `/goal update <note>`, "
                "`/goal complete`, `/goal clear`, or `/goal replace <objective>`."
            )
        else:
            self._sessions.save_goal(session_key, objective=raw_action)
            response = "Goal set for this topic."
        self._telegram.send_message(
            message.chat_id,
            response,
            reply_to_message_id=message.message_id,
            message_thread_id=message.message_thread_id,
        )

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

    def _progress_text_from_codex_event(self, event: dict[str, object]) -> str | None:
        item = event.get("item")
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        event_type = event.get("type")
        if item_type != "command_execution":
            return None
        if event_type == "item.started":
            return sanitize_progress_text("Working: running a local command for this topic.")
        if event_type == "item.completed":
            exit_code = item.get("exit_code")
            if exit_code == 0:
                return sanitize_progress_text("Working: local command finished.")
            return sanitize_progress_text("Working: local command failed; the final answer will summarize it.")
        return None

    def _progress_callback_for_message(self, message: IncomingMessage):
        last_sent_at = 0.0
        last_text = ""

        def callback(event: dict[str, object]) -> None:
            nonlocal last_sent_at, last_text
            text = self._progress_text_from_codex_event(event)
            if text is None or text == last_text:
                return
            now = time.monotonic()
            if last_sent_at and now - last_sent_at < PROGRESS_MIN_INTERVAL_SECONDS:
                return
            try:
                self._telegram.send_message(
                    message.chat_id,
                    text,
                    reply_to_message_id=message.message_id,
                    message_thread_id=message.message_thread_id,
                )
            except Exception:
                logger.warning("Failed to send Telegram progress message", exc_info=True)
                return
            last_sent_at = now
            last_text = text

        return callback

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
        if command == "/goal":
            self._handle_goal_command(message)
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

        forum_status = self._general_forum_status(message)
        if forum_status is None:
            self._telegram.send_message(
                message.chat_id,
                "I could not verify whether this group has Telegram topics enabled. Try again in a moment.",
                reply_to_message_id=message.message_id,
            )
            return
        if forum_status:
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
                progress_callback=self._progress_callback_for_message(message),
            )
        finally:
            self._active_session_keys.discard(session_key)
        response = self._telegram_response_text(result.text)
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
                self._sessions.clear_goal(session_key)
                self._telegram.answer_callback_query(callback.callback_query_id, text="Session started.")
                status = self.status(session_key)
                self._reply_to_callback(
                    callback,
                    f"Session workspace:\n{path}\n\n"
                    f"Model: {status.model}\n"
                    f"Thinking: {status.reasoning_effort}\n"
                    f"Sandbox: {status.sandbox}\n"
                    f"Fast mode: {status.fast_mode}\n"
                    f"Goal: {status.goal}",
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
