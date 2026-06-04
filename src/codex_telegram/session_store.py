from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    role: Role
    text: str


@dataclass(frozen=True)
class ChatModelPreference:
    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class TopicSession:
    chat_id: str
    message_thread_id: int
    session_key: str
    topic_name: str
    workspace: str
    model: str
    reasoning_effort: str
    sandbox_mode: str
    is_closed: bool


def _safe_chat_key(chat_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id).strip("_") or "chat"


class SessionStore:
    def __init__(self, state_dir: Path, *, history_turns: int = 8):
        self._state_dir = state_dir
        self._history_turns = history_turns
        self._sessions_dir = state_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._processed_path = state_dir / "processed_messages.json"
        self._preferences_path = state_dir / "model_preferences.json"
        self._sandbox_preferences_path = state_dir / "sandbox_preferences.json"
        self._workspaces_path = state_dir / "workspaces.json"
        self._workspace_tokens_path = state_dir / "workspace_tokens.json"
        self._topic_sessions_path = state_dir / "topic_sessions.json"

    def _path(self, chat_id: str) -> Path:
        return self._sessions_dir / f"{_safe_chat_key(chat_id)}.json"

    def load(self, chat_id: str) -> list[ChatTurn]:
        path = self._path(chat_id)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        turns: list[ChatTurn] = []
        if not isinstance(raw, list):
            return []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            text = item.get("text")
            if role in {"user", "assistant"} and isinstance(text, str):
                turns.append(ChatTurn(role=role, text=text))
        return turns

    def append(self, chat_id: str, role: Role, text: str) -> None:
        turns = self.load(chat_id)
        turns.append(ChatTurn(role=role, text=text))
        self.save(chat_id, turns[-self._history_turns * 2 :])

    def save(self, chat_id: str, turns: list[ChatTurn]) -> None:
        path = self._path(chat_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps([turn.__dict__ for turn in turns], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def reset(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)

    def render_recent(self, chat_id: str) -> str:
        turns = self.load(chat_id)[-self._history_turns * 2 :]
        if not turns:
            return ""
        lines = ["Recent Telegram conversation context:"]
        for turn in turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.text}")
        return "\n".join(lines)

    def _load_processed(self) -> list[str]:
        if not self._processed_path.exists():
            return []
        raw = json.loads(self._processed_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if isinstance(item, str)]

    def _save_processed(self, keys: list[str]) -> None:
        self._processed_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._processed_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(keys[-1000:], indent=2), encoding="utf-8")
        tmp.replace(self._processed_path)

    @staticmethod
    def processed_key(chat_id: str, message_id: int | None, update_id: int) -> str:
        stable_id = f"message:{message_id}" if message_id is not None else f"update:{update_id}"
        return f"{chat_id}:{stable_id}"

    def was_processed(self, chat_id: str, message_id: int | None, update_id: int) -> bool:
        key = self.processed_key(chat_id, message_id, update_id)
        return key in set(self._load_processed())

    def mark_processed(self, chat_id: str, message_id: int | None, update_id: int) -> None:
        key = self.processed_key(chat_id, message_id, update_id)
        keys = self._load_processed()
        if key in keys:
            return
        keys.append(key)
        self._save_processed(keys)

    def _load_preferences(self) -> dict[str, dict[str, str]]:
        if not self._preferences_path.exists():
            return {}
        raw = json.loads(self._preferences_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, str]] = {}
        for chat_id, value in raw.items():
            if not isinstance(chat_id, str) or not isinstance(value, dict):
                continue
            model = value.get("model")
            effort = value.get("reasoning_effort")
            if isinstance(model, str) and isinstance(effort, str):
                out[chat_id] = {"model": model, "reasoning_effort": effort}
        return out

    def _save_preferences(self, preferences: dict[str, dict[str, str]]) -> None:
        self._preferences_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._preferences_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(preferences, indent=2), encoding="utf-8")
        tmp.replace(self._preferences_path)

    def load_model_preference(self, chat_id: str) -> ChatModelPreference | None:
        raw = self._load_preferences().get(chat_id)
        if raw is None:
            return None
        return ChatModelPreference(model=raw["model"], reasoning_effort=raw["reasoning_effort"])

    def save_model_preference(self, chat_id: str, *, model: str, reasoning_effort: str) -> None:
        preferences = self._load_preferences()
        preferences[chat_id] = {"model": model, "reasoning_effort": reasoning_effort}
        self._save_preferences(preferences)

    def clear_model_preference(self, chat_id: str) -> None:
        preferences = self._load_preferences()
        if chat_id not in preferences:
            return
        del preferences[chat_id]
        self._save_preferences(preferences)

    def load_sandbox_mode(self, chat_id: str) -> str | None:
        mode = self._load_string_map(self._sandbox_preferences_path).get(chat_id)
        return mode if mode in {"constrained", "yolo"} else None

    def save_sandbox_mode(self, chat_id: str, mode: str) -> None:
        if mode not in {"constrained", "yolo"}:
            raise ValueError(f"Unsupported sandbox mode: {mode}")
        preferences = self._load_string_map(self._sandbox_preferences_path)
        preferences[chat_id] = mode
        self._save_string_map(self._sandbox_preferences_path, preferences)

    def clear_sandbox_mode(self, chat_id: str) -> None:
        preferences = self._load_string_map(self._sandbox_preferences_path)
        if chat_id not in preferences:
            return
        del preferences[chat_id]
        self._save_string_map(self._sandbox_preferences_path, preferences)

    def _load_string_map(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(key): value for key, value in raw.items() if isinstance(value, str)}

    def _save_string_map(self, path: Path, values: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(values, indent=2), encoding="utf-8")
        tmp.replace(path)

    def load_active_workspace(self, chat_id: str) -> str:
        return self._load_string_map(self._workspaces_path).get(chat_id, "")

    def save_active_workspace(self, chat_id: str, relative_path: str) -> None:
        workspaces = self._load_string_map(self._workspaces_path)
        workspaces[chat_id] = relative_path
        self._save_string_map(self._workspaces_path, workspaces)

    def clear_active_workspace(self, chat_id: str) -> None:
        workspaces = self._load_string_map(self._workspaces_path)
        if chat_id not in workspaces:
            return
        del workspaces[chat_id]
        self._save_string_map(self._workspaces_path, workspaces)

    def remember_workspace_token(self, relative_path: str) -> str:
        token = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
        tokens = self._load_string_map(self._workspace_tokens_path)
        tokens[token] = relative_path
        self._save_string_map(self._workspace_tokens_path, tokens)
        return token

    def resolve_workspace_token(self, token: str) -> str | None:
        return self._load_string_map(self._workspace_tokens_path).get(token)

    def _load_topic_sessions(self) -> dict[str, dict[str, object]]:
        if not self._topic_sessions_path.exists():
            return {}
        raw = json.loads(self._topic_sessions_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}

    def _save_topic_sessions(self, sessions: dict[str, dict[str, object]]) -> None:
        self._topic_sessions_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._topic_sessions_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
        tmp.replace(self._topic_sessions_path)

    def save_topic_session(
        self,
        *,
        chat_id: str,
        message_thread_id: int,
        session_key: str,
        topic_name: str,
        workspace: str,
        model: str,
        reasoning_effort: str,
        sandbox_mode: str,
    ) -> None:
        sessions = self._load_topic_sessions()
        sessions[session_key] = {
            "chat_id": chat_id,
            "message_thread_id": message_thread_id,
            "session_key": session_key,
            "topic_name": topic_name,
            "workspace": workspace,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "sandbox_mode": sandbox_mode,
            "is_closed": False,
        }
        self._save_topic_sessions(sessions)

    def load_topic_session(self, session_key: str) -> TopicSession | None:
        raw = self._load_topic_sessions().get(session_key)
        if raw is None:
            return None
        chat_id = raw.get("chat_id")
        message_thread_id = raw.get("message_thread_id")
        topic_name = raw.get("topic_name")
        workspace = raw.get("workspace")
        model = raw.get("model")
        reasoning_effort = raw.get("reasoning_effort")
        sandbox_mode = raw.get("sandbox_mode")
        is_closed = raw.get("is_closed", False)
        if not (
            isinstance(chat_id, str)
            and isinstance(message_thread_id, int)
            and isinstance(topic_name, str)
            and isinstance(workspace, str)
            and isinstance(model, str)
            and isinstance(reasoning_effort, str)
            and isinstance(sandbox_mode, str)
            and isinstance(is_closed, bool)
        ):
            return None
        return TopicSession(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            session_key=session_key,
            topic_name=topic_name,
            workspace=workspace,
            model=model,
            reasoning_effort=reasoning_effort,
            sandbox_mode=sandbox_mode,
            is_closed=is_closed,
        )

    def list_topic_sessions(self, chat_id: str | None = None) -> list[TopicSession]:
        sessions = []
        for session_key in self._load_topic_sessions():
            session = self.load_topic_session(session_key)
            if session is None:
                continue
            if chat_id is not None and session.chat_id != chat_id:
                continue
            sessions.append(session)
        return sorted(sessions, key=lambda session: session.topic_name.lower())

    def update_topic_session_name(self, session_key: str, topic_name: str) -> bool:
        sessions = self._load_topic_sessions()
        session = sessions.get(session_key)
        if not isinstance(session, dict):
            return False
        session["topic_name"] = topic_name
        self._save_topic_sessions(sessions)
        return True

    def set_topic_session_closed(self, session_key: str, is_closed: bool) -> bool:
        sessions = self._load_topic_sessions()
        session = sessions.get(session_key)
        if not isinstance(session, dict):
            return False
        session["is_closed"] = is_closed
        self._save_topic_sessions(sessions)
        return True

    def remove_topic_session(self, session_key: str) -> bool:
        sessions = self._load_topic_sessions()
        existed = session_key in sessions
        sessions.pop(session_key, None)
        self._save_topic_sessions(sessions)
        self.reset(session_key)
        self.clear_active_workspace(session_key)
        self.clear_model_preference(session_key)
        self.clear_sandbox_mode(session_key)
        return existed
