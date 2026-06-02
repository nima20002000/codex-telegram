from __future__ import annotations

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


def _safe_chat_key(chat_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id).strip("_") or "chat"


class SessionStore:
    def __init__(self, state_dir: Path, *, history_turns: int = 8):
        self._state_dir = state_dir
        self._history_turns = history_turns
        self._sessions_dir = state_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._processed_path = state_dir / "processed_messages.json"

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
