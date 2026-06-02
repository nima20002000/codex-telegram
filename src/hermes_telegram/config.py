from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is not None:
            key, value = parsed
            values[key] = value
    return values


def _csv_ints(value: str) -> frozenset[int]:
    out: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.add(int(raw))
    return frozenset(out)


def _csv_strings(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    bot_token: str
    allowed_users: frozenset[int]
    allowed_chats: frozenset[str]
    codex_command: str
    codex_workdir: Path
    codex_model: str
    codex_profile: str
    codex_sandbox: str
    codex_extra_args: tuple[str, ...]
    codex_timeout_seconds: int
    telegram_poll_timeout_seconds: int
    telegram_request_timeout_seconds: int
    max_telegram_response_chars: int
    session_history_turns: int
    state_dir: Path

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | None = None,
        environ: Mapping[str, str] | None = None,
        default_workdir: Path | None = None,
    ) -> "Settings":
        base_env = dict(os.environ if environ is None else environ)
        file_values: dict[str, str] = {}
        if env_file is not None:
            file_values = load_env_file(env_file)
        env = {**file_values, **base_env}

        token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        workdir = Path(env.get("CODEX_WORKDIR", "").strip() or default_workdir or Path.cwd())
        workdir = workdir.expanduser().resolve()

        state_dir = Path(env.get("HERMES_TELEGRAM_STATE_DIR", "").strip() or workdir / ".hermes-telegram")
        state_dir = state_dir.expanduser().resolve()

        return cls(
            bot_token=token,
            allowed_users=_csv_ints(env.get("TELEGRAM_ALLOWED_USERS", "")),
            allowed_chats=_csv_strings(env.get("TELEGRAM_ALLOWED_CHATS", "")),
            codex_command=env.get("CODEX_COMMAND", "codex").strip() or "codex",
            codex_workdir=workdir,
            codex_model=env.get("CODEX_MODEL", "").strip(),
            codex_profile=env.get("CODEX_PROFILE", "").strip(),
            codex_sandbox=env.get("CODEX_SANDBOX", "workspace-write").strip() or "workspace-write",
            codex_extra_args=tuple(shlex.split(env.get("CODEX_EXTRA_ARGS", ""))),
            codex_timeout_seconds=_int_env(env, "CODEX_TIMEOUT_SECONDS", 1800),
            telegram_poll_timeout_seconds=_int_env(env, "TELEGRAM_POLL_TIMEOUT_SECONDS", 30),
            telegram_request_timeout_seconds=_int_env(env, "TELEGRAM_REQUEST_TIMEOUT_SECONDS", 45),
            max_telegram_response_chars=_int_env(env, "MAX_TELEGRAM_RESPONSE_CHARS", 12000),
            session_history_turns=_int_env(env, "SESSION_HISTORY_TURNS", 8),
            state_dir=state_dir,
        )

    def validate(self) -> None:
        if not self.codex_workdir.is_dir():
            raise ValueError(f"CODEX_WORKDIR does not exist: {self.codex_workdir}")
        if self.codex_sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"Unsupported CODEX_SANDBOX: {self.codex_sandbox}")
