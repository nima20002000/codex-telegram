from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_env_file


DEFAULT_GROUP_TITLE = "Codex Telegram E2E"
DEFAULT_SERVICE_NAME = "codex-telegram.service"
DEFAULT_SESSION_BASE = ".codex-telegram/e2e/admin-account"


@dataclass(frozen=True)
class TelegramAppCredentials:
    api_id: int
    api_hash: str


@dataclass(frozen=True)
class BotIdentity:
    user_id: int
    username: str
    first_name: str


@dataclass(frozen=True)
class ServiceStatus:
    active_state: str
    sub_state: str
    main_pid: str
    working_directory: Path | None
    branch: str | None
    commit: str | None


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


class E2EHarnessError(RuntimeError):
    pass


def parse_app_credentials(text: str) -> TelegramAppCredentials:
    api_id_match = re.search(r"App api_id\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    api_hash_match = re.search(r"App api_hash\s*[:=]\s*([0-9a-fA-F]+)", text, flags=re.IGNORECASE)
    if api_id_match is None or api_hash_match is None:
        raise ValueError("Could not find App api_id and App api_hash")
    return TelegramAppCredentials(api_id=int(api_id_match.group(1)), api_hash=api_hash_match.group(1))


def parse_allowed_users(value: str) -> frozenset[int]:
    out: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if item:
            out.add(int(item))
    return frozenset(out)


def parse_allowed_chats(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def bot_api_chat_id(entity: Any) -> str | None:
    raw_id = getattr(entity, "id", None)
    if not isinstance(raw_id, int):
        return None
    if raw_id < 0:
        return str(raw_id)
    if getattr(entity, "megagroup", False) or entity.__class__.__name__ == "Channel":
        return f"-100{raw_id}"
    return str(raw_id)


def sanitize_detail(text: str, *, token: str | None = None, api_hash: str | None = None) -> str:
    sanitized = text
    for secret in (token, api_hash):
        if secret:
            sanitized = sanitized.replace(secret, "<redacted>")
    sanitized = re.sub(r"-100\d{6,}", "<chat>", sanitized)
    sanitized = re.sub(r"\b\d{8,}\b", "<id>", sanitized)
    return sanitized


def format_check(result: CheckResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    return f"[{status}] {result.name}: {result.detail}"


def bot_lookup_reference(identity: BotIdentity) -> str | int:
    if identity.username:
        return f"@{identity.username.lstrip('@')}"
    if identity.user_id:
        return identity.user_id
    raise ValueError("Bot identity did not include a username or id")


def message_replies_to(message: Any, message_id: int) -> bool:
    direct = getattr(message, "reply_to_msg_id", None)
    if direct == message_id:
        return True
    reply = getattr(message, "reply_to", None)
    return getattr(reply, "reply_to_msg_id", None) == message_id


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise E2EHarnessError((completed.stderr or completed.stdout).strip() or f"{args[0]} failed")
    return completed.stdout.strip()


def _parse_systemctl_show(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def get_service_status(service_name: str) -> ServiceStatus:
    output = _run(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
            "-p",
            "WorkingDirectory",
        ]
    )
    values = _parse_systemctl_show(output)
    workdir_raw = values.get("WorkingDirectory", "").strip()
    workdir = Path(workdir_raw).expanduser().resolve() if workdir_raw else None
    branch: str | None = None
    commit: str | None = None
    if workdir is not None and (workdir / ".git").exists():
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=workdir)
        commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=workdir)
    return ServiceStatus(
        active_state=values.get("ActiveState", ""),
        sub_state=values.get("SubState", ""),
        main_pid=values.get("MainPID", ""),
        working_directory=workdir,
        branch=branch,
        commit=commit,
    )


def check_service(
    status: ServiceStatus,
    *,
    expected_workdir: Path | None = None,
    expected_branch: str | None = None,
    expected_commit: str | None = None,
) -> list[CheckResult]:
    results = [
        CheckResult(
            "service running",
            status.active_state == "active" and status.sub_state == "running",
            f"{status.active_state}/{status.sub_state} pid={status.main_pid or 'unknown'}",
        )
    ]
    if expected_workdir is not None:
        resolved = expected_workdir.expanduser().resolve()
        results.append(
            CheckResult(
                "service checkout",
                status.working_directory == resolved,
                f"workdir={status.working_directory or 'unknown'}",
            )
        )
    elif status.working_directory is not None:
        results.append(CheckResult("service checkout", True, f"workdir={status.working_directory}"))
    if expected_branch is not None:
        results.append(
            CheckResult(
                "service branch",
                status.branch == expected_branch,
                f"branch={status.branch or 'unknown'}",
            )
        )
    elif status.branch is not None:
        results.append(CheckResult("service branch", True, f"branch={status.branch}"))
    if expected_commit is not None:
        results.append(
            CheckResult(
                "service commit",
                status.commit == expected_commit,
                f"commit={status.commit or 'unknown'}",
            )
        )
    elif status.commit is not None:
        results.append(CheckResult("service commit", True, f"commit={status.commit}"))
    return results


def get_bot_identity(token: str, *, timeout_seconds: int = 30) -> BotIdentity:
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/getMe", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise E2EHarnessError(f"Bot API getMe failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise E2EHarnessError(f"Bot API getMe failed: {exc.reason}") from exc
    if not payload.get("ok") or not isinstance(payload.get("result"), dict):
        raise E2EHarnessError("Bot API getMe returned an unexpected response")
    result = payload["result"]
    return BotIdentity(
        user_id=int(result.get("id") or 0),
        username=str(result.get("username") or ""),
        first_name=str(result.get("first_name") or ""),
    )


async def run_telegram_checks(
    *,
    api_credentials: TelegramAppCredentials,
    session_base: Path,
    env_values: dict[str, str],
    group_title: str,
    bot_identity: BotIdentity,
    marker: bool,
    marker_timeout_seconds: int,
) -> list[CheckResult]:
    try:
        from telethon import TelegramClient
        from telethon.tl.functions.channels import GetParticipantRequest
    except ImportError as exc:
        raise E2EHarnessError("Telethon is not installed in this Python environment") from exc

    results: list[CheckResult] = []
    client = TelegramClient(str(session_base), api_credentials.api_id, api_credentials.api_hash)
    try:
        await client.connect()
    except Exception as exc:  # Telethon may raise sqlite OperationalError for locked sessions.
        raise E2EHarnessError(f"Could not open Telethon session: {exc}") from exc
    try:
        authorized = await client.is_user_authorized()
        results.append(CheckResult("Telethon session authorized", authorized, "authorized" if authorized else "not authorized"))
        if not authorized:
            return results

        admin = await client.get_me()
        admin_id = int(admin.id)
        allowed = parse_allowed_users(env_values.get("TELEGRAM_ALLOWED_USERS", ""))
        if allowed:
            results.append(
                CheckResult(
                    "admin account allowlisted",
                    admin_id in allowed,
                    "admin account is allowed" if admin_id in allowed else "admin account is not in TELEGRAM_ALLOWED_USERS",
                )
            )
        else:
            results.append(
                CheckResult(
                    "admin account allowlisted",
                    True,
                    "TELEGRAM_ALLOWED_USERS is empty; gateway allows any user",
                )
            )

        entity = None
        async for dialog in client.iter_dialogs():
            if dialog.name == group_title:
                entity = dialog.entity
                break
        results.append(
            CheckResult(
                "target forum group found",
                entity is not None,
                f"group title={group_title!r}" if entity is not None else f"group title={group_title!r} not found",
            )
        )
        if entity is None:
            return results

        allowed_chats = parse_allowed_chats(env_values.get("TELEGRAM_ALLOWED_CHATS", ""))
        target_chat_id = bot_api_chat_id(entity)
        if allowed_chats:
            results.append(
                CheckResult(
                    "target chat allowlisted",
                    target_chat_id in allowed_chats,
                    "target chat is allowed" if target_chat_id in allowed_chats else "target chat is not in TELEGRAM_ALLOWED_CHATS",
                )
            )
        else:
            results.append(
                CheckResult(
                    "target chat allowlisted",
                    True,
                    "TELEGRAM_ALLOWED_CHATS is empty; gateway does not restrict chats",
                )
            )

        bot_entity = await client.get_entity(bot_lookup_reference(bot_identity))
        bot_user_id = int(bot_entity.id)
        participant = await client(GetParticipantRequest(entity, bot_entity))
        rights = getattr(participant.participant, "admin_rights", None)
        is_admin = rights is not None
        detail = "admin"
        if rights is not None:
            detail = (
                f"admin change_info={bool(getattr(rights, 'change_info', False))} "
                f"manage_topics={bool(getattr(rights, 'manage_topics', False))} "
                f"delete_messages={bool(getattr(rights, 'delete_messages', False))}"
            )
        results.append(CheckResult("bot admin status", is_admin, detail if is_admin else "bot is not admin"))

        if marker:
            sent = await client.send_message(entity, f"/status nim70-marker-{int(time.time())}")
            reply = await _wait_for_bot_reply(
                client,
                entity,
                bot_user_id,
                after_id=int(sent.id),
                timeout_seconds=marker_timeout_seconds,
            )
            results.append(
                CheckResult(
                    "marker message reply",
                    reply is not None,
                    f"reply message id={getattr(reply, 'id', 'unknown')}" if reply is not None else "no bot reply before timeout",
                )
            )
    finally:
        await client.disconnect()
    return results


async def _wait_for_bot_reply(client: Any, entity: Any, bot_id: int, *, after_id: int, timeout_seconds: int) -> Any | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        async for message in client.iter_messages(entity, limit=40):
            if int(getattr(message, "id", 0)) <= after_id:
                continue
            if getattr(message, "sender_id", None) == bot_id:
                if not message_replies_to(message, after_id):
                    continue
                return message
        await asyncio.sleep(1.5)
    return None


def _print_results(results: Iterable[CheckResult], *, token: str | None, api_hash: str | None) -> bool:
    ok = True
    for result in results:
        ok = ok and result.passed
        print(format_check(CheckResult(result.name, result.passed, sanitize_detail(result.detail, token=token, api_hash=api_hash))))
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run redacted Telegram E2E preflight checks for codex-telegram.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Local bot env file.")
    parser.add_argument("--credentials", type=Path, default=Path("telegram-cred.md"), help="Ignored Telegram app credential note.")
    parser.add_argument("--session", type=Path, default=Path(DEFAULT_SESSION_BASE), help="Telethon session path without .session suffix.")
    parser.add_argument("--group", default=DEFAULT_GROUP_TITLE, help="Private forum group title to inspect.")
    parser.add_argument("--service", default=DEFAULT_SERVICE_NAME, help="systemd user service name.")
    parser.add_argument("--skip-service-check", action="store_true", help="Skip systemd service status checks.")
    parser.add_argument("--expected-service-workdir", type=Path, help="Fail if the service runs from a different checkout.")
    parser.add_argument("--expected-service-branch", help="Fail if the service checkout is on a different branch.")
    parser.add_argument("--expected-service-commit", help="Fail if the service checkout is on a different short commit.")
    parser.add_argument("--marker", action="store_true", help="Send a harmless /status marker and wait for the bot reply.")
    parser.add_argument("--marker-timeout-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_values = load_env_file(args.env_file)
    token = env_values.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("[FAIL] env file: TELEGRAM_BOT_TOKEN is missing", file=sys.stderr)
        return 2
    try:
        credentials = parse_app_credentials(args.credentials.read_text(encoding="utf-8"))
        bot_identity = get_bot_identity(token, timeout_seconds=int(env_values.get("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "30") or "30"))
        all_results: list[CheckResult] = [
            CheckResult(
                "Bot API getMe",
                True,
                f"bot=@{bot_identity.username}" if bot_identity.username else f"bot={bot_identity.first_name or 'unknown'}",
            )
        ]
        if not args.skip_service_check:
            all_results.extend(
                check_service(
                    get_service_status(args.service),
                    expected_workdir=args.expected_service_workdir,
                    expected_branch=args.expected_service_branch,
                    expected_commit=args.expected_service_commit,
                )
            )
        all_results.extend(
            asyncio.run(
                run_telegram_checks(
                    api_credentials=credentials,
                    session_base=args.session,
                    env_values=env_values,
                    group_title=args.group,
                    bot_identity=bot_identity,
                    marker=args.marker,
                    marker_timeout_seconds=args.marker_timeout_seconds,
                )
            )
        )
    except Exception as exc:
        print(f"[FAIL] harness: {sanitize_detail(str(exc), token=token)}", file=sys.stderr)
        return 1
    return 0 if _print_results(all_results, token=token, api_hash=credentials.api_hash) else 1


if __name__ == "__main__":
    raise SystemExit(main())
