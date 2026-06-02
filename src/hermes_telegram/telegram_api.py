from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class TelegramAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class IncomingMessage:
    update_id: int
    chat_id: str
    user_id: int | None
    username: str
    text: str
    message_id: int | None
    chat_type: str


def parse_message_update(update: dict[str, Any]) -> IncomingMessage | None:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    text = message.get("text") or message.get("caption") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None
    user_id = sender.get("id") if isinstance(sender.get("id"), int) else None
    username = sender.get("username") or sender.get("first_name") or ""
    return IncomingMessage(
        update_id=int(update["update_id"]),
        chat_id=str(chat_id),
        user_id=user_id,
        username=str(username),
        text=text.strip(),
        message_id=message.get("message_id"),
        chat_type=str(chat.get("type") or ""),
    )


def split_telegram_text(text: str, *, limit: int = 4096) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunk = remaining[:limit]
        split_at = max(chunk.rfind("\n"), chunk.rfind(" "))
        if split_at > limit // 2:
            chunk = chunk[: split_at + 1]
        chunks.append(chunk)
        remaining = remaining[len(chunk) :]
    return chunks


class TelegramAPI:
    def __init__(self, token: str, *, request_timeout_seconds: int = 45):
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._request_timeout_seconds = request_timeout_seconds

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        data = urllib.parse.urlencode(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in params.items()
                if value is not None
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/{method}",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TelegramAPIError(f"Telegram {method} failed: HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise TelegramAPIError(f"Telegram {method} failed: {exc.reason}") from exc
        if not payload.get("ok"):
            raise TelegramAPIError(f"Telegram {method} failed: {payload}")
        return payload

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        payload = self._request(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message"],
            },
        )
        result = payload.get("result", [])
        return result if isinstance(result, list) else []

    def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        for chunk in split_telegram_text(text):
            self._request(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "reply_to_message_id": reply_to_message_id,
                    "disable_web_page_preview": True,
                },
            )
            reply_to_message_id = None

    def send_chat_action(self, chat_id: str, action: str = "typing") -> None:
        self._request("sendChatAction", {"chat_id": chat_id, "action": action})
