from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_telegram.codex_runner import CodexResult
from hermes_telegram.config import Settings
from hermes_telegram.gateway import HermesTelegramGateway
from hermes_telegram.session_store import SessionStore
from hermes_telegram.telegram_api import IncomingMessage


class FakeTelegram:
    def __init__(self, updates=None):
        self.messages: list[tuple[str, str, int | None]] = []
        self.actions: list[str] = []
        self.updates = list(updates or [])
        self.calls: list[tuple[int | None, int]] = []
        self.fail_chat_action = False

    def send_message(self, chat_id, text, *, reply_to_message_id=None):
        self.messages.append((chat_id, text, reply_to_message_id))

    def send_chat_action(self, chat_id, action="typing"):
        if self.fail_chat_action:
            raise RuntimeError("typing failed")
        self.actions.append(f"{chat_id}:{action}")

    def get_updates(self, *, offset, timeout):
        self.calls.append((offset, timeout))
        return list(self.updates)


class FakeCodex:
    def __init__(self, response="done"):
        self.prompts: list[str] = []
        self.response = response

    def run(self, prompt):
        self.prompts.append(prompt)
        return CodexResult(text=self.response, returncode=0, stderr="")


class GatewayTests(unittest.TestCase):
    def _settings(self, workdir: Path, *, allowed_users=frozenset({42})) -> Settings:
        return Settings(
            bot_token="token",
            allowed_users=allowed_users,
            allowed_chats=frozenset(),
            codex_command="codex",
            codex_workdir=workdir,
            codex_model="",
            codex_profile="",
            codex_sandbox="workspace-write",
            codex_extra_args=(),
            codex_timeout_seconds=10,
            telegram_poll_timeout_seconds=30,
            telegram_request_timeout_seconds=45,
            max_telegram_response_chars=12000,
            session_history_turns=8,
            state_dir=workdir / ".state",
        )

    def _message(self, text: str, *, user_id=42) -> IncomingMessage:
        return IncomingMessage(
            update_id=1,
            chat_id="100",
            user_id=user_id,
            username="nima",
            text=text,
            message_id=9,
            chat_type="private",
        )

    def test_help_command_does_not_run_codex(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/help"))

            self.assertEqual(len(codex.prompts), 0)
            self.assertIn("online", telegram.messages[0][1])

    def test_authorized_message_runs_codex_and_stores_history(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                sessions=store,
            )

            gateway.handle_message(self._message("change the repo"))

            self.assertEqual(telegram.actions, ["100:typing"])
            self.assertEqual(telegram.messages[-1][1], "finished")
            self.assertIn("Current Telegram message", codex.prompts[0])
            self.assertNotIn("User: change the repo", codex.prompts[0])
            self.assertEqual([turn.role for turn in store.load("100")], ["user", "assistant"])

    def test_typing_indicator_failure_does_not_drop_request(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            telegram.fail_chat_action = True
            codex = FakeCodex("finished")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("do it"))

            self.assertEqual(len(codex.prompts), 1)
            self.assertEqual(telegram.messages[-1][1], "finished")

    def test_duplicate_message_does_not_rerun_codex(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                sessions=SessionStore(Path(tmp) / "state"),
            )

            message = self._message("change the repo")
            gateway.handle_message(message)
            gateway.handle_update(
                {
                    "update_id": message.update_id,
                    "message": {
                        "message_id": message.message_id,
                        "text": message.text,
                        "chat": {"id": message.chat_id, "type": "private"},
                        "from": {"id": message.user_id, "username": message.username},
                    },
                }
            )
            gateway.handle_update(
                {
                    "update_id": message.update_id,
                    "message": {
                        "message_id": message.message_id,
                        "text": message.text,
                        "chat": {"id": message.chat_id, "type": "private"},
                        "from": {"id": message.user_id, "username": message.username},
                    },
                }
            )

            self.assertEqual(len(codex.prompts), 2)

    def test_unauthorized_message_is_ignored(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("hello", user_id=99))

            self.assertEqual(codex.prompts, [])
            self.assertEqual(telegram.messages, [])

    def test_poll_once_advances_offset_when_handler_raises(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram(
                [
                    {
                        "update_id": 50,
                        "message": {
                            "message_id": 9,
                            "text": "hello",
                            "chat": {"id": 100, "type": "private"},
                            "from": {"id": 42, "username": "nima"},
                        },
                    }
                ]
            )
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=FakeCodex(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            def fail(_update):
                raise RuntimeError("send failed")

            gateway.handle_update = fail

            with self.assertRaisesRegex(RuntimeError, "send failed"):
                gateway.poll_once()

            self.assertEqual(gateway._offset, 51)


if __name__ == "__main__":
    unittest.main()
