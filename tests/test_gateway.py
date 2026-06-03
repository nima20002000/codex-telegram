from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_telegram.codex_runner import CodexResult
from hermes_telegram.config import Settings
from hermes_telegram.gateway import HermesTelegramGateway
from hermes_telegram.model_catalog import ModelChoice
from hermes_telegram.session_store import SessionStore
from hermes_telegram.telegram_api import IncomingCallback, IncomingMessage


class FakeTelegram:
    def __init__(self, updates=None):
        self.messages: list[tuple[str, str, int | None]] = []
        self.reply_markups: list[dict | None] = []
        self.edits: list[tuple[str, int, str, dict | None]] = []
        self.callback_answers: list[tuple[str, str | None]] = []
        self.actions: list[str] = []
        self.updates = list(updates or [])
        self.calls: list[tuple[int | None, int]] = []
        self.fail_chat_action = False

    def send_message(self, chat_id, text, *, reply_to_message_id=None, reply_markup=None):
        self.messages.append((chat_id, text, reply_to_message_id))
        self.reply_markups.append(reply_markup)

    def edit_message_text(self, chat_id, message_id, text, *, reply_markup=None):
        self.edits.append((chat_id, message_id, text, reply_markup))

    def answer_callback_query(self, callback_query_id, *, text=None):
        self.callback_answers.append((callback_query_id, text))

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
        self.runs: list[tuple[str | None, str | None, Path | None]] = []
        self.response = response

    def run(self, prompt, *, model=None, reasoning_effort=None, workdir=None):
        self.prompts.append(prompt)
        self.runs.append((model, reasoning_effort, workdir))
        return CodexResult(text=self.response, returncode=0, stderr="")


class FakeModelCatalog:
    def __init__(self):
        self.models = (
            ModelChoice("gpt-5.5", "GPT-5.5", ("low", "medium", "high", "xhigh"), "medium"),
            ModelChoice("gpt-5.4-mini", "GPT-5.4-Mini", ("low", "medium", "high"), "medium"),
        )

    def list_models(self):
        return self.models

    def get_model(self, slug):
        for model in self.models:
            if model.slug == slug:
                return model
        return None

    def is_authoritative(self):
        return True


class NonAuthoritativeModelCatalog(FakeModelCatalog):
    def is_authoritative(self):
        return False


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

    def _callback(self, data: str, *, user_id=42) -> IncomingCallback:
        return IncomingCallback(
            update_id=2,
            callback_query_id="cb1",
            chat_id="100",
            user_id=user_id,
            username="nima",
            data=data,
            message_id=10,
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
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/help"))

            self.assertEqual(len(codex.prompts), 0)
            self.assertEqual(telegram.messages[0][1], "/reset\n/models\n/workspace")

    def test_status_command_shows_status(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/status"))

            self.assertEqual(len(codex.prompts), 0)
            self.assertIn(f"Workspace: {Path(tmp).resolve()}", telegram.messages[0][1])
            self.assertIn("Model: default", telegram.messages[0][1])

    def test_workspace_command_shows_folder_buttons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "avatar").mkdir()
            (root / "beta").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(self._message("/workspace"))

            self.assertEqual(len(codex.prompts), 0)
            self.assertEqual(telegram.messages[0][1], f"Workspace:\n{root.resolve()}")
            keyboard = telegram.reply_markups[0]["inline_keyboard"]
            self.assertEqual(keyboard[0][0]["text"], "Start session")
            self.assertEqual([row[0]["text"] for row in keyboard[1:]], ["avatar", "beta"])

    def test_workspace_command_starts_from_root_even_after_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "avatar").mkdir()
            store = SessionStore(root / ".state")
            store.save_active_workspace("100", "avatar")
            telegram = FakeTelegram()
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/workspace"))

            self.assertEqual(telegram.messages[0][1], f"Workspace:\n{root.resolve()}")

    def test_workspace_command_skips_external_symlinked_directories(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "desktop"
            root.mkdir()
            (root / "avatar").mkdir()
            external = base / "external"
            external.mkdir()
            (root / "outside").symlink_to(external, target_is_directory=True)
            telegram = FakeTelegram()
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(self._message("/workspace"))

            keyboard = telegram.reply_markups[0]["inline_keyboard"]
            self.assertEqual([row[0]["text"] for row in keyboard[1:]], ["avatar"])

    def test_workspace_folder_button_navigates_deeper(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            avatar = root / "avatar"
            avatar.mkdir()
            (avatar / "child").mkdir()
            telegram = FakeTelegram()
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(self._message("/workspace"))
            avatar_button = telegram.reply_markups[0]["inline_keyboard"][1][0]
            gateway.handle_callback(self._callback(avatar_button["callback_data"]))

            self.assertEqual(telegram.edits[0][2], f"Workspace:\n{avatar.resolve()}")
            keyboard = telegram.edits[0][3]["inline_keyboard"]
            self.assertEqual([row[0]["text"] for row in keyboard], ["Start session", "child", "Back"])

    def test_start_workspace_session_sets_workdir_and_default_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            avatar = root / "avatar"
            avatar.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(root / ".state")
            store.save_model_preference("100", model="gpt-5.5", reasoning_effort="xhigh")
            store.append("100", "user", "old context")
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/workspace"))
            avatar_button = telegram.reply_markups[0]["inline_keyboard"][1][0]
            gateway.handle_callback(self._callback(avatar_button["callback_data"]))
            start_button = telegram.edits[0][3]["inline_keyboard"][0][0]
            gateway.handle_callback(self._callback(start_button["callback_data"]))
            gateway.handle_message(self._message("do the task"))

            self.assertEqual(store.load_active_workspace("100"), "avatar")
            self.assertIsNone(store.load_model_preference("100"))
            self.assertIn(f"Session workspace:\n{avatar.resolve()}", telegram.edits[1][2])
            self.assertEqual(codex.runs[-1], (None, None, avatar.resolve()))
            self.assertNotIn("old context", codex.prompts[-1])

    def test_workspace_callback_rejects_path_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            token = store.remember_workspace_token("..")
            gateway = HermesTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(self._callback(f"ws:o:{token}"))

            self.assertEqual(telegram.callback_answers, [("cb1", "Workspace is not available.")])
            self.assertIn("Workspace is not available", telegram.edits[0][2])

    def test_authorized_message_runs_codex_and_stores_history(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("change the repo"))

            self.assertEqual(telegram.actions, ["100:typing"])
            self.assertEqual(telegram.messages[-1][1], "finished")
            self.assertEqual(codex.runs[-1], (None, None, Path(tmp).resolve()))
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
                model_catalog=FakeModelCatalog(),
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
                model_catalog=FakeModelCatalog(),
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
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("hello", user_id=99))

            self.assertEqual(codex.prompts, [])
            self.assertEqual(telegram.messages, [])

    def test_models_command_sends_inline_model_buttons(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/models"))

            self.assertEqual(codex.prompts, [])
            self.assertIn("Choose a Codex model", telegram.messages[0][1])
            self.assertEqual(
                telegram.reply_markups[0]["inline_keyboard"][0][0],
                {"text": "GPT-5.5", "callback_data": "model:gpt-5.5"},
            )

    def test_model_callback_shows_reasoning_buttons(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_callback(self._callback("model:gpt-5.5"))

            self.assertEqual(telegram.callback_answers, [("cb1", None)])
            self.assertIn("Choose thinking amount", telegram.edits[0][2])
            self.assertEqual(
                telegram.edits[0][3]["inline_keyboard"][0][-1],
                {"text": "X High", "callback_data": "effort:gpt-5.5:xhigh"},
            )

    def test_effort_callback_saves_selection_and_next_message_uses_it(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(self._callback("effort:gpt-5.5:xhigh"))
            gateway.handle_message(self._message("do the task"))

            preference = store.load_model_preference("100")
            self.assertIsNotNone(preference)
            self.assertEqual(preference.model, "gpt-5.5")
            self.assertEqual(preference.reasoning_effort, "xhigh")
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "xhigh", Path(tmp).resolve()))
            self.assertIn("Selected GPT-5.5", telegram.edits[0][2])

    def test_unavailable_saved_model_is_ignored_and_cleared(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            store.save_model_preference("100", model="gpt-5.2", reasoning_effort="medium")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do the task"))

            self.assertEqual(codex.runs[-1], (None, None, Path(tmp).resolve()))
            self.assertIsNone(store.load_model_preference("100"))

    def test_saved_model_is_kept_when_catalog_is_not_authoritative(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            store.save_model_preference("100", model="gpt-5.2", reasoning_effort="medium")
            gateway = HermesTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=NonAuthoritativeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do the task"))

            self.assertEqual(codex.runs[-1], ("gpt-5.2", "medium", Path(tmp).resolve()))
            self.assertIsNotNone(store.load_model_preference("100"))

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
                model_catalog=FakeModelCatalog(),
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
