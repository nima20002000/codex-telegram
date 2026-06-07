from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import codex_telegram.gateway as gateway_module
from codex_telegram.codex_runner import EMPTY_CODEX_RESPONSE, CodexResult
from codex_telegram.config import Settings
from codex_telegram.gateway import (
    CodexTelegramGateway,
    safe_command_progress_preview,
    sanitize_progress_text,
    shape_telegram_response_text,
)
from codex_telegram.model_catalog import ModelChoice
from codex_telegram.session_store import SessionStore
from codex_telegram.telegram_api import ChatInfo, ForumTopic, IncomingCallback, IncomingMessage, TelegramAPIError


class FakeTelegram:
    def __init__(self, updates=None):
        self.messages: list[tuple[str, str, int | None]] = []
        self.message_threads: list[int | None] = []
        self.reply_markups: list[dict | None] = []
        self.created_topics: list[tuple[str, str]] = []
        self.edited_forum_topics: list[tuple[str, int, str]] = []
        self.closed_forum_topics: list[tuple[str, int]] = []
        self.reopened_forum_topics: list[tuple[str, int]] = []
        self.deleted_forum_topics: list[tuple[str, int]] = []
        self.chat_info = ChatInfo(title="Dev Group", chat_type="supergroup", is_forum=True)
        self.group_renames: list[tuple[str, str]] = []
        self.next_thread_id = 50
        self.edits: list[tuple[str, int, str, dict | None]] = []
        self.callback_answers: list[tuple[str, str | None]] = []
        self.actions: list[str] = []
        self.updates = list(updates or [])
        self.calls: list[tuple[int | None, int]] = []
        self.fail_chat_action = False
        self.fail_create_forum_topic = False
        self.fail_topic_lifecycle = False
        self.fail_get_chat = False
        self.fail_group_metadata = False

    def send_message(self, chat_id, text, *, reply_to_message_id=None, message_thread_id=None, reply_markup=None):
        self.messages.append((chat_id, text, reply_to_message_id))
        self.message_threads.append(message_thread_id)
        self.reply_markups.append(reply_markup)

    def create_forum_topic(self, chat_id, name):
        if self.fail_create_forum_topic:
            raise TelegramAPIError("missing manage topics permission")
        self.created_topics.append((chat_id, name))
        topic = ForumTopic(message_thread_id=self.next_thread_id, name=name)
        self.next_thread_id += 1
        return topic

    def edit_forum_topic(self, chat_id, message_thread_id, *, name):
        if self.fail_topic_lifecycle:
            raise TelegramAPIError("missing topic admin permission")
        self.edited_forum_topics.append((chat_id, message_thread_id, name))

    def close_forum_topic(self, chat_id, message_thread_id):
        if self.fail_topic_lifecycle:
            raise TelegramAPIError("missing topic admin permission")
        self.closed_forum_topics.append((chat_id, message_thread_id))

    def reopen_forum_topic(self, chat_id, message_thread_id):
        if self.fail_topic_lifecycle:
            raise TelegramAPIError("missing topic admin permission")
        self.reopened_forum_topics.append((chat_id, message_thread_id))

    def delete_forum_topic(self, chat_id, message_thread_id):
        if self.fail_topic_lifecycle:
            raise TelegramAPIError("missing topic admin permission")
        self.deleted_forum_topics.append((chat_id, message_thread_id))

    def get_chat(self, chat_id):
        if self.fail_get_chat:
            raise TelegramAPIError("temporary getChat failure")
        return self.chat_info

    def set_chat_title(self, chat_id, title):
        if self.fail_group_metadata:
            raise TelegramAPIError("missing group admin permission")
        self.group_renames.append((chat_id, title))

    def edit_message_text(self, chat_id, message_id, text, *, reply_markup=None):
        self.edits.append((chat_id, message_id, text, reply_markup))

    def answer_callback_query(self, callback_query_id, *, text=None):
        self.callback_answers.append((callback_query_id, text))

    def send_chat_action(self, chat_id, action="typing", *, message_thread_id=None):
        if self.fail_chat_action:
            raise RuntimeError("typing failed")
        thread = f":thread:{message_thread_id}" if message_thread_id is not None else ""
        self.actions.append(f"{chat_id}{thread}:{action}")

    def get_updates(self, *, offset, timeout):
        self.calls.append((offset, timeout))
        return list(self.updates)


class FakeCodex:
    def __init__(self, response="done"):
        self.prompts: list[str] = []
        self.runs: list[tuple[str | None, str | None, Path | None, str | None]] = []
        self.extra_args: list[tuple[str, ...]] = []
        self.compact_prompts: list[str] = []
        self.compact_runs: list[tuple[str | None, str | None, Path | None, str | None]] = []
        self.compact_response = "compacted summary"
        self.compact_returncode = 0
        if isinstance(response, list):
            self.responses = list(response)
            self.response = response[-1] if response else "done"
        else:
            self.responses = []
            self.response = response
        self.emit_progress = False
        self.sleep_seconds = 0.0

    def run(
        self,
        prompt,
        *,
        model=None,
        reasoning_effort=None,
        workdir=None,
        sandbox_mode=None,
        extra_args=(),
        progress_callback=None,
    ):
        self.prompts.append(prompt)
        self.runs.append((model, reasoning_effort, workdir, sandbox_mode))
        self.extra_args.append(tuple(extra_args))
        if self.emit_progress and progress_callback is not None:
            progress_callback(
                {
                    "type": "item.started",
                    "item": {
                        "type": "command_execution",
                        "command": "/bin/bash -lc pwd",
                        "status": "in_progress",
                    },
                }
            )
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        text = self.responses.pop(0) if self.responses else self.response
        return CodexResult(text=text, returncode=0, stderr="")

    def compact(
        self,
        conversation_context,
        *,
        existing_summary="",
        model=None,
        reasoning_effort=None,
        workdir=None,
        sandbox_mode=None,
    ):
        self.compact_prompts.append(f"existing={existing_summary}\n{conversation_context}")
        self.compact_runs.append((model, reasoning_effort, workdir, sandbox_mode))
        return CodexResult(text=self.compact_response, returncode=self.compact_returncode, stderr="")


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


class PrefixModelCatalog(FakeModelCatalog):
    def __init__(self):
        self.models = (
            ModelChoice("gpt-5.4", "GPT-5.4", ("low", "medium", "high"), "medium"),
            ModelChoice("gpt-5.4-mini", "GPT-5.4-Mini", ("low", "medium", "high"), "medium"),
        )


class NoLowModelCatalog(FakeModelCatalog):
    def __init__(self):
        self.models = (
            ModelChoice("gpt-special", "GPT Special", ("high", "medium"), "medium"),
        )


class SparseEffortModelCatalog(FakeModelCatalog):
    def __init__(self):
        self.models = (
            ModelChoice("gpt-tiny", "GPT Tiny", ("minimal", "none"), "minimal"),
        )


class GatewayTests(unittest.TestCase):
    def _settings(
        self,
        workdir: Path,
        *,
        allowed_users=frozenset({42}),
        codex_sandbox="workspace-write",
        max_telegram_response_chars=12000,
    ) -> Settings:
        return Settings(
            bot_token="token",
            allowed_users=allowed_users,
            allowed_chats=frozenset(),
            codex_command="codex",
            codex_workdir=workdir,
            codex_model="",
            codex_profile="",
            codex_sandbox=codex_sandbox,
            codex_extra_args=(),
            codex_timeout_seconds=10,
            telegram_poll_timeout_seconds=30,
            telegram_request_timeout_seconds=45,
            telegram_disable_link_previews=False,
            max_telegram_response_chars=max_telegram_response_chars,
            session_history_turns=8,
            state_dir=workdir / ".state",
        )

    def _message(
        self,
        text: str,
        *,
        user_id=42,
        chat_id="100",
        chat_type="private",
        message_thread_id=None,
    ) -> IncomingMessage:
        return IncomingMessage(
            update_id=1,
            chat_id=chat_id,
            user_id=user_id,
            username="nima",
            text=text,
            message_id=9,
            chat_type=chat_type,
            message_thread_id=message_thread_id,
        )

    def _callback(self, data: str, *, user_id=42, chat_id="100", message_thread_id=None) -> IncomingCallback:
        return IncomingCallback(
            update_id=2,
            callback_query_id="cb1",
            chat_id=chat_id,
            user_id=user_id,
            username="nima",
            data=data,
            message_id=10,
            chat_type="private",
            message_thread_id=message_thread_id,
        )

    def _create_action(
        self,
        workspace: str,
        *,
        model: str = "gpt-5.5",
        effort: str = "high",
        sandbox: str = "yolo",
        topic_name: str | None = None,
    ) -> str:
        payload = {
            "action": "create_topic_session",
            "workspace": workspace,
            "model": model,
            "reasoning_effort": effort,
            "sandbox_mode": sandbox,
        }
        if topic_name is not None:
            payload["topic_name"] = topic_name
        return json.dumps(payload)

    def test_help_command_does_not_run_codex(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/help"))

            self.assertEqual(len(codex.prompts), 0)
            self.assertEqual(telegram.messages[0][1], "/reset\n/compact\n/fast\n/goal\n/models\n/workspace\n/sandbox")

    def test_status_command_shows_status(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = CodexTelegramGateway(
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
            self.assertIn("Sandbox: configured (workspace-write)", telegram.messages[0][1])
            self.assertIn("Fast mode: off", telegram.messages[0][1])
            self.assertIn("Goal: none", telegram.messages[0][1])

    def test_progress_text_suppresses_code_diffs_and_redacts_secrets(self):
        self.assertIsNone(sanitize_progress_text("```python\nprint('secret')\n```"))
        self.assertIsNone(sanitize_progress_text("diff --git a/app.py b/app.py"))
        self.assertIsNone(sanitize_progress_text("+ token = 'abc'"))
        text = sanitize_progress_text(
            "Using TELEGRAM_BOT_TOKEN=123456:abcdefghijklmnopqrstuvwxyz "
            "from /tmp/.env and .codex-telegram/e2e/admin-account.session in chat -1001234567890"
        )

        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("TELEGRAM_BOT_TOKEN=<redacted>", text)
        self.assertIn("<env-file>", text)
        self.assertIn("<session-file>", text)
        self.assertIn("<chat>", text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", text)
        self.assertNotIn("-1001234567890", text)

    def test_progress_text_redacts_positive_telegram_ids(self):
        text = sanitize_progress_text("telegram-send user_id=123456789 chat_id=987654321 bare 123456789")

        self.assertEqual(text, "telegram-send user_id=<id> chat_id=<id> bare <id>")

    def test_shape_telegram_response_text_expands_dense_inline_code_lists(self):
        shaped = shape_telegram_response_text(
            "Folders on your Desktop:\n\n`.agents`, `.git`, `codex-telegram`, `tailwind`."
        )

        self.assertEqual(
            shaped,
            "Folders on your Desktop:\n\n- `.agents`\n- `.git`\n- `codex-telegram`\n- `tailwind`",
        )

    def test_shape_telegram_response_text_preserves_code_blocks(self):
        shaped = shape_telegram_response_text("```text\n`a`, `b`, `c`\n```\n~~~text\n`d`, `e`, `f`\n~~~")

        self.assertEqual(shaped, "```text\n`a`, `b`, `c`\n```\n~~~text\n`d`, `e`, `f`\n~~~")

    def test_shape_telegram_response_text_preserves_long_fence_code_blocks(self):
        text = "````text\n```\n`a`, `b`, `c`\n```\n````"
        shaped = shape_telegram_response_text(text)

        self.assertEqual(shaped, text)

    def test_shape_telegram_response_text_preserves_indented_code_blocks(self):
        shaped = shape_telegram_response_text("    `a`, `b`, `c`")

        self.assertEqual(shaped, "    `a`, `b`, `c`")

    def test_safe_command_progress_preview_allows_simple_shell_commands(self):
        self.assertEqual(safe_command_progress_preview("/bin/bash -lc pwd"), "pwd")
        self.assertEqual(safe_command_progress_preview(["/bin/bash", "-lc", "ls -la src"]), "ls -la src")

    def test_safe_command_progress_preview_rejects_risky_command_text(self):
        self.assertIsNone(safe_command_progress_preview("python - <<'PY'\nprint('code')\nPY"))
        self.assertIsNone(safe_command_progress_preview("curl -H 'Authorization: Bearer abc' https://example.test"))
        self.assertIsNone(safe_command_progress_preview("python -c 'print(1)'"))
        self.assertIsNone(safe_command_progress_preview(["node", "--eval=console.log(1)"]))
        self.assertIsNone(safe_command_progress_preview(["perl", "-eprint 1"]))

    def test_workspace_command_shows_folder_buttons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "avatar").mkdir()
            (root / "beta").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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

    def test_start_workspace_session_keeps_selected_model_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            avatar = root / "avatar"
            avatar.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(root / ".state")
            store.save_model_preference("100", model="gpt-5.5", reasoning_effort="xhigh")
            store.append("100", "user", "old context")
            gateway = CodexTelegramGateway(
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
            preference = store.load_model_preference("100")
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.model, "gpt-5.5")
            self.assertEqual(preference.reasoning_effort, "xhigh")
            self.assertIn(f"Session workspace:\n{avatar.resolve()}", telegram.edits[1][2])
            self.assertIn("Model: gpt-5.5", telegram.edits[1][2])
            self.assertIn("Sandbox: configured (workspace-write)", telegram.edits[1][2])
            self.assertIn("Fast mode: off", telegram.edits[1][2])
            self.assertIn("Goal: none", telegram.edits[1][2])
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "xhigh", avatar.resolve(), None))
            self.assertNotIn("old context", codex.prompts[-1])

    def test_reset_command_keeps_selected_model_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_model_preference("100", model="gpt-5.5", reasoning_effort="xhigh")
            store.append("100", "user", "old context")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset"))
            gateway.handle_message(self._message("do the task"))

            preference = store.load_model_preference("100")
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.model, "gpt-5.5")
            self.assertEqual(preference.reasoning_effort, "xhigh")
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "xhigh", root.resolve(), None))
            self.assertNotIn("old context", codex.prompts[-1])

    def test_sandbox_command_sends_mode_buttons(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            gateway.handle_message(self._message("/sandbox"))

            self.assertEqual(codex.prompts, [])
            self.assertEqual(telegram.messages[0][1], "Choose sandbox mode:")
            self.assertEqual(
                telegram.reply_markups[0]["inline_keyboard"],
                [
                    [{"text": "Constrained", "callback_data": "sandbox:constrained"}],
                    [{"text": "YOLO", "callback_data": "sandbox:yolo"}],
                ],
            )

    def test_sandbox_callback_saves_default_for_next_message(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(self._callback("sandbox:yolo"))
            gateway.handle_message(self._message("do the task"))

            self.assertEqual(store.load_sandbox_mode("100"), "yolo")
            self.assertEqual(telegram.callback_answers, [("cb1", "Sandbox selected.")])
            self.assertIn("Selected sandbox: YOLO", telegram.edits[0][2])
            self.assertEqual(codex.runs[-1], (None, None, Path(tmp).resolve(), "yolo"))

    def test_reset_command_keeps_selected_sandbox_default(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_sandbox_mode("100", "yolo")
            store.append("100", "user", "old context")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset"))
            gateway.handle_message(self._message("do the task"))

            self.assertEqual(store.load_sandbox_mode("100"), "yolo")
            self.assertEqual(codex.runs[-1], (None, None, root.resolve(), "yolo"))
            self.assertNotIn("old context", codex.prompts[-1])

    def test_unset_sandbox_preserves_configured_read_only_sandbox(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root, codex_sandbox="read-only"),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do the task"))
            gateway.handle_message(self._message("/status"))

            self.assertIsNone(store.load_sandbox_mode("100"))
            self.assertEqual(codex.runs[-1], (None, None, root.resolve(), None))
            self.assertIn("Sandbox: configured (read-only)", telegram.messages[-1][1])

    def test_workspace_callback_rejects_path_escape(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            token = store.remember_workspace_token("..")
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("change the repo"))

            self.assertEqual(telegram.actions, ["100:typing"])
            self.assertEqual(telegram.messages[-1][1], "finished")
            self.assertEqual(codex.runs[-1], (None, None, Path(tmp).resolve(), None))
            self.assertIn("Current Telegram message", codex.prompts[0])
            self.assertNotIn("User: change the repo", codex.prompts[0])
            self.assertEqual([turn.role for turn in store.load("100")], ["user", "assistant"])

    def test_forum_topic_message_replies_in_thread_and_uses_topic_session(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("topic task", chat_id="-1001", message_thread_id=7))

            self.assertEqual(telegram.actions, ["-1001:thread:7:typing"])
            self.assertEqual(telegram.messages[-1][0], "-1001")
            self.assertEqual(telegram.messages[-1][1], "finished")
            self.assertEqual(telegram.message_threads[-1], 7)
            self.assertEqual([turn.role for turn in store.load("-1001:thread:7")], ["user", "assistant"])
            self.assertEqual(store.load("-1001"), [])
            self.assertIn("message_thread_id=7", codex.prompts[0])

    def test_reset_command_in_forum_topic_resets_only_that_topic(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            store = SessionStore(Path(tmp) / "state")
            store.append("-1001:thread:7", "user", "topic seven")
            store.append("-1001:thread:8", "user", "topic eight")
            store.append("-1001", "user", "general")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset", chat_id="-1001", message_thread_id=7))

            self.assertEqual(store.load("-1001:thread:7"), [])
            self.assertEqual([turn.text for turn in store.load("-1001:thread:8")], ["topic eight"])
            self.assertEqual([turn.text for turn in store.load("-1001")], ["general"])
            self.assertEqual(telegram.message_threads[-1], 7)

    def test_reset_command_in_topic_clears_compact_summary_only_for_that_topic(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            store = SessionStore(Path(tmp) / "state")
            for thread_id in (7, 8):
                key = f"-1001:thread:{thread_id}"
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=key,
                    topic_name=f"topic {thread_id}",
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
                store.save_compact_metadata(
                    key,
                    summary=f"summary {thread_id}",
                    source_char_count=10,
                    turns_compacted=1,
                    auto=False,
                    compacted_at=1,
                )
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset", chat_id="-1001", message_thread_id=7))

            seven = store.load_topic_session("-1001:thread:7")
            eight = store.load_topic_session("-1001:thread:8")
            self.assertIsNotNone(seven)
            self.assertIsNotNone(eight)
            assert seven is not None and eight is not None
            self.assertEqual(seven.compact_metadata, {})
            self.assertEqual(eight.compact_metadata["summary"], "summary 8")

    def test_compact_command_in_topic_persists_summary_and_clears_raw_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            store.save_active_workspace(topic_key, "kitia")
            store.save_model_preference(topic_key, model="gpt-5.5", reasoning_effort="high")
            store.save_sandbox_mode(topic_key, "yolo")
            store.append(topic_key, "user", "remember alpha")
            store.append(topic_key, "assistant", "alpha done")
            (root / "kitia").mkdir()
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/compact", chat_id="-1001", message_thread_id=7))

            self.assertEqual(telegram.messages[-2][1], "conversation compact started")
            self.assertEqual(telegram.messages[-1][1], "conversation compact finished")
            self.assertEqual(telegram.message_threads[-2:], [7, 7])
            self.assertIn("remember alpha", codex.compact_prompts[0])
            self.assertEqual(codex.compact_runs[-1], ("gpt-5.5", "high", (root / "kitia").resolve(), "read-only"))
            topic_session = store.load_topic_session(topic_key)
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.compact_metadata["summary"], "compacted summary")
            self.assertFalse(topic_session.compact_metadata["auto"])
            self.assertEqual(store.load(topic_key), [])

    def test_compact_command_in_general_chat_does_not_compact_topics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.append("-1001:thread:7", "user", "topic context")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/compact", chat_id="-1001", chat_type="supergroup"))

            self.assertEqual(codex.compact_prompts, [])
            self.assertIn("General chat compaction does not compact topic sessions", telegram.messages[-1][1])
            self.assertEqual([turn.text for turn in store.load("-1001:thread:7")], ["topic context"])

    def test_compact_command_reports_empty_topic_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/compact", chat_id="-1001", message_thread_id=7))

            self.assertEqual(codex.compact_prompts, [])
            self.assertIn("no conversation context", telegram.messages[-1][1])

    def test_compact_failure_leaves_existing_context_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex()
            codex.compact_returncode = 1
            codex.compact_response = "Codex failed"
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.append(topic_key, "user", "keep me")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/compact", chat_id="-1001", message_thread_id=7))

            self.assertIn("conversation compact failed", telegram.messages[-1][1])
            self.assertEqual([turn.text for turn in store.load(topic_key)], ["keep me"])
            topic_session = store.load_topic_session(topic_key)
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.compact_metadata, {})

    def test_compact_empty_codex_output_leaves_existing_context_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex()
            codex.compact_response = EMPTY_CODEX_RESPONSE
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.append(topic_key, "user", "keep me")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/compact", chat_id="-1001", message_thread_id=7))

            self.assertIn("conversation compact failed", telegram.messages[-1][1])
            self.assertEqual([turn.text for turn in store.load(topic_key)], ["keep me"])
            topic_session = store.load_topic_session(topic_key)
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.compact_metadata, {})

    def test_future_topic_prompt_includes_compact_summary_without_old_raw_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_compact_metadata(
                topic_key,
                summary="summary says alpha matters",
                source_char_count=100,
                turns_compacted=2,
                auto=False,
                compacted_at=1,
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("continue", chat_id="-1001", message_thread_id=7))

            self.assertIn("Compacted Telegram conversation context", codex.prompts[-1])
            self.assertIn("summary says alpha matters", codex.prompts[-1])
            self.assertNotIn("User: continue", codex.prompts[-1])

    def test_auto_compact_runs_before_topic_message_when_history_is_large(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("after compact")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.append(topic_key, "user", "x" * 25000)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("continue after compaction", chat_id="-1001", message_thread_id=7))

            self.assertEqual(len(codex.compact_prompts), 1)
            topic_session = store.load_topic_session(topic_key)
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertTrue(topic_session.compact_metadata["auto"])
            self.assertEqual([turn.text for turn in store.load(topic_key)], ["continue after compaction", "after compact"])
            self.assertIn("compacted summary", codex.prompts[-1])

    def test_goal_command_in_topic_sets_status_and_prompt_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("goal response")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/goal ship the topic feature", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal status", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("continue work", chat_id="-1001", message_thread_id=7))

            session = store.load_topic_session(topic_key)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.goal_metadata["objective"], "ship the topic feature")
            self.assertEqual(session.goal_metadata["status"], "active")
            self.assertIn("Goal: active", telegram.messages[-2][1])
            self.assertIn("Objective: ship the topic feature", telegram.messages[-2][1])
            self.assertIn("Active Telegram goal", codex.prompts[-1])
            self.assertIn("ship the topic feature", codex.prompts[-1])

    def test_progress_messages_route_only_to_active_topic(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("final response")
            codex.emit_progress = True
            store = SessionStore(root / ".state")
            for thread_id in (7, 8):
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=f"-1001:thread:{thread_id}",
                    topic_name=f"topic {thread_id}",
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do work", chat_id="-1001", message_thread_id=7))

            self.assertEqual(codex.runs[-1], (None, None, root.resolve(), None))
            self.assertIn("🖥 terminal:", telegram.messages[-2][1])
            self.assertIn("pwd", telegram.messages[-2][1])
            self.assertEqual(telegram.message_threads[-2], 7)
            self.assertEqual(telegram.messages[-1][1], "final response")
            self.assertEqual(telegram.message_threads[-1], 7)
            self.assertEqual(store.load("-1001:thread:8"), [])

    def test_progress_dedupe_survives_typing_restore_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            telegram.fail_chat_action = True
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )
            callback = gateway._progress_callback_for_message(self._message("do work", chat_id="-1001", message_thread_id=7))
            event = {
                "type": "item.started",
                "item": {
                    "type": "command_execution",
                    "command": "/bin/bash -lc pwd",
                    "status": "in_progress",
                },
            }

            callback(event)
            callback(event)

            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("🖥 terminal:", telegram.messages[0][1])
            self.assertEqual(telegram.message_threads, [7])

    def test_typing_indicator_refreshes_until_request_finishes(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            codex.sleep_seconds = 0.04
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )
            original_interval = gateway_module.TYPING_REFRESH_INTERVAL_SECONDS
            gateway_module.TYPING_REFRESH_INTERVAL_SECONDS = 0.01
            try:
                gateway.handle_message(self._message("do it", chat_id="-1001", message_thread_id=7))
            finally:
                gateway_module.TYPING_REFRESH_INTERVAL_SECONDS = original_interval

            action_count = len(telegram.actions)
            time.sleep(0.03)

            self.assertGreaterEqual(action_count, 2)
            self.assertEqual(len(telegram.actions), action_count)
            self.assertTrue(all(action == "-1001:thread:7:typing" for action in telegram.actions))
            self.assertEqual(telegram.messages[-1][1], "finished")

    def test_goal_command_is_topic_scoped_and_general_chat_does_not_mutate_topics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            for thread_id in (7, 8):
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=f"-1001:thread:{thread_id}",
                    topic_name=f"topic {thread_id}",
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/goal first topic goal", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal", chat_id="-1001", chat_type="supergroup"))
            gateway.handle_message(self._message("/status", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/status", chat_id="-1001", message_thread_id=8))

            seven = store.load_topic_session("-1001:thread:7")
            eight = store.load_topic_session("-1001:thread:8")
            self.assertIsNotNone(seven)
            self.assertIsNotNone(eight)
            assert seven is not None and eight is not None
            self.assertEqual(seven.goal_metadata["objective"], "first topic goal")
            self.assertEqual(eight.goal_metadata, {})
            self.assertIn("General chat goals do not target topic sessions", telegram.messages[-3][1])
            self.assertIn("Goal: active: first topic goal", telegram.messages[-2][1])
            self.assertIn("Goal: none", telegram.messages[-1][1])

    def test_goal_update_complete_clear_and_active_goal_guard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/goal first goal", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal second goal", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal update check edge cases", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal complete", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/goal clear", chat_id="-1001", message_thread_id=7))

            self.assertIn("already active", telegram.messages[-4][1])
            self.assertIn("Goal updated", telegram.messages[-3][1])
            self.assertIn("Goal marked complete", telegram.messages[-2][1])
            self.assertIn("Goal cleared", telegram.messages[-1][1])
            session = store.load_topic_session(topic_key)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.goal_metadata, {})

    def test_reset_preserves_active_goal_but_clears_history_and_compact_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_goal(topic_key, objective="keep goal", created_at=1)
            store.save_compact_metadata(
                topic_key,
                summary="old compact",
                source_char_count=10,
                turns_compacted=1,
                auto=False,
                compacted_at=1,
            )
            store.append(topic_key, "user", "old history")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset", chat_id="-1001", message_thread_id=7))

            session = store.load_topic_session(topic_key)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(store.load(topic_key), [])
            self.assertEqual(session.compact_metadata, {})
            self.assertEqual(session.goal_metadata["objective"], "keep goal")
            self.assertIn("Active goal was kept", telegram.messages[-1][1])

    def test_compact_includes_active_goal_and_preserves_goal_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("after compact")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_goal(topic_key, objective="preserve active goal", created_at=1)
            store.append(topic_key, "user", "x" * 25000)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("continue after compaction", chat_id="-1001", message_thread_id=7))

            self.assertIn("Active Telegram goal", codex.compact_prompts[-1])
            self.assertIn("preserve active goal", codex.compact_prompts[-1])
            self.assertIn("Active Telegram goal", codex.prompts[-1])
            session = store.load_topic_session(topic_key)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.goal_metadata["status"], "active")
            self.assertEqual(session.goal_metadata["objective"], "preserve active goal")

    def test_fast_command_in_topic_lowers_reasoning_without_changing_model_or_sandbox(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("fast response")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            store.save_model_preference(topic_key, model="gpt-5.5", reasoning_effort="high")
            store.save_sandbox_mode(topic_key, "yolo")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/fast", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("do it quickly", chat_id="-1001", message_thread_id=7))

            self.assertTrue(store.load_fast_mode(topic_key))
            self.assertIn("Fast mode enabled", telegram.messages[-2][1])
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "low", root.resolve(), "yolo"))
            preference = store.load_model_preference(topic_key)
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.reasoning_effort, "high")

    def test_fast_command_is_topic_scoped_and_status_reports_mode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            for thread_id in (7, 8):
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=f"-1001:thread:{thread_id}",
                    topic_name=f"topic {thread_id}",
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
                store.save_model_preference(f"-1001:thread:{thread_id}", model="gpt-5.5", reasoning_effort="high")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/fast", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/status", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("/status", chat_id="-1001", message_thread_id=8))

            self.assertTrue(store.load_fast_mode("-1001:thread:7"))
            self.assertFalse(store.load_fast_mode("-1001:thread:8"))
            self.assertIn("Thinking: low", telegram.messages[-2][1])
            self.assertIn("Fast mode: on", telegram.messages[-2][1])
            self.assertIn("Thinking: high", telegram.messages[-1][1])
            self.assertIn("Fast mode: off", telegram.messages[-1][1])

    def test_fast_command_in_general_chat_does_not_affect_topics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/fast", chat_id="-1001", chat_type="supergroup"))

            self.assertFalse(store.load_fast_mode("-1001:thread:7"))
            self.assertIn("Run /fast inside a Codex session topic", telegram.messages[-1][1])

    def test_fast_off_disables_topic_fast_mode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.set_fast_mode(topic_key, True)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/fast off", chat_id="-1001", message_thread_id=7))

            self.assertFalse(store.load_fast_mode(topic_key))
            self.assertIn("Fast mode disabled", telegram.messages[-1][1])

    def test_explicit_model_effort_selection_disables_fast_mode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.set_fast_mode(topic_key, True)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(self._callback("effort:gpt-5.5:high", chat_id="-1001", message_thread_id=7))

            self.assertFalse(store.load_fast_mode(topic_key))
            preference = store.load_model_preference(topic_key)
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.reasoning_effort, "high")

    def test_fast_mode_uses_lowest_supported_effort_when_low_is_unavailable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("fast response")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-special",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_model_preference(topic_key, model="gpt-special", reasoning_effort="high")
            store.set_fast_mode(topic_key, True)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=NoLowModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do it quickly", chat_id="-1001", message_thread_id=7))

            self.assertEqual(codex.runs[-1], ("gpt-special", "medium", root.resolve(), None))

    def test_fast_mode_defaults_to_low_when_model_metadata_is_unavailable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            codex = FakeCodex("fast response")
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-missing",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_model_preference(topic_key, model="gpt-missing", reasoning_effort="high")
            store.set_fast_mode(topic_key, True)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=NonAuthoritativeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do it quickly", chat_id="-1001", message_thread_id=7))

            self.assertEqual(codex.runs[-1], ("gpt-missing", "low", root.resolve(), None))

    def test_reset_preserves_fast_mode_but_clears_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.set_fast_mode(topic_key, True)
            store.append(topic_key, "user", "old")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/reset", chat_id="-1001", message_thread_id=7))

            self.assertTrue(store.load_fast_mode(topic_key))
            self.assertEqual(store.load(topic_key), [])

    def test_workspace_start_in_topic_clears_compact_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            avatar = root / "avatar"
            avatar.mkdir()
            topic_key = "-1001:thread:7"
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key=topic_key,
                topic_name="kitia topic",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_compact_metadata(
                topic_key,
                summary="stale summary",
                source_char_count=100,
                turns_compacted=2,
                auto=False,
                compacted_at=1,
            )
            store.save_goal(topic_key, objective="stale goal", created_at=1)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("/workspace", chat_id="-1001", message_thread_id=7))
            avatar_button = telegram.reply_markups[0]["inline_keyboard"][1][0]
            gateway.handle_callback(self._callback(avatar_button["callback_data"], chat_id="-1001", message_thread_id=7))
            start_button = telegram.edits[0][3]["inline_keyboard"][0][0]
            gateway.handle_callback(self._callback(start_button["callback_data"], chat_id="-1001", message_thread_id=7))

            session = store.load_topic_session(topic_key)
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.compact_metadata, {})
            self.assertEqual(session.goal_metadata, {})

    def test_general_controller_creates_configured_topic_session(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready. Send the task here.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "Hey can you make me a new topic with gpt 5.5 high in kitia yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            topic_key = "-1001:thread:50"
            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])
            self.assertEqual(len(codex.prompts), 2)
            self.assertIn("General-chat controller", codex.prompts[0])
            self.assertIn('"topic_name":"<optional topic title>"', codex.prompts[0])
            self.assertIn("Current General-chat message:", codex.prompts[0])
            self.assertIn("topic-scoped Codex agent", codex.prompts[1])
            self.assertEqual(store.load_active_workspace(topic_key), "kitia")
            self.assertEqual(store.load_sandbox_mode(topic_key), "yolo")
            preference = store.load_model_preference(topic_key)
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.model, "gpt-5.5")
            self.assertEqual(preference.reasoning_effort, "high")
            topic_session = store.load_topic_session(topic_key)
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.workspace, "kitia")
            self.assertEqual(topic_session.model, "gpt-5.5")
            self.assertEqual(topic_session.reasoning_effort, "high")
            self.assertEqual(topic_session.sandbox_mode, "yolo")
            self.assertEqual(codex.runs[0][0:2], (None, None))
            self.assertNotEqual(codex.runs[0][2], root.resolve())
            self.assertEqual(codex.runs[0][3], "read-only")
            self.assertEqual(codex.extra_args[0], ("--skip-git-repo-check",))
            self.assertEqual(codex.runs[1], ("gpt-5.5", "high", kitia.resolve(), "read-only"))
            self.assertEqual(codex.extra_args[1], ())
            self.assertIn("Topic agent ready", telegram.messages[0][1])
            self.assertEqual(telegram.message_threads[0], 50)
            self.assertIn("Created topic `kitia | gpt-5.5 high | yolo`", telegram.messages[1][1])
            self.assertEqual(telegram.message_threads[1], None)

    def test_topic_agent_intro_is_truncated_to_telegram_limit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "a" * 13000,
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root, max_telegram_response_chars=40),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertLess(len(telegram.messages[0][1]), 13000)
            self.assertIn("[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]", telegram.messages[0][1])
            self.assertEqual(store.load("-1001:thread:50")[0].text, telegram.messages[0][1])

    def test_topic_agent_fallback_intro_is_saved_to_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root, max_telegram_response_chars=40),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            history = store.load("-1001:thread:50")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].role, "assistant")
            self.assertIn("Session ready", history[0].text)
            self.assertIn("[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]", history[0].text)
            self.assertEqual(history[0].text, telegram.messages[0][1])

    def test_general_controller_uses_controller_supplied_topic_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="Injected Title"),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "Injected Title")])

    def test_general_controller_accepts_controller_root_workspace_typo_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action(".", effort="medium", sandbox="yolo", topic_name="desktop | gpt-5.5 medium | yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "Hey can you make me a new topic with gpt 5.5 medium in deaktop yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "desktop | gpt-5.5 medium | yolo")])
            topic_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.workspace, "")
            self.assertEqual(topic_session.reasoning_effort, "medium")
            self.assertEqual(topic_session.sandbox_mode, "yolo")

    def test_general_controller_trusts_controller_root_workspace_from_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action(".", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in Desktop/kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", f"{root.name} | gpt-5.5 high | yolo")])

    def test_general_controller_trusts_controller_root_word_workspace_from_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action(".", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in root/kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", f"{root.name} | gpt-5.5 high | yolo")])

    def test_general_controller_trusts_controller_hyphenated_root_alias_context(self):
        for visible_workspace in ("root-kitia", "Desktop-kitia", "deaktop-kitia"):
            with self.subTest(visible_workspace=visible_workspace):
                with TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    telegram = FakeTelegram()
                    codex = FakeCodex(self._create_action(".", effort="high", sandbox="yolo"))
                    gateway = CodexTelegramGateway(
                        settings=self._settings(root),
                        telegram=telegram,
                        codex=codex,
                        model_catalog=FakeModelCatalog(),
                        sessions=SessionStore(root / ".state"),
                    )

                    gateway.handle_message(
                        self._message(
                            f"make a new topic in {visible_workspace} with gpt 5.5 high yolo",
                            chat_id="-1001",
                            chat_type="supergroup",
                        )
                    )

                    self.assertEqual(telegram.created_topics, [("-1001", f"{root.name} | gpt-5.5 high | yolo")])

    def test_general_controller_accepts_i_need_a_new_topic_create_intent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "I need a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_accepts_controller_dot_root_workspace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action(".", topic_name="root | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in . with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "root | gpt-5.5 high | yolo")])

    def test_general_controller_prefers_real_workspace_named_desktop_over_root_alias(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            desktop = root / "desktop"
            desktop.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("desktop", topic_name="desktop | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in desktop with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "desktop")
            self.assertEqual(codex.runs[1][2], desktop.resolve())

    def test_general_controller_prefers_capitalized_real_workspace_over_root_alias(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            desktop = root / "desktop"
            desktop.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("Desktop", topic_name="desktop | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in Desktop with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "desktop")
            self.assertEqual(codex.runs[1][2], desktop.resolve())

    def test_general_controller_create_action_trusts_controller_settings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "please create a new topic",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_accepts_catalog_specific_controller_reasoning_effort(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action(
                        "kitia",
                        model="gpt-tiny",
                        effort="minimal",
                        topic_name="kitia | gpt-tiny minimal | yolo",
                    ),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=SparseEffortModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt tiny minimal yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-tiny minimal | yolo")])
            self.assertIn("gpt-tiny: efforts=minimal, none", codex.prompts[0])
            self.assertIn("selected model's listed reasoning efforts", codex.prompts[0])
            topic_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.reasoning_effort, "minimal")

    def test_general_controller_rejects_controller_model_prefix_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                self._create_action(
                    "kitia",
                    model="gpt-5.4-unknown",
                    effort="high",
                    sandbox="yolo",
                    topic_name="kitia | gpt-5.4 high | yolo",
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=PrefixModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.4 mini high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [])
            self.assertIn("model", telegram.messages[-1][1])

    def test_general_controller_trusts_available_controller_model_from_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                self._create_action(
                    "kitia",
                    model="gpt-5.4-mini",
                    effort="high",
                    sandbox="yolo",
                    topic_name="kitia | gpt-5.4-mini high | yolo",
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=PrefixModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.4 minimal high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.4-mini high | yolo")])

    def test_general_controller_rejects_controller_effort_substring_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                self._create_action(
                    "kitia",
                    effort="extreme",
                    sandbox="yolo",
                    topic_name="kitia | gpt-5.5 high | yolo",
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 xhigh yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [])
            self.assertIn("thinking", telegram.messages[-1][1])

    def test_general_controller_rejects_controller_xhigh_for_model_without_xhigh(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                self._create_action(
                    "kitia",
                    model="gpt-5.4-mini",
                    effort="xhigh",
                    sandbox="yolo",
                    topic_name="kitia | gpt-5.4-mini high | yolo",
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=PrefixModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.4 mini x high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [])
            self.assertIn("thinking", telegram.messages[-1][1])

    def test_general_controller_trusts_controller_yolo_even_when_text_is_negated(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high do not use the yolo sandbox",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_accepts_negated_yolo_as_constrained_sandbox(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", effort="high", sandbox="constrained"),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high not yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | constrained")])

    def test_general_controller_trusts_controller_yolo_from_never_use_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high never use yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_trusts_controller_yolo_from_avoid_bypass_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high avoid bypass",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_trusts_controller_yolo_from_avoid_using_bypass_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", effort="high", sandbox="yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high avoid using bypass",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_non_forum_supergroup_message_uses_regular_codex_session(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            telegram.chat_info = ChatInfo(title="Plain Group", chat_type="supergroup", is_forum=False)
            codex = FakeCodex("regular group response")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [])
            self.assertEqual(telegram.messages[-1][1], "regular group response")
            self.assertNotIn("General-chat controller", codex.prompts[0])

    def test_general_forum_metadata_failure_does_not_fall_through_or_cache_false(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            telegram.fail_get_chat = True
            codex = FakeCodex("regular group response")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )
            telegram.fail_get_chat = False
            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(len(codex.prompts), 1)
            self.assertIn("could not verify", telegram.messages[0][1])
            self.assertIn("General-chat controller", codex.prompts[0])

    def test_general_controller_does_not_treat_workspace_write_as_root_workspace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action(".", sandbox="constrained", topic_name="root | gpt-5.5 high | constrained"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "create a new topic with gpt 5.5 high workspace write",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "root | gpt-5.5 high | constrained")])

    def test_general_controller_trusts_controller_nested_workspace_from_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prod" / "app").mkdir(parents=True)
            telegram = FakeTelegram()
            codex = FakeCodex(
                self._create_action("prod/app", topic_name="prod/app | gpt-5.5 high | yolo")
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in app with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "prod/app | gpt-5.5 high | yolo")])

    def test_general_controller_lists_and_accepts_controller_nested_workspace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "prod" / "app"
            nested.mkdir(parents=True)
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("prod/app", topic_name="prod/app | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in prod/app with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertIn("- prod/app", codex.prompts[0])
            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "prod/app")
            self.assertEqual(codex.runs[1][2], nested.resolve())

    def test_general_controller_lists_direct_workspaces_before_nested_limit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = root / "aaa"
            early.mkdir()
            for index in range(250):
                (early / f"nested-{index:03d}").mkdir()
            target = root / "zzz-target"
            target.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("zzz-target", topic_name="zzz-target | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in zzz-target with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertIn("- zzz-target", codex.prompts[0])
            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "zzz-target")

    def test_general_forum_message_reports_group_and_topic_metadata_without_ids(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia | gpt-5.5 high | yolo",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            store.set_topic_session_closed("-1001:thread:50", True)
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=FakeCodex(json.dumps({"action": "report_metadata"})),
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("report metadata", chat_id="-1001", chat_type="supergroup")
            )

            text = telegram.messages[-1][1]
            self.assertIn("Group metadata:", text)
            self.assertIn("Title: Dev Group", text)
            self.assertIn("Forum topics enabled: yes", text)
            self.assertIn("Recorded topic sessions: 1", text)
            self.assertIn("kitia | gpt-5.5 high | yolo [closed]", text)
            self.assertIn("goal=none", text)
            self.assertNotIn("-1001", text)
            self.assertNotIn("thread", text.lower())

    def test_general_controller_metadata_trusts_controller_action(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "report_metadata"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )

            self.assertIn("Group metadata:", telegram.messages[-1][1])

    def test_general_controller_metadata_action_can_use_natural_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "report_metadata"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "summarize this text: show metadata",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertIn("Group metadata:", telegram.messages[-1][1])

    def test_general_controller_preserves_metadata_shortcut_phrase(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "report_metadata"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("group metadata", chat_id="-1001", chat_type="supergroup")
            )

            self.assertIn("Group metadata:", telegram.messages[-1][1])

    def test_general_forum_message_renames_group_explicitly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "Temporary Dev Group"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to Temporary Dev Group", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "Temporary Dev Group")])
            self.assertIn("Renamed group to `Temporary Dev Group`.", telegram.messages[-1][1])

    def test_general_forum_group_rename_uses_controller_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "my project"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to My Project", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "my project")])
            self.assertIn("Renamed group to `my project`.", telegram.messages[-1][1])

    def test_general_controller_group_rename_trusts_controller_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "Hidden Title"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "Hidden Title")])

    def test_general_controller_group_rename_uses_controller_title_over_message_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "Dev"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to Development", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "Dev")])

    def test_general_controller_group_rename_accepts_unicode_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            title = "گروه تست"
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": title}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(f"rename group to {title}", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", title)])

    def test_general_controller_group_rename_trusts_partial_unicode_controller_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "گروه"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to گروه تست", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "گروه")])

    def test_general_controller_group_rename_trusts_extra_controller_suffix(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "Dev 🚨"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to Dev", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [("-1001", "Dev 🚨")])

    def test_general_controller_accepts_explicit_admin_phrasing_with_articles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitia topic"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "please delete the topic kitia topic",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])

    def test_general_forum_group_rename_reports_permission_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            telegram.fail_group_metadata = True
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "Temporary Dev Group"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("rename group to Temporary Dev Group", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [])
            self.assertIn("could not rename the group", telegram.messages[-1][1])

    def test_general_forum_group_rename_rejects_overlong_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "rename_group", "title": "a" * 129}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    f"rename group to {'a' * 129}",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.group_renames, [])
            self.assertIn("Group titles must be 1 to 128 characters.", telegram.messages[-1][1])

    def test_general_forum_group_admin_text_must_be_explicit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "reply", "text": "What group action should I take?"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("please think about renaming the group", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.group_renames, [])
            self.assertEqual(len(codex.prompts), 1)
            self.assertIn("What group action should I take?", telegram.messages[-1][1])

    def test_general_forum_message_renames_recorded_topic_session(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                    json.dumps(
                        {
                            "action": "rename_topic",
                            "target": "kitia | gpt-5.5 high | yolo",
                            "new_name": "kitia renamed",
                        }
                    ),
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )
            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            gateway.handle_message(
                self._message(
                    "rename topic kitia | gpt-5.5 high | yolo to kitia renamed",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertIn("- #50 kitia | gpt-5.5 high | yolo [open]", codex.prompts[2])
            self.assertEqual(telegram.edited_forum_topics, [("-1001", 50, "kitia renamed")])
            topic_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.topic_name, "kitia renamed")
            self.assertIn("Renamed topic", telegram.messages[-1][1])

    def test_general_forum_topic_rename_uses_controller_new_name(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia | gpt-5.5 high | yolo",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            codex = FakeCodex(
                json.dumps(
                    {
                        "action": "rename_topic",
                        "target": "kitia | gpt-5.5 high | yolo",
                        "new_name": "kitia renamed",
                    }
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "rename topic kitia | gpt-5.5 high | yolo to Kitia Renamed",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.edited_forum_topics, [("-1001", 50, "kitia renamed")])
            topic_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(topic_session)
            assert topic_session is not None
            self.assertEqual(topic_session.topic_name, "kitia renamed")

    def test_general_forum_message_closes_and_reopens_topic_session(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                    json.dumps({"action": "close_topic", "target": "kitia | gpt-5.5 high | yolo"}),
                    json.dumps({"action": "reopen_topic", "target": "kitia | gpt-5.5 high | yolo"}),
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )
            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            gateway.handle_message(
                self._message("close topic kitia | gpt-5.5 high | yolo", chat_id="-1001", chat_type="supergroup")
            )
            closed_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(closed_session)
            assert closed_session is not None
            self.assertTrue(closed_session.is_closed)

            gateway.handle_message(
                self._message("reopen topic kitia | gpt-5.5 high | yolo", chat_id="-1001", chat_type="supergroup")
            )
            reopened_session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(reopened_session)
            assert reopened_session is not None
            self.assertFalse(reopened_session.is_closed)
            self.assertEqual(telegram.closed_forum_topics, [("-1001", 50)])
            self.assertEqual(telegram.reopened_forum_topics, [("-1001", 50)])

    def test_general_forum_message_deletes_topic_and_cleans_bridge_state_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                    json.dumps({"action": "delete_topic", "target": "kitia | gpt-5.5 high | yolo"}),
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )
            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            store.append("-1001:thread:50", "user", "old topic state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=60,
                session_key="-1001:thread:60",
                topic_name="other topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )

            gateway.handle_message(
                self._message("delete topic kitia | gpt-5.5 high | yolo", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))
            self.assertEqual(store.load("-1001:thread:50"), [])
            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "")
            self.assertIsNone(store.load_model_preference("-1001:thread:50"))
            self.assertIsNone(store.load_sandbox_mode("-1001:thread:50"))
            self.assertIsNotNone(store.load_topic_session("-1001:thread:60"))
            self.assertTrue((root / "kitia").is_dir())
            self.assertIn("removed only its bridge session state", telegram.messages[-1][1])

    def test_general_forum_lifecycle_rejects_partial_topic_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            for thread_id, name in ((50, "kitia dev"), (60, "kitia prod")):
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=f"-1001:thread:{thread_id}",
                    topic_name=name,
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
            codex = FakeCodex(json.dumps({"action": "close_topic", "target": "kitia"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("close topic kitia", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.closed_forum_topics, [])
            self.assertIn("ambiguous", telegram.messages[-1][1])

    def test_general_forum_lifecycle_accepts_unique_partial_topic_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia | gpt-5.5 high | yolo",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitia"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic kitia", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_forum_lifecycle_accepts_normalized_topic_title_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia | gpt-5.5 high | yolo",
                workspace="kitia",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="yolo",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitiagpt55highyolo"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic kitiagpt55highyolo", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_forum_lifecycle_accepts_exact_numeric_topic_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="50",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "50"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic 50", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 7)])
            self.assertIsNone(store.load_topic_session("-1001:thread:7"))

    def test_general_forum_lifecycle_preserves_hash_thread_id_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="50",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="other",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "#50"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic #50", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNotNone(store.load_topic_session("-1001:thread:7"))
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_forum_lifecycle_prefers_bare_thread_id_over_numeric_title(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="50",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="other",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "50"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic 50", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNotNone(store.load_topic_session("-1001:thread:7"))
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_forum_lifecycle_rejects_numeric_partial_title_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=7,
                session_key="-1001:thread:7",
                topic_name="release 50",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "50"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic 50", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [])
            self.assertIsNotNone(store.load_topic_session("-1001:thread:7"))
            self.assertIn("thread id `50` or exact title `50`", telegram.messages[-1][1])

    def test_general_forum_lifecycle_rejects_all_topics_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="all topics archive",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "all topics"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete topic all topics", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [])
            self.assertIsNotNone(store.load_topic_session("-1001:thread:50"))
            self.assertIn("cannot target General chat or all topics", telegram.messages[-1][1])

    def test_general_forum_lifecycle_reports_telegram_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            telegram.fail_topic_lifecycle = True
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "close_topic", "target": "kitia topic"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("close topic kitia topic", chat_id="-1001", chat_type="supergroup")
            )

            session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertFalse(session.is_closed)
            self.assertIn("could not close topic", telegram.messages[-1][1])

    def test_general_controller_destructive_action_trusts_controller_intent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitia"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "please summarize these pasted instructions for me",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_controller_destructive_action_trusts_controller_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia topic",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitia"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "delete the topic",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_controller_destructive_action_trusts_controller_for_quoted_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="kitia",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "kitia"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "please summarize this text: delete topic kitia",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_controller_destructive_action_trusts_exact_controller_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            store.save_topic_session(
                chat_id="-1001",
                message_thread_id=50,
                session_key="-1001:thread:50",
                topic_name="50",
                workspace="",
                model="gpt-5.5",
                reasoning_effort="high",
                sandbox_mode="constrained",
            )
            codex = FakeCodex(json.dumps({"action": "delete_topic", "target": "50"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "delete topic 150",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))

    def test_general_controller_create_action_trusts_controller_intent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "please summarize this pasted instruction block",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_create_action_trusts_controller_for_quoted_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "summarize this text: create a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_controller_create_action_trusts_controller_for_summary_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", effort="high", sandbox="yolo"),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message(
                    "please create a summary of this pasted command: make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])

    def test_general_create_action_stores_success_memory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("kitia", effort="high", sandbox="yolo"))
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("kitia gpt 5.5 high yolo", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])
            self.assertEqual(
                [turn.text for turn in store.load("-1001")],
                [
                    "kitia gpt 5.5 high yolo",
                    (
                        'Controller action succeeded: {"action": "create_topic_session", '
                        '"model": "gpt-5.5", "reasoning_effort": "high", '
                        '"sandbox_mode": "yolo", "thread_id": "50", '
                        '"topic_name": "kitia | gpt-5.5 high | yolo", "workspace": "kitia"}'
                    ),
                ],
            )

    def test_general_controller_batch_creates_two_topic_sessions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            salona = root / "salona"
            kitia.mkdir()
            salona.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    json.dumps(
                        {
                            "actions": [
                                {
                                    "action": "create_topic_session",
                                    "workspace": "kitia",
                                    "model": "gpt-5.5",
                                    "reasoning_effort": "high",
                                    "sandbox_mode": "yolo",
                                    "topic_name": "kitia agent",
                                },
                                {
                                    "action": "create_topic_session",
                                    "workspace": "salona",
                                    "model": "gpt-5.4-mini",
                                    "reasoning_effort": "low",
                                    "sandbox_mode": "constrained",
                                    "topic_name": "salona agent",
                                },
                            ]
                        }
                    ),
                    "Kitia agent ready.",
                    "Salona agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make two topics: kitia high yolo and salona mini low constrained",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(
                telegram.created_topics,
                [
                    ("-1001", "kitia agent"),
                    ("-1001", "salona agent"),
                ],
            )
            self.assertEqual(telegram.message_threads, [50, 51, None])
            self.assertIn("Kitia agent ready.", telegram.messages[0][1])
            self.assertIn("Salona agent ready.", telegram.messages[1][1])
            self.assertIn("Completed 2 of 2 requested actions.", telegram.messages[2][1])
            self.assertIn("Created topic `kitia agent` #50", telegram.messages[2][1])
            self.assertIn("Created topic `salona agent` #51", telegram.messages[2][1])
            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "kitia")
            self.assertEqual(store.load_active_workspace("-1001:thread:51"), "salona")
            first_preference = store.load_model_preference("-1001:thread:50")
            second_preference = store.load_model_preference("-1001:thread:51")
            self.assertIsNotNone(first_preference)
            self.assertIsNotNone(second_preference)
            assert first_preference is not None
            assert second_preference is not None
            self.assertEqual(first_preference.model, "gpt-5.5")
            self.assertEqual(second_preference.model, "gpt-5.4-mini")
            stored = "\n".join(turn.text for turn in store.load("-1001"))
            self.assertIn("Controller batch result:", stored)
            self.assertIn("kitia agent", stored)
            self.assertIn("salona agent", stored)
            self.assertIn('"success": "yes"', stored)

    def test_general_controller_batch_can_create_then_rename_topic(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    json.dumps(
                        {
                            "actions": [
                                {
                                    "action": "create_topic_session",
                                    "workspace": "kitia",
                                    "model": "gpt-5.5",
                                    "reasoning_effort": "high",
                                    "sandbox_mode": "yolo",
                                    "topic_name": "kitia temp",
                                },
                                {
                                    "action": "rename_topic",
                                    "target": "kitia temp",
                                    "new_name": "kitia final",
                                },
                            ]
                        }
                    ),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("make kitia temp and rename it kitia final", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia temp")])
            self.assertEqual(telegram.edited_forum_topics, [("-1001", 50, "kitia final")])
            session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.topic_name, "kitia final")
            self.assertEqual(telegram.message_threads, [50, None])
            self.assertIn("Completed 2 of 2 requested actions.", telegram.messages[-1][1])
            self.assertIn("Renamed topic `kitia temp` to `kitia final`.", telegram.messages[-1][1])

    def test_general_controller_batch_deletes_multiple_topics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            for thread_id, topic_name in ((50, "kitia topic"), (51, "salona topic")):
                store.save_topic_session(
                    chat_id="-1001",
                    message_thread_id=thread_id,
                    session_key=f"-1001:thread:{thread_id}",
                    topic_name=topic_name,
                    workspace="",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    sandbox_mode="constrained",
                )
            codex = FakeCodex(
                json.dumps(
                    {
                        "actions": [
                            {"action": "delete_topic", "target": "kitia topic"},
                            {"action": "delete_topic", "target": "salona topic"},
                        ]
                    }
                )
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("delete kitia and salona topics", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.deleted_forum_topics, [("-1001", 50), ("-1001", 51)])
            self.assertIsNone(store.load_topic_session("-1001:thread:50"))
            self.assertIsNone(store.load_topic_session("-1001:thread:51"))
            self.assertEqual(len(telegram.messages), 1)
            self.assertIn("Completed 2 of 2 requested actions.", telegram.messages[0][1])
            self.assertIn("Deleted topic `kitia topic`", telegram.messages[0][1])
            self.assertIn("Deleted topic `salona topic`", telegram.messages[0][1])

    def test_general_controller_batch_keeps_successes_when_later_action_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    json.dumps(
                        {
                            "actions": [
                                {
                                    "action": "create_topic_session",
                                    "workspace": "kitia",
                                    "model": "gpt-5.5",
                                    "reasoning_effort": "high",
                                    "sandbox_mode": "yolo",
                                    "topic_name": "kitia agent",
                                },
                                {
                                    "action": "create_topic_session",
                                    "workspace": "missing",
                                    "model": "gpt-5.5",
                                    "reasoning_effort": "high",
                                    "sandbox_mode": "yolo",
                                    "topic_name": "missing agent",
                                },
                            ]
                        }
                    ),
                    "Kitia agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("make kitia and missing topics", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia agent")])
            self.assertIsNotNone(store.load_topic_session("-1001:thread:50"))
            self.assertIn("Completed 1 of 2 requested actions.", telegram.messages[-1][1])
            self.assertIn("OK: Created topic `kitia agent` #50", telegram.messages[-1][1])
            self.assertIn("Failed: I could not find a workspace named `missing`", telegram.messages[-1][1])
            stored = "\n".join(turn.text for turn in store.load("-1001"))
            self.assertIn('"success": "yes"', stored)
            self.assertIn('"success": "no"', stored)

    def test_general_controller_creates_topic_from_multiturn_context_without_confirmation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    json.dumps({"action": "reply", "text": "Which workspace, model, thinking, and sandbox?"}),
                    self._create_action("kitia", effort="high", sandbox="yolo"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("please create a new topic", chat_id="-1001", chat_type="supergroup")
            )
            gateway.handle_message(
                self._message("kitia gpt 5.5 high yolo", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])
            self.assertIn("Recent General-chat controller context:", codex.prompts[1])
            self.assertIn("User: please create a new topic", codex.prompts[1])
            self.assertEqual(store.load_active_workspace("-1001:thread:50"), "kitia")
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "high", kitia.resolve(), "read-only"))

    def test_general_controller_confirm_is_ordinary_general_text(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex("should not run")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(self._message("confirm", chat_id="-1001", chat_type="supergroup"))

            self.assertEqual(telegram.created_topics, [])
            self.assertEqual(len(codex.prompts), 1)
            self.assertIn("could not understand", telegram.messages[-1][1])

    def test_general_controller_cancel_is_ordinary_general_text_after_direct_create(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", effort="high", sandbox="yolo"),
                    "Topic agent ready.",
                    json.dumps({"action": "reply", "text": "No pending proposal exists."}),
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            gateway.handle_message(self._message("cancel", chat_id="-1001", chat_type="supergroup"))

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.5 high | yolo")])
            self.assertIn("No pending proposal exists.", telegram.messages[-1][1])

    def test_general_controller_new_create_request_creates_another_topic_without_pending_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            salona = root / "salona"
            salona.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", effort="high", sandbox="yolo"),
                    "Topic agent ready.",
                    self._create_action("salona", model="gpt-5.4-mini", effort="low", sandbox="constrained"),
                    "Topic agent ready.",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make a new topic in kitia with gpt 5.5 high yolo",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            gateway.handle_message(
                self._message(
                    "actually make an agent in salona with gpt 5.4 mini low constrained",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            self.assertEqual(
                telegram.created_topics,
                [
                    ("-1001", "kitia | gpt-5.5 high | yolo"),
                    ("-1001", "salona | gpt-5.4-mini low | constrained"),
                ],
            )
            self.assertEqual(store.load_active_workspace("-1001:thread:51"), "salona")
            self.assertEqual(codex.runs[-1], ("gpt-5.4-mini", "low", salona.resolve(), "read-only"))

    def test_general_controller_memory_redacts_local_paths_from_errors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(self._create_action("missing", effort="high", sandbox="yolo"))
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make me a session in missing folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertIn(str(root), telegram.messages[-1][1])
            stored = "\n".join(turn.text for turn in store.load("-1001"))
            self.assertIn("under <path>", stored)
            self.assertNotIn(str(root), stored)

    def test_general_forum_message_rejects_unrecognized_session_request(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "reply", "text": "Which workspace, model, thinking, and sandbox should I use?"}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )

            self.assertEqual(telegram.created_topics, [])
            self.assertEqual(len(codex.prompts), 1)
            self.assertIn("Which workspace", telegram.messages[0][1])
            self.assertEqual(telegram.messages[0][2], 9)

    def test_general_controller_memory_redacts_sensitive_values(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(
                json.dumps(
                    {
                        "action": "reply",
                        "text": (
                            "I will not store api_hash=abcdefabcdefabcd or "
                            "/tmp/bot.env or -1001234567890."
                        ),
                    }
                )
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    (
                        "token=123456:abcdefghijklmnopqrstuvwxyz1234567890 use /tmp/bot.env and "
                        "admin.session in -1001234567890 from `/home/example/codex-telegram` "
                        "and (/tmp)"
                    ),
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            stored = "\n".join(turn.text for turn in store.load("-1001"))
            self.assertIn("token=<redacted>", stored)
            self.assertIn("api_hash=<redacted>", stored)
            self.assertIn("<env-file>", stored)
            self.assertIn("<session-file>", stored)
            self.assertIn("<path>", stored)
            self.assertIn("<chat>", stored)
            self.assertNotIn("123456:abcdefghijklmnopqrstuvwxyz1234567890", stored)
            self.assertNotIn("abcdefabcdefabcd", stored)
            self.assertNotIn("/tmp/bot.env", stored)
            self.assertNotIn("admin.session", stored)
            self.assertNotIn("/home/example/codex-telegram", stored)
            self.assertNotIn("/tmp", stored)
            self.assertNotIn("-1001234567890", stored)

    def test_general_controller_prompt_includes_recent_general_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    json.dumps({"action": "reply", "text": "Which workspace should I use?"}),
                    json.dumps({"action": "reply", "text": "You asked me to create a topic."}),
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message("please create a new topic", chat_id="-1001", chat_type="supergroup")
            )
            gateway.handle_message(
                self._message("what did I ask you to do?", chat_id="-1001", chat_type="supergroup")
            )

            self.assertIn("Recent General-chat controller context:", codex.prompts[1])
            self.assertIn("User: please create a new topic", codex.prompts[1])
            self.assertIn("Assistant: Controller reply: Which workspace should I use?", codex.prompts[1])
            self.assertEqual(
                [turn.text for turn in store.load("-1001")],
                [
                    "please create a new topic",
                    "Controller reply: Which workspace should I use?",
                    "what did I ask you to do?",
                    "Controller reply: You asked me to create a topic.",
                ],
            )

    def test_general_controller_reply_is_truncated_to_telegram_limit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            telegram = FakeTelegram()
            codex = FakeCodex(json.dumps({"action": "reply", "text": "a" * 13000}))
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(root / ".state"),
            )

            gateway.handle_message(
                self._message("hello there", chat_id="-1001", chat_type="supergroup")
            )

            self.assertLess(len(telegram.messages[0][1]), 13000)
            self.assertIn("[Response truncated by MAX_TELEGRAM_RESPONSE_CHARS.]", telegram.messages[0][1])

    def test_general_forum_message_reports_topic_creation_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            telegram.fail_create_forum_topic = True
            codex = FakeCodex(self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"))
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(len(codex.prompts), 1)
            self.assertEqual(store.load_topic_session("-1001:thread:50"), None)
            self.assertIn("could not create a forum topic", telegram.messages[0][1])
            self.assertEqual(telegram.messages[0][2], 9)

    def test_general_forum_message_prefers_longest_model_match(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kitia").mkdir()
            telegram = FakeTelegram()
            store = SessionStore(root / ".state")
            codex = FakeCodex(
                [
                    self._create_action(
                        "kitia",
                        model="gpt-5.4-mini",
                        effort="high",
                        sandbox="yolo",
                        topic_name="kitia | gpt-5.4-mini high | yolo",
                    ),
                    "Topic agent ready.",
                ]
            )
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=PrefixModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.4 mini high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )

            self.assertEqual(telegram.created_topics, [("-1001", "kitia | gpt-5.4-mini high | yolo")])
            preference = store.load_model_preference("-1001:thread:50")
            self.assertIsNotNone(preference)
            assert preference is not None
            self.assertEqual(preference.model, "gpt-5.4-mini")

    def test_created_topic_uses_isolated_session_state_for_next_message(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            other = root / "other"
            other.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                    "finished",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            store.save_active_workspace("-1001", "other")
            gateway.handle_message(
                self._message(
                    "work only in this topic",
                    chat_id="-1001",
                    chat_type="supergroup",
                    message_thread_id=50,
                )
            )

            self.assertEqual(codex.runs[-1], ("gpt-5.5", "high", kitia.resolve(), "yolo"))
            self.assertIn("message_thread_id=50", codex.prompts[-1])
            self.assertIn(f"workspace={kitia.resolve()}", codex.prompts[-1])
            self.assertEqual(telegram.message_threads[-1], 50)

    def test_two_created_topics_keep_independent_runtime_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            kitia = root / "kitia"
            kitia.mkdir()
            salona = root / "salona"
            salona.mkdir()
            telegram = FakeTelegram()
            codex = FakeCodex(
                [
                    self._create_action("kitia", topic_name="kitia | gpt-5.5 high | yolo"),
                    "Topic agent ready.",
                    self._create_action(
                        "salona",
                        model="gpt-5.4-mini",
                        effort="low",
                        sandbox="constrained",
                        topic_name="salona | gpt-5.4-mini low | constrained",
                    ),
                    "Topic agent ready.",
                    "finished",
                    "finished",
                ]
            )
            store = SessionStore(root / ".state")
            gateway = CodexTelegramGateway(
                settings=self._settings(root),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(
                self._message(
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            gateway.handle_message(
                self._message(
                    "make me an agent in salona folder with gpt 5.4 mini low in constrained mode",
                    chat_id="-1001",
                    chat_type="supergroup",
                )
            )
            gateway.handle_message(
                self._message(
                    "kitia task",
                    chat_id="-1001",
                    chat_type="supergroup",
                    message_thread_id=50,
                )
            )
            gateway.handle_message(
                self._message(
                    "salona task",
                    chat_id="-1001",
                    chat_type="supergroup",
                    message_thread_id=51,
                )
            )

            self.assertEqual(
                telegram.created_topics,
                [
                    ("-1001", "kitia | gpt-5.5 high | yolo"),
                    ("-1001", "salona | gpt-5.4-mini low | constrained"),
                ],
            )
            self.assertEqual(codex.runs[-2], ("gpt-5.5", "high", kitia.resolve(), "yolo"))
            self.assertEqual(codex.runs[-1], ("gpt-5.4-mini", "low", salona.resolve(), "constrained"))
            self.assertIn("Current Telegram message:\nkitia task", codex.prompts[-2])
            self.assertNotIn("salona task", codex.prompts[-2])
            self.assertIn("Current Telegram message:\nsalona task", codex.prompts[-1])
            self.assertNotIn("kitia task", codex.prompts[-1])
            self.assertEqual(
                [turn.text for turn in store.load("-1001:thread:50")],
                ["Topic agent ready.", "kitia task", "finished"],
            )
            self.assertEqual(
                [turn.text for turn in store.load("-1001:thread:51")],
                ["Topic agent ready.", "salona task", "finished"],
            )
            general_turns = store.load("-1001")
            self.assertEqual(
                [turn.role for turn in general_turns],
                ["user", "assistant", "user", "assistant"],
            )
            self.assertEqual(
                [turn.text for turn in general_turns if turn.role == "user"],
                [
                    "make me a session in kitia folder with gpt 5.5 high in yolo mode",
                    "make me an agent in salona folder with gpt 5.4 mini low in constrained mode",
                ],
            )
            self.assertTrue(general_turns[1].text.startswith("Controller action succeeded: "))
            self.assertIn('"action": "create_topic_session"', general_turns[1].text)
            self.assertIn('"workspace": "kitia"', general_turns[1].text)
            self.assertTrue(general_turns[3].text.startswith("Controller action succeeded: "))
            self.assertIn('"action": "create_topic_session"', general_turns[3].text)
            self.assertIn('"workspace": "salona"', general_turns[3].text)

    def test_forum_topic_model_preference_is_separate_from_group_default(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(
                self._callback("effort:gpt-5.5:xhigh", chat_id="-1001", message_thread_id=7)
            )
            gateway.handle_message(self._message("topic task", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("general task", chat_id="-1001"))

            self.assertEqual(store.load_model_preference("-1001:thread:7").model, "gpt-5.5")
            self.assertIsNone(store.load_model_preference("-1001"))
            self.assertEqual(codex.runs[-2][0:2], ("gpt-5.5", "xhigh"))
            self.assertEqual(codex.runs[-1][0:2], (None, None))

    def test_callbacks_in_two_topics_keep_preferences_separate(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_callback(
                self._callback("effort:gpt-5.5:xhigh", chat_id="-1001", message_thread_id=7)
            )
            gateway.handle_callback(
                self._callback("effort:gpt-5.4-mini:low", chat_id="-1001", message_thread_id=8)
            )
            gateway.handle_callback(
                self._callback("sandbox:yolo", chat_id="-1001", message_thread_id=7)
            )
            gateway.handle_callback(
                self._callback("sandbox:constrained", chat_id="-1001", message_thread_id=8)
            )
            gateway.handle_message(self._message("topic seven", chat_id="-1001", message_thread_id=7))
            gateway.handle_message(self._message("topic eight", chat_id="-1001", message_thread_id=8))

            self.assertEqual(codex.runs[-2][0:2], ("gpt-5.5", "xhigh"))
            self.assertEqual(codex.runs[-2][3], "yolo")
            self.assertEqual(codex.runs[-1][0:2], ("gpt-5.4-mini", "low"))
            self.assertEqual(codex.runs[-1][3], "constrained")
            self.assertIsNone(store.load_model_preference("-1001"))
            self.assertIsNone(store.load_sandbox_mode("-1001"))

    def test_typing_indicator_failure_does_not_drop_request(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            telegram.fail_chat_action = True
            codex = FakeCodex("finished")
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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

    def test_same_message_id_in_different_topics_is_not_treated_as_duplicate(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=SessionStore(Path(tmp) / "state"),
            )

            def update(update_id: int, thread_id: int, text: str) -> dict:
                return {
                    "update_id": update_id,
                    "message": {
                        "message_id": 9,
                        "message_thread_id": thread_id,
                        "text": text,
                        "chat": {"id": -1001, "type": "supergroup"},
                        "from": {"id": 42, "username": "nima"},
                    },
                }

            gateway.handle_update(update(10, 7, "topic seven first"))
            gateway.handle_update(update(11, 8, "topic eight first"))
            gateway.handle_update(update(12, 7, "topic seven duplicate"))

            self.assertEqual(len(codex.prompts), 2)
            self.assertIn("topic seven first", codex.prompts[0])
            self.assertIn("topic eight first", codex.prompts[1])
            self.assertNotIn("topic seven duplicate", "\n".join(codex.prompts))
            self.assertEqual(telegram.message_threads[-2:], [7, 8])

    def test_unauthorized_message_is_ignored(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex()
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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
            gateway = CodexTelegramGateway(
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
            self.assertEqual(codex.runs[-1], ("gpt-5.5", "xhigh", Path(tmp).resolve(), None))
            self.assertIn("Selected GPT-5.5", telegram.edits[0][2])

    def test_unavailable_saved_model_is_ignored_and_cleared(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            store.save_model_preference("100", model="gpt-5.2", reasoning_effort="medium")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=FakeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do the task"))

            self.assertEqual(codex.runs[-1], (None, None, Path(tmp).resolve(), None))
            self.assertIsNone(store.load_model_preference("100"))

    def test_saved_model_is_kept_when_catalog_is_not_authoritative(self):
        with TemporaryDirectory() as tmp:
            telegram = FakeTelegram()
            codex = FakeCodex("finished")
            store = SessionStore(Path(tmp) / "state")
            store.save_model_preference("100", model="gpt-5.2", reasoning_effort="medium")
            gateway = CodexTelegramGateway(
                settings=self._settings(Path(tmp)),
                telegram=telegram,
                codex=codex,
                model_catalog=NonAuthoritativeModelCatalog(),
                sessions=store,
            )

            gateway.handle_message(self._message("do the task"))

            self.assertEqual(codex.runs[-1], ("gpt-5.2", "medium", Path(tmp).resolve(), None))
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
            gateway = CodexTelegramGateway(
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
