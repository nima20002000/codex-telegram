from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codex_telegram.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_append_load_and_reset(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp), history_turns=1)
            store.append("chat/1", "user", "hello")
            store.append("chat/1", "assistant", "hi")
            store.append("chat/1", "user", "next")

            turns = store.load("chat/1")
            self.assertEqual([turn.text for turn in turns], ["hi", "next"])
            self.assertIn("User: next", store.render_recent("chat/1"))

            store.reset("chat/1")
            self.assertEqual(store.load("chat/1"), [])

    def test_processed_message_ledger(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            self.assertFalse(store.was_processed("chat", 10, 1))
            store.mark_processed("chat", 10, 1)
            self.assertTrue(store.was_processed("chat", 10, 1))
            self.assertFalse(store.was_processed("chat", 11, 2))

    def test_model_preference_round_trip(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            self.assertIsNone(store.load_model_preference("chat"))
            store.save_model_preference("chat", model="gpt-5.5", reasoning_effort="xhigh")

            preference = store.load_model_preference("chat")
            self.assertIsNotNone(preference)
            self.assertEqual(preference.model, "gpt-5.5")
            self.assertEqual(preference.reasoning_effort, "xhigh")

            store.clear_model_preference("chat")
            self.assertIsNone(store.load_model_preference("chat"))

    def test_active_workspace_round_trip(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            self.assertEqual(store.load_active_workspace("chat"), "")
            store.save_active_workspace("chat", "avatar/project")
            self.assertEqual(store.load_active_workspace("chat"), "avatar/project")

            store.clear_active_workspace("chat")
            self.assertEqual(store.load_active_workspace("chat"), "")

    def test_sandbox_mode_round_trip(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            self.assertIsNone(store.load_sandbox_mode("chat"))
            store.save_sandbox_mode("chat", "yolo")
            self.assertEqual(store.load_sandbox_mode("chat"), "yolo")
            store.save_sandbox_mode("chat", "constrained")
            self.assertEqual(store.load_sandbox_mode("chat"), "constrained")

    def test_sandbox_mode_rejects_unknown_value(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            with self.assertRaisesRegex(ValueError, "Unsupported sandbox mode"):
                store.save_sandbox_mode("chat", "danger-full-access")

    def test_workspace_token_round_trip(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            token = store.remember_workspace_token("avatar/project")

            self.assertEqual(store.resolve_workspace_token(token), "avatar/project")
            self.assertLessEqual(len(token), 16)

    def test_topic_session_round_trip(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

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

            session = store.load_topic_session("-1001:thread:50")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(session.chat_id, "-1001")
            self.assertEqual(session.message_thread_id, 50)
            self.assertEqual(session.session_key, "-1001:thread:50")
            self.assertEqual(session.topic_name, "kitia | gpt-5.5 high | yolo")
            self.assertEqual(session.workspace, "kitia")
            self.assertEqual(session.model, "gpt-5.5")
            self.assertEqual(session.reasoning_effort, "high")
            self.assertEqual(session.sandbox_mode, "yolo")


if __name__ == "__main__":
    unittest.main()
