from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_telegram.session_store import SessionStore


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


if __name__ == "__main__":
    unittest.main()
