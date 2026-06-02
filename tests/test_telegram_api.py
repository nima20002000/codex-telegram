from __future__ import annotations

import unittest

from hermes_telegram.telegram_api import parse_message_update, split_telegram_text


class TelegramAPITests(unittest.TestCase):
    def test_parse_message_update(self):
        message = parse_message_update(
            {
                "update_id": 99,
                "message": {
                    "message_id": 7,
                    "text": "hello",
                    "chat": {"id": -100, "type": "private"},
                    "from": {"id": 42, "username": "nima"},
                },
            }
        )

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.update_id, 99)
        self.assertEqual(message.chat_id, "-100")
        self.assertEqual(message.user_id, 42)
        self.assertEqual(message.text, "hello")

    def test_parse_ignores_non_text_update(self):
        self.assertIsNone(parse_message_update({"update_id": 1, "message": {"chat": {"id": 1}}}))

    def test_parse_ignores_edited_message_update(self):
        self.assertIsNone(
            parse_message_update(
                {
                    "update_id": 2,
                    "edited_message": {
                        "message_id": 8,
                        "text": "edited",
                        "chat": {"id": 1},
                        "from": {"id": 42},
                    },
                }
            )
        )

    def test_split_telegram_text_preserves_content(self):
        text = "a" * 100 + " \n    " + "b" * 100
        chunks = split_telegram_text(text, limit=120)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)


if __name__ == "__main__":
    unittest.main()
