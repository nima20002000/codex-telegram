from __future__ import annotations

import unittest

from hermes_telegram.telegram_api import TelegramAPI, parse_callback_update, parse_message_update, split_telegram_text


class RecordingTelegramAPI(TelegramAPI):
    def __init__(self):
        super().__init__("token")
        self.requests: list[tuple[str, dict]] = []

    def _request(self, method, params):
        self.requests.append((method, params))
        return {"ok": True, "result": True}


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

    def test_parse_callback_update(self):
        callback = parse_callback_update(
            {
                "update_id": 100,
                "callback_query": {
                    "id": "abc",
                    "data": "effort:gpt-5.5:xhigh",
                    "from": {"id": 42, "username": "nima"},
                    "message": {
                        "message_id": 8,
                        "chat": {"id": 123, "type": "private"},
                    },
                },
            }
        )

        self.assertIsNotNone(callback)
        assert callback is not None
        self.assertEqual(callback.update_id, 100)
        self.assertEqual(callback.callback_query_id, "abc")
        self.assertEqual(callback.chat_id, "123")
        self.assertEqual(callback.user_id, 42)
        self.assertEqual(callback.data, "effort:gpt-5.5:xhigh")

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

    def test_set_bot_commands_uses_short_menu(self):
        telegram = RecordingTelegramAPI()

        telegram.set_bot_commands(
            [
                ("/reset", "Reset"),
                ("/models", "Models"),
                ("/workspace", "Workspace"),
            ]
        )

        self.assertEqual(
            telegram.requests,
            [
                (
                    "setMyCommands",
                    {
                        "commands": [
                            {"command": "reset", "description": "Reset"},
                            {"command": "models", "description": "Models"},
                            {"command": "workspace", "description": "Workspace"},
                        ]
                    },
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
