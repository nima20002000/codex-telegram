from __future__ import annotations

import unittest
import re

from codex_telegram.telegram_api import TelegramAPI, TelegramAPIError, parse_callback_update, parse_message_update, split_telegram_text


class RecordingTelegramAPI(TelegramAPI):
    def __init__(self):
        super().__init__("token")
        self.requests: list[tuple[str, dict]] = []
        self.responses: dict[str, dict] = {}
        self.errors: dict[str, list[TelegramAPIError]] = {}

    def _request(self, method, params):
        self.requests.append((method, params))
        errors = self.errors.get(method)
        if errors:
            raise errors.pop(0)
        if method in self.responses:
            return self.responses[method]
        return {"ok": True, "result": True}


class TelegramAPITests(unittest.TestCase):
    def test_parse_message_update(self):
        message = parse_message_update(
            {
                "update_id": 99,
                "message": {
                    "message_id": 7,
                    "message_thread_id": 3,
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
        self.assertEqual(message.message_thread_id, 3)

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
                        "message_thread_id": 5,
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
        self.assertEqual(callback.message_thread_id, 5)

    def test_send_message_can_target_forum_topic(self):
        telegram = RecordingTelegramAPI()

        telegram.send_message("123", "**hello**", reply_to_message_id=8, message_thread_id=5)

        self.assertEqual(
            telegram.requests,
            [
                (
                    "sendMessage",
                    {
                        "chat_id": "123",
                        "text": "*hello*",
                        "message_thread_id": 5,
                        "reply_to_message_id": 8,
                        "parse_mode": "MarkdownV2",
                        "disable_web_page_preview": None,
                        "reply_markup": None,
                    },
                )
            ],
        )

    def test_create_forum_topic_returns_thread_id_and_name(self):
        telegram = RecordingTelegramAPI()
        telegram.responses["createForumTopic"] = {
            "ok": True,
            "result": {"message_thread_id": 50, "name": "kitia | gpt-5.5 high | yolo"},
        }

        topic = telegram.create_forum_topic("-1001", "kitia | gpt-5.5 high | yolo")

        self.assertEqual(topic.message_thread_id, 50)
        self.assertEqual(topic.name, "kitia | gpt-5.5 high | yolo")
        self.assertEqual(
            telegram.requests,
            [
                (
                    "createForumTopic",
                    {"chat_id": "-1001", "name": "kitia | gpt-5.5 high | yolo"},
                )
            ],
        )

    def test_group_metadata_methods_send_expected_requests(self):
        telegram = RecordingTelegramAPI()
        telegram.responses["getChat"] = {
            "ok": True,
            "result": {"title": "Dev Group", "type": "supergroup", "is_forum": True},
        }

        chat = telegram.get_chat("-1001")
        telegram.set_chat_title("-1001", "Renamed Group")

        self.assertEqual(chat.title, "Dev Group")
        self.assertEqual(chat.chat_type, "supergroup")
        self.assertTrue(chat.is_forum)
        self.assertEqual(
            telegram.requests,
            [
                ("getChat", {"chat_id": "-1001"}),
                ("setChatTitle", {"chat_id": "-1001", "title": "Renamed Group"}),
            ],
        )

    def test_get_chat_treats_missing_forum_flag_as_false(self):
        telegram = RecordingTelegramAPI()
        telegram.responses["getChat"] = {
            "ok": True,
            "result": {"title": "Plain Group", "type": "supergroup"},
        }

        chat = telegram.get_chat("-1001")

        self.assertEqual(chat.title, "Plain Group")
        self.assertEqual(chat.chat_type, "supergroup")
        self.assertFalse(chat.is_forum)

    def test_topic_lifecycle_methods_send_expected_requests(self):
        telegram = RecordingTelegramAPI()

        telegram.edit_forum_topic("-1001", 50, name="renamed topic")
        telegram.close_forum_topic("-1001", 50)
        telegram.reopen_forum_topic("-1001", 50)
        telegram.delete_forum_topic("-1001", 50)

        self.assertEqual(
            telegram.requests,
            [
                (
                    "editForumTopic",
                    {"chat_id": "-1001", "message_thread_id": 50, "name": "renamed topic"},
                ),
                ("closeForumTopic", {"chat_id": "-1001", "message_thread_id": 50}),
                ("reopenForumTopic", {"chat_id": "-1001", "message_thread_id": 50}),
                ("deleteForumTopic", {"chat_id": "-1001", "message_thread_id": 50}),
            ],
        )

    def test_split_send_message_keeps_forum_topic_on_all_chunks(self):
        telegram = RecordingTelegramAPI()

        telegram.send_message("123", "a" * 5000, message_thread_id=5)

        chunks = [params for method, params in telegram.requests if method == "sendMessage"]
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["message_thread_id"] == 5 for chunk in chunks))
        self.assertTrue(chunks[0]["text"].endswith("\\(1/2\\)"))

    def test_split_send_message_groups_continuations_when_message_ids_are_returned(self):
        telegram = RecordingTelegramAPI()

        def response(message_id):
            return {"ok": True, "result": {"message_id": message_id}}

        telegram.responses["sendMessage"] = response(10)
        calls = 0

        def request(method, params):
            nonlocal calls
            telegram.requests.append((method, params))
            calls += 1
            return response(9 + calls)

        telegram._request = request

        telegram.send_message("123", "a" * 5000, reply_to_message_id=8, message_thread_id=5)

        chunks = [params for method, params in telegram.requests if method == "sendMessage"]
        self.assertEqual(chunks[0]["reply_to_message_id"], 8)
        self.assertEqual(chunks[1]["reply_to_message_id"], 10)

    def test_send_message_retries_markdown_parse_error_as_plain_text(self):
        telegram = RecordingTelegramAPI()
        telegram.errors["sendMessage"] = [TelegramAPIError("Bad Request: can't parse entities")]

        telegram.send_message("123", "**hello**")

        first = telegram.requests[0][1]
        second = telegram.requests[1][1]
        self.assertEqual(first["text"], "*hello*")
        self.assertEqual(first["parse_mode"], "MarkdownV2")
        self.assertEqual(second["text"], "hello")
        self.assertIsNone(second["parse_mode"])

    def test_send_message_does_not_retry_non_markdown_parse_error(self):
        telegram = RecordingTelegramAPI()
        telegram.errors["sendMessage"] = [TelegramAPIError("Bad Request: can't parse reply keyboard markup")]

        with self.assertRaises(TelegramAPIError):
            telegram.send_message("123", "**hello**", reply_markup={"inline_keyboard": "bad"})

        self.assertEqual(len(telegram.requests), 1)
        self.assertEqual(telegram.requests[0][1]["parse_mode"], "MarkdownV2")

    def test_link_previews_are_allowed_by_default_and_can_be_disabled(self):
        default = RecordingTelegramAPI()
        disabled = RecordingTelegramAPI()
        disabled._disable_link_previews = True

        default.send_message("123", "[OpenAI](https://openai.com)")
        disabled.send_message("123", "[OpenAI](https://openai.com)")

        self.assertIsNone(default.requests[0][1]["disable_web_page_preview"])
        self.assertTrue(disabled.requests[0][1]["disable_web_page_preview"])

    def test_edit_message_uses_markdown_and_plain_fallback(self):
        telegram = RecordingTelegramAPI()
        telegram.errors["editMessageText"] = [TelegramAPIError("Bad Request: can't parse entities")]

        telegram.edit_message_text("123", 5, "# Title")

        first = telegram.requests[0][1]
        second = telegram.requests[1][1]
        self.assertEqual(first["text"], "*Title*")
        self.assertEqual(first["parse_mode"], "MarkdownV2")
        self.assertEqual(second["text"], "Title")
        self.assertIsNone(second["parse_mode"])

    def test_send_chat_action_can_target_forum_topic(self):
        telegram = RecordingTelegramAPI()

        telegram.send_chat_action("123", message_thread_id=5)

        self.assertEqual(
            telegram.requests,
            [
                (
                    "sendChatAction",
                    {
                        "chat_id": "123",
                        "action": "typing",
                        "message_thread_id": 5,
                    },
                )
            ],
        )

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
        restored = "".join(re.sub(r" \\\(\d+/\d+\\\)$", "", chunk) for chunk in chunks)
        self.assertEqual(restored, text)

    def test_set_bot_commands_uses_short_menu(self):
        telegram = RecordingTelegramAPI()

        telegram.set_bot_commands(
            [
                ("/reset", "Reset"),
                ("/compact", "Compact context"),
                ("/fast", "Fast mode"),
                ("/goal", "Goal tracking"),
                ("/models", "Models"),
                ("/workspace", "Workspace"),
                ("/sandbox", "Sandbox"),
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
                            {"command": "compact", "description": "Compact context"},
                            {"command": "fast", "description": "Fast mode"},
                            {"command": "goal", "description": "Goal tracking"},
                            {"command": "models", "description": "Models"},
                            {"command": "workspace", "description": "Workspace"},
                            {"command": "sandbox", "description": "Sandbox"},
                        ]
                    },
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
