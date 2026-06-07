from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from codex_telegram.e2e_harness import (
    BotIdentity,
    CheckResult,
    ServiceStatus,
    _bot_reply_chain_after,
    _ui_long_results,
    _ui_parity_results,
    _wait_for_bot_reply,
    bot_api_chat_id,
    bot_lookup_reference,
    check_service,
    format_check,
    message_replies_to,
    parse_allowed_chats,
    parse_allowed_users,
    parse_app_credentials,
    sanitize_detail,
)


class E2EHarnessTests(unittest.TestCase):
    def test_parse_app_credentials(self):
        credentials = parse_app_credentials(
            "\n".join(
                [
                    "App api_id: 12345",
                    "App api_hash: abcdefabcdefabcd",
                ]
            )
        )

        self.assertEqual(credentials.api_id, 12345)
        self.assertEqual(credentials.api_hash, "abcdefabcdefabcd")

    def test_parse_app_credentials_requires_both_values(self):
        with self.assertRaisesRegex(ValueError, "api_id"):
            parse_app_credentials("App api_id: 12345")

    def test_parse_allowed_users(self):
        self.assertEqual(parse_allowed_users("42, 99,,100"), frozenset({42, 99, 100}))

    def test_parse_allowed_chats(self):
        self.assertEqual(parse_allowed_chats("-1001, 200,,abc"), frozenset({"-1001", "200", "abc"}))

    def test_bot_api_chat_id_formats_forum_supergroup(self):
        entity = type("Channel", (), {"id": 1234567890, "megagroup": True})()

        self.assertEqual(bot_api_chat_id(entity), "-1001234567890")

    def test_message_replies_to_supports_direct_and_nested_reply_ids(self):
        direct = type("Message", (), {"reply_to_msg_id": 10})()
        nested = type("Message", (), {"reply_to": type("Reply", (), {"reply_to_msg_id": 11})()})()
        unrelated = type("Message", (), {"reply_to_msg_id": 12})()

        self.assertTrue(message_replies_to(direct, 10))
        self.assertTrue(message_replies_to(nested, 11))
        self.assertFalse(message_replies_to(unrelated, 10))

    def test_sanitize_detail_redacts_secrets_and_large_ids(self):
        detail = "token abc:secret hash deadbeef user 1234567890 chat -1001234567890"

        sanitized = sanitize_detail(detail, token="abc:secret", api_hash="deadbeef")

        self.assertNotIn("abc:secret", sanitized)
        self.assertNotIn("deadbeef", sanitized)
        self.assertNotIn("1234567890", sanitized)
        self.assertNotIn("-1001234567890", sanitized)
        self.assertIn("<redacted>", sanitized)
        self.assertIn("<id>", sanitized)
        self.assertIn("<chat>", sanitized)

    def test_check_service_validates_expected_checkout(self):
        status = ServiceStatus(
            active_state="active",
            sub_state="running",
            main_pid="123",
            working_directory=Path("/repo"),
            branch="feature/add-feature",
            commit="abc1234",
        )

        checks = check_service(
            status,
            expected_workdir=Path("/repo"),
            expected_branch="feature/add-feature",
            expected_commit="abc1234",
        )

        self.assertTrue(all(check.passed for check in checks))

    def test_check_service_reports_wrong_branch(self):
        status = ServiceStatus(
            active_state="active",
            sub_state="running",
            main_pid="123",
            working_directory=Path("/repo"),
            branch="main",
            commit="abc1234",
        )

        checks = check_service(status, expected_branch="feature/add-feature")

        self.assertFalse([check for check in checks if check.name == "service branch"][0].passed)

    def test_format_check(self):
        self.assertEqual(format_check(CheckResult("service", True, "running")), "[PASS] service: running")

    def test_ui_parity_results_require_rendered_entities(self):
        bold = type("MessageEntityBold", (), {})()
        code = type("MessageEntityCode", (), {})()
        italic = type("MessageEntityItalic", (), {})()
        pre = type("MessageEntityPre", (), {})()
        text_url = type("MessageEntityTextUrl", (), {})()
        reply = type(
            "Message",
            (),
            {
                "id": 10,
                "raw_text": "nim-ui-parity-1\nUI Heading\nThis has bold, italic, inline_code.\nAlpha\n• Status: Done (1/2)",
                "entities": [bold, code, italic, pre, text_url],
            },
        )()
        continuation = type(
            "Message",
            (),
            {
                "id": 11,
                "raw_text": "LongSegment LongSegment",
                "entities": [],
            },
        )()

        checks = _ui_parity_results(reply, "nim-ui-parity-1", related_messages=[reply, continuation])

        self.assertTrue(all(check.passed for check in checks), checks)

    def test_ui_parity_results_require_bold_and_code_entities(self):
        code = type("MessageEntityCode", (), {})()
        reply = type(
            "Message",
            (),
            {
                "id": 10,
                "raw_text": "nim-ui-parity-1\nUI Heading\nThis has bold, inline_code.\nAlpha\n• Status: Done",
                "entities": [code],
            },
        )()

        checks = _ui_parity_results(reply, "nim-ui-parity-1", related_messages=[reply])
        entity_check = next(check for check in checks if check.name == "ui parity entities")

        self.assertFalse(entity_check.passed)

    def test_ui_long_results_require_part_indicators(self):
        reply = type("Message", (), {"id": 10, "raw_text": "nim-ui-long-1 LongSegment (1/2)", "entities": []})()
        continuation = type("Message", (), {"id": 11, "raw_text": "LongSegment (2/2)", "entities": []})()

        checks = _ui_long_results(reply, "nim-ui-long-1", related_messages=[reply, continuation])

        self.assertTrue(all(check.passed for check in checks), checks)

    def test_bot_reply_chain_after_filters_unrelated_bot_messages(self):
        root = type("Message", (), {"id": 12, "sender_id": 7, "reply_to_msg_id": 5, "raw_text": "marker (1/2)"})()
        continuation = type("Message", (), {"id": 13, "sender_id": 7, "reply_to_msg_id": 12, "raw_text": "continued (2/2)"})()
        unrelated = type("Message", (), {"id": 14, "sender_id": 7, "reply_to_msg_id": 99, "raw_text": "unrelated"})()

        class Client:
            async def iter_messages(self, entity, limit):
                yield unrelated
                yield continuation
                yield root

        chain = asyncio.run(
            _bot_reply_chain_after(
                Client(),
                object(),
                7,
                root_message_id=12,
                after_id=5,
            )
        )

        self.assertEqual(chain, [root, continuation])

    def test_wait_for_bot_reply_can_filter_by_marker_text(self):
        progress = type("Message", (), {"id": 11, "sender_id": 7, "reply_to_msg_id": 5, "raw_text": "Working..."})()
        final = type("Message", (), {"id": 12, "sender_id": 7, "reply_to_msg_id": 5, "raw_text": "done marker-123"})()

        class Client:
            async def iter_messages(self, entity, limit):
                yield progress
                yield final

        reply = asyncio.run(
            _wait_for_bot_reply(
                Client(),
                object(),
                7,
                after_id=5,
                timeout_seconds=1,
                text_contains="marker-123",
            )
        )

        self.assertIs(reply, final)

    def test_bot_lookup_reference_prefers_username(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="botname", first_name="Bot")), "@botname")

    def test_bot_lookup_reference_falls_back_to_id(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="", first_name="Bot")), 123456789)


if __name__ == "__main__":
    unittest.main()
