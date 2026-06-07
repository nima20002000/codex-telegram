from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from codex_telegram.e2e_harness import (
    BotIdentity,
    CheckResult,
    ServiceStatus,
    _bot_messages_after,
    _bot_reply_chain_after,
    _keyboard_results,
    _raw_typing_update_from_bot,
    _runtime_ux_results,
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

    def test_bot_messages_after_returns_sorted_bot_messages(self):
        first = type("Message", (), {"id": 12, "sender_id": 7, "raw_text": "progress"})()
        second = type("Message", (), {"id": 13, "sender_id": 7, "raw_text": "final"})()
        unrelated = type("Message", (), {"id": 14, "sender_id": 9, "raw_text": "unrelated"})()

        class Client:
            async def iter_messages(self, entity, limit):
                yield unrelated
                yield second
                yield first

        messages = asyncio.run(_bot_messages_after(Client(), object(), 7, after_id=5))

        self.assertEqual(messages, [first, second])

    def test_raw_typing_update_from_bot_matches_user_ids(self):
        matching = type("UpdateChannelUserTyping", (), {"from_id": type("PeerUser", (), {"user_id": 7})()})()
        other = type("UpdateChannelUserTyping", (), {"from_id": type("PeerUser", (), {"user_id": 8})()})()
        no_id = type("UpdateUserTyping", (), {})()
        non_typing = type("UpdateNewMessage", (), {"user_id": 7})()

        self.assertTrue(_raw_typing_update_from_bot(matching, 7))
        self.assertTrue(_raw_typing_update_from_bot(no_id, 7))
        self.assertFalse(_raw_typing_update_from_bot(other, 7))
        self.assertFalse(_raw_typing_update_from_bot(non_typing, 7))

    def test_raw_typing_update_from_bot_scopes_private_chat_updates(self):
        private_matching = type("UpdateUserTyping", (), {"user_id": 7})()
        private_other = type("UpdateUserTyping", (), {"user_id": 8})()
        group_matching = type("UpdateChannelUserTyping", (), {"from_id": type("PeerUser", (), {"user_id": 7})()})()

        self.assertTrue(_raw_typing_update_from_bot(private_matching, 7, private_chat=True))
        self.assertFalse(_raw_typing_update_from_bot(private_other, 7, private_chat=True))
        self.assertFalse(_raw_typing_update_from_bot(group_matching, 7, private_chat=True))

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

    def test_runtime_ux_results_require_typing_progress_and_clean_list(self):
        progress = type("Message", (), {"id": 11, "raw_text": "🖥 terminal: sleep 6"})()
        final = type(
            "Message",
            (),
            {
                "id": 12,
                "raw_text": "nim-runtime-ux-1\n\nFolders on your Desktop:\n\n- .agents\n- .git\n- codex-telegram\n- tailwind",
            },
        )()

        checks = _runtime_ux_results(final, "nim-runtime-ux-1", bot_messages=[progress, final], typing_seen=True)

        self.assertTrue(all(check.passed for check in checks), checks)

    def test_runtime_ux_results_fail_dense_list(self):
        progress = type("Message", (), {"id": 11, "raw_text": "🖥 terminal: sleep 6"})()
        final = type(
            "Message",
            (),
            {
                "id": 12,
                "raw_text": "nim-runtime-ux-1\n\nFolders on your Desktop:\n\n.agents, .git, codex-telegram, tailwind",
            },
        )()

        checks = _runtime_ux_results(final, "nim-runtime-ux-1", bot_messages=[progress, final], typing_seen=True)
        list_check = next(check for check in checks if check.name == "runtime ux list shaping")

        self.assertFalse(list_check.passed)

    def test_keyboard_results_accept_telethon_buttons(self):
        reply = type("Message", (), {"id": 22, "buttons": [[object()]], "reply_markup": None})()

        checks = _keyboard_results("/models", reply)

        self.assertTrue(all(check.passed for check in checks), checks)

    def test_keyboard_results_accept_reply_markup_rows(self):
        reply_markup = type("ReplyMarkup", (), {"rows": [object()]})()
        reply = type("Message", (), {"id": 22, "buttons": None, "reply_markup": reply_markup})()

        checks = _keyboard_results("/sandbox", reply)

        self.assertTrue(all(check.passed for check in checks), checks)

    def test_keyboard_results_fail_missing_buttons(self):
        reply = type("Message", (), {"id": 22, "buttons": None, "reply_markup": None})()

        checks = _keyboard_results("/workspace", reply)

        self.assertFalse(checks[0].passed)

    def test_bot_lookup_reference_prefers_username(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="botname", first_name="Bot")), "@botname")

    def test_bot_lookup_reference_falls_back_to_id(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="", first_name="Bot")), 123456789)


if __name__ == "__main__":
    unittest.main()
