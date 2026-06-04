from __future__ import annotations

import unittest
from pathlib import Path

from codex_telegram.e2e_harness import (
    BotIdentity,
    CheckResult,
    ServiceStatus,
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

    def test_bot_lookup_reference_prefers_username(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="botname", first_name="Bot")), "@botname")

    def test_bot_lookup_reference_falls_back_to_id(self):
        self.assertEqual(bot_lookup_reference(BotIdentity(user_id=123456789, username="", first_name="Bot")), 123456789)


if __name__ == "__main__":
    unittest.main()
