from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codex_telegram.config import Settings, load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_parses_basic_dotenv(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "TELEGRAM_BOT_TOKEN='abc'",
                        'CODEX_MODEL="gpt-5"',
                        "EMPTY=",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_env_file(path),
                {
                    "TELEGRAM_BOT_TOKEN": "abc",
                    "CODEX_MODEL": "gpt-5",
                    "EMPTY": "",
                },
            )

    def test_settings_from_env_uses_file_and_environ(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "repo"
            workdir.mkdir()
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "TELEGRAM_BOT_TOKEN=file-token",
                        "TELEGRAM_ALLOWED_USERS=123,456",
                        f"CODEX_WORKDIR={workdir}",
                    ]
                ),
                encoding="utf-8",
            )

            settings = Settings.from_env(
                env_file=env_file,
                environ={"TELEGRAM_BOT_TOKEN": "env-token"},
            )

            self.assertEqual(settings.bot_token, "env-token")
            self.assertEqual(settings.allowed_users, frozenset({123, 456}))
            self.assertEqual(settings.codex_workdir, workdir.resolve())
            self.assertFalse(settings.telegram_disable_link_previews)

    def test_legacy_state_dir_env_var_is_used_as_fallback(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "repo"
            state_dir = Path(tmp) / "state"
            workdir.mkdir()

            settings = Settings.from_env(
                environ={
                    "TELEGRAM_BOT_TOKEN": "token",
                    "CODEX_WORKDIR": str(workdir),
                    "HERMES_TELEGRAM_STATE_DIR": str(state_dir),
                }
            )

            self.assertEqual(settings.state_dir, state_dir.resolve())

    def test_settings_requires_token(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
            Settings.from_env(environ={})

    def test_settings_can_disable_telegram_link_previews(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "repo"
            workdir.mkdir()

            settings = Settings.from_env(
                environ={
                    "TELEGRAM_BOT_TOKEN": "token",
                    "CODEX_WORKDIR": str(workdir),
                    "TELEGRAM_DISABLE_LINK_PREVIEWS": "true",
                }
            )

            self.assertTrue(settings.telegram_disable_link_previews)

    def test_settings_rejects_invalid_boolean_link_preview_flag(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_DISABLE_LINK_PREVIEWS"):
            Settings.from_env(
                environ={
                    "TELEGRAM_BOT_TOKEN": "token",
                    "TELEGRAM_DISABLE_LINK_PREVIEWS": "sometimes",
                }
            )

    def test_empty_environ_does_not_read_ambient_environment(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
            Settings.from_env(
                environ={},
                default_workdir=Path.cwd(),
            )


if __name__ == "__main__":
    unittest.main()
