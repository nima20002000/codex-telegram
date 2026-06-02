from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_telegram.config import Settings, load_env_file


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

    def test_settings_requires_token(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
            Settings.from_env(environ={})

    def test_empty_environ_does_not_read_ambient_environment(self):
        with self.assertRaisesRegex(ValueError, "TELEGRAM_BOT_TOKEN"):
            Settings.from_env(
                environ={},
                default_workdir=Path.cwd(),
            )


if __name__ == "__main__":
    unittest.main()
