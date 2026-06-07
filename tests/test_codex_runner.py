from __future__ import annotations

import io
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codex_telegram.codex_runner import CodexRunner
from codex_telegram.config import Settings


class FakePopen:
    def __init__(self, command, *, stdin, stdout, stderr, text, cwd):
        self.command = command
        self.stdin = io.StringIO()
        self.stdout = iter(
            [
                '{"type":"item.started","item":{"type":"command_execution"}}\n',
                '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n',
            ]
        )
        self.stderr = stderr
        self.returncode = 0
        self._polled = False

    def poll(self):
        if not self._polled:
            self._polled = True
            return None
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = 124


class CodexRunnerTests(unittest.TestCase):
    def _settings(self, workdir: Path) -> Settings:
        return Settings(
            bot_token="token",
            allowed_users=frozenset(),
            allowed_chats=frozenset(),
            codex_command="codex",
            codex_workdir=workdir,
            codex_model="gpt-5",
            codex_profile="default",
            codex_sandbox="workspace-write",
            codex_extra_args=("--color", "never"),
            codex_timeout_seconds=10,
            telegram_poll_timeout_seconds=30,
            telegram_request_timeout_seconds=45,
            telegram_disable_link_previews=False,
            max_telegram_response_chars=12000,
            session_history_turns=8,
            state_dir=workdir / ".state",
        )

    def test_runner_invokes_codex_exec_and_reads_last_message(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                seen["kwargs"] = kwargs
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="events", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                result = CodexRunner(self._settings(workdir)).run("do work")

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertEqual(command[:2], ["codex", "exec"])
            self.assertIn("-C", command)
            self.assertIn(str(workdir), command)
            self.assertIn("--model", command)
            self.assertIn("gpt-5", command)
            self.assertEqual(result.text, "final answer")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(seen["kwargs"]["input"], "do work")

    def test_timeout_returns_user_visible_result(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            def fake_run(*args, **kwargs):
                raise subprocess.TimeoutExpired(cmd=args[0], timeout=10, stderr="too slow")

            with patch("subprocess.run", side_effect=fake_run):
                result = CodexRunner(self._settings(workdir)).run("do work")

            self.assertEqual(result.returncode, 124)
            self.assertIn("timed out", result.text)
            self.assertEqual(result.stderr, "too slow")

    def test_runner_applies_per_chat_model_and_reasoning_override(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(workdir)).run(
                    "do work",
                    model="gpt-5.5",
                    reasoning_effort="xhigh",
                )

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--model", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
            self.assertIn("-c", command)
            self.assertIn('model_reasoning_effort="xhigh"', command)

    def test_runner_uses_per_chat_workdir_override(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "avatar"
            selected.mkdir()
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                seen["kwargs"] = kwargs
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(root)).run("do work", workdir=selected)

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertEqual(command[command.index("-C") + 1], str(selected))
            self.assertEqual(seen["kwargs"]["cwd"], str(selected))

    def test_runner_uses_constrained_sandbox_override(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(workdir)).run("do work", sandbox_mode="constrained")

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--sandbox", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "workspace-write")
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)

    def test_runner_uses_read_only_sandbox_override(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(workdir)).run("do work", sandbox_mode="read-only")

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--sandbox", command)
            self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)

    def test_runner_uses_yolo_bypass_flag(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(workdir)).run("do work", sandbox_mode="yolo")

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertNotIn("--sandbox", command)

    def test_runner_applies_per_run_extra_args(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("final answer", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                CodexRunner(self._settings(workdir)).run(
                    "do work",
                    sandbox_mode="read-only",
                    extra_args=("--skip-git-repo-check",),
                )

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--skip-git-repo-check", command)

    def test_runner_streams_json_events_and_reads_last_message(self):
        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            events: list[dict[str, object]] = []
            seen: dict[str, object] = {}

            def fake_popen(command, **kwargs):
                seen["command"] = command
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("streamed final answer", encoding="utf-8")
                return FakePopen(command, **kwargs)

            with patch("subprocess.Popen", side_effect=fake_popen):
                result = CodexRunner(self._settings(workdir)).run(
                    "do work",
                    progress_callback=events.append,
                )

            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertIn("--json", command)
            self.assertEqual(result.text, "streamed final answer")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["type"], "item.started")

    def test_compact_uses_safe_summary_prompt_and_runtime_overrides(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "project"
            selected.mkdir()
            seen: dict[str, object] = {}

            def fake_run(command, **kwargs):
                seen["command"] = command
                seen["kwargs"] = kwargs
                output_path = Path(command[command.index("--output-last-message") + 1])
                output_path.write_text("summary", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_run):
                result = CodexRunner(self._settings(root)).compact(
                    "User: keep context",
                    existing_summary="old summary",
                    model="gpt-5.5",
                    reasoning_effort="high",
                    workdir=selected,
                    sandbox_mode="constrained",
                )

            prompt = seen["kwargs"]["input"]
            self.assertIsInstance(prompt, str)
            assert isinstance(prompt, str)
            self.assertIn("Summarize this Telegram Codex session", prompt)
            self.assertIn("Do not include raw code blocks", prompt)
            self.assertIn("old summary", prompt)
            self.assertIn("User: keep context", prompt)
            self.assertEqual(result.text, "summary")
            command = seen["command"]
            self.assertIsInstance(command, list)
            assert isinstance(command, list)
            self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
            self.assertEqual(command[command.index("-C") + 1], str(selected))


if __name__ == "__main__":
    unittest.main()
