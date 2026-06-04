from __future__ import annotations

import json
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Settings

EMPTY_CODEX_RESPONSE = "(Codex produced no final response.)"


@dataclass(frozen=True)
class CodexResult:
    text: str
    returncode: int
    stderr: str


class CodexRunner:
    def __init__(self, settings: Settings):
        self._settings = settings

    def _build_command(
        self,
        output_path: Path,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: Path | None = None,
        sandbox_mode: str | None = None,
        json_output: bool = False,
        extra_args: tuple[str, ...] = (),
    ) -> list[str]:
        selected_workdir = workdir or self._settings.codex_workdir
        args = [
            self._settings.codex_command,
            "exec",
            "-C",
            str(selected_workdir),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if json_output:
            args.insert(2, "--json")
        if sandbox_mode == "yolo":
            args[2:2] = ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            if sandbox_mode == "constrained":
                sandbox = "workspace-write"
            elif sandbox_mode == "read-only":
                sandbox = "read-only"
            else:
                sandbox = self._settings.codex_sandbox
            args[2:2] = ["--sandbox", sandbox]
        selected_model = model or self._settings.codex_model
        if selected_model:
            args[2:2] = ["--model", selected_model]
        if reasoning_effort:
            args[2:2] = ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if self._settings.codex_profile:
            args[2:2] = ["--profile", self._settings.codex_profile]
        if self._settings.codex_extra_args:
            args[2:2] = list(self._settings.codex_extra_args)
        if extra_args:
            args[2:2] = list(extra_args)
        return args

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: Path | None = None,
        sandbox_mode: str | None = None,
        extra_args: tuple[str, ...] = (),
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> CodexResult:
        selected_workdir = workdir or self._settings.codex_workdir
        with tempfile.TemporaryDirectory(prefix="codex-telegram-codex-") as tmpdir:
            output_path = Path(tmpdir) / "last-message.txt"
            command = self._build_command(
                output_path,
                model=model,
                reasoning_effort=reasoning_effort,
                workdir=selected_workdir,
                sandbox_mode=sandbox_mode,
                json_output=progress_callback is not None,
                extra_args=extra_args,
            )
            if progress_callback is not None:
                return self._run_streaming(
                    command,
                    prompt,
                    output_path,
                    selected_workdir=selected_workdir,
                    progress_callback=progress_callback,
                )
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self._settings.codex_timeout_seconds,
                    cwd=str(selected_workdir),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stderr = ""
                if isinstance(exc.stderr, bytes):
                    stderr = exc.stderr.decode("utf-8", errors="replace")
                elif isinstance(exc.stderr, str):
                    stderr = exc.stderr
                return CodexResult(
                    text=(
                        "Codex timed out after "
                        f"{self._settings.codex_timeout_seconds} seconds. "
                        "Try a narrower request or increase CODEX_TIMEOUT_SECONDS."
                    ),
                    returncode=124,
                    stderr=stderr.strip(),
                )
            output_text = ""
            if output_path.exists():
                output_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
            if not output_text:
                output_text = completed.stdout.strip()
            if completed.returncode != 0:
                error = completed.stderr.strip() or completed.stdout.strip() or "Codex exited with an error."
                output_text = f"Codex failed with exit code {completed.returncode}.\n\n{error}"
            return CodexResult(
                text=output_text or EMPTY_CODEX_RESPONSE,
                returncode=completed.returncode,
                stderr=completed.stderr.strip(),
            )

    def _run_streaming(
        self,
        command: list[str],
        prompt: str,
        output_path: Path,
        *,
        selected_workdir: Path,
        progress_callback: Callable[[dict[str, object]], None],
    ) -> CodexResult:
        stderr_path = output_path.with_name("stderr.txt")
        event_queue: queue.Queue[str] = queue.Queue()
        deadline = time.monotonic() + self._settings.codex_timeout_seconds
        with stderr_path.open("w+", encoding="utf-8") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    cwd=str(selected_workdir),
                )
            except Exception as exc:
                return CodexResult(
                    text=f"Codex failed to start: {exc}",
                    returncode=1,
                    stderr=str(exc),
                )

            assert process.stdin is not None
            process.stdin.write(prompt)
            process.stdin.close()

            def read_stdout() -> None:
                if process.stdout is None:
                    return
                for line in process.stdout:
                    event_queue.put(line)

            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()
            timed_out = False
            while process.poll() is None or not event_queue.empty():
                if time.monotonic() > deadline and process.poll() is None:
                    timed_out = True
                    process.kill()
                    break
                try:
                    line = event_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._handle_json_event_line(line, progress_callback)

            reader.join(timeout=1)
            while not event_queue.empty():
                self._handle_json_event_line(event_queue.get(), progress_callback)
            returncode = process.wait(timeout=5)
            stderr_file.seek(0)
            stderr = stderr_file.read().strip()

        if timed_out:
            return CodexResult(
                text=(
                    "Codex timed out after "
                    f"{self._settings.codex_timeout_seconds} seconds. "
                    "Try a narrower request or increase CODEX_TIMEOUT_SECONDS."
                ),
                returncode=124,
                stderr=stderr,
            )

        output_text = ""
        if output_path.exists():
            output_text = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if returncode != 0:
            error = stderr or output_text or "Codex exited with an error."
            output_text = f"Codex failed with exit code {returncode}.\n\n{error}"
        return CodexResult(
            text=output_text or EMPTY_CODEX_RESPONSE,
            returncode=returncode,
            stderr=stderr,
        )

    @staticmethod
    def _handle_json_event_line(
        line: str,
        progress_callback: Callable[[dict[str, object]], None],
    ) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict):
            progress_callback(event)

    def compact(
        self,
        conversation_context: str,
        *,
        existing_summary: str = "",
        model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: Path | None = None,
        sandbox_mode: str | None = None,
    ) -> CodexResult:
        sections = [
            "Summarize this Telegram Codex session for future context.",
            "Preserve user goals, decisions, constraints, files touched, commands run, and unresolved next steps.",
            "Do not include raw code blocks, raw diffs, secrets, tokens, private chat ids, or long command output.",
            "Write a concise factual summary that can be prepended to a later Codex prompt.",
        ]
        if existing_summary:
            sections.append(f"Existing compact summary:\n{existing_summary}")
        sections.append(f"Recent conversation to compact:\n{conversation_context}")
        return self.run(
            "\n\n".join(sections),
            model=model,
            reasoning_effort=reasoning_effort,
            workdir=workdir,
            sandbox_mode=sandbox_mode,
        )
