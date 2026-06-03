from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


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
    ) -> list[str]:
        selected_workdir = workdir or self._settings.codex_workdir
        args = [
            self._settings.codex_command,
            "exec",
            "-C",
            str(selected_workdir),
            "--sandbox",
            self._settings.codex_sandbox,
            "--output-last-message",
            str(output_path),
            "-",
        ]
        selected_model = model or self._settings.codex_model
        if selected_model:
            args[2:2] = ["--model", selected_model]
        if reasoning_effort:
            args[2:2] = ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if self._settings.codex_profile:
            args[2:2] = ["--profile", self._settings.codex_profile]
        if self._settings.codex_extra_args:
            args[2:2] = list(self._settings.codex_extra_args)
        return args

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workdir: Path | None = None,
    ) -> CodexResult:
        selected_workdir = workdir or self._settings.codex_workdir
        with tempfile.TemporaryDirectory(prefix="hermes-telegram-codex-") as tmpdir:
            output_path = Path(tmpdir) / "last-message.txt"
            command = self._build_command(
                output_path,
                model=model,
                reasoning_effort=reasoning_effort,
                workdir=selected_workdir,
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
                text=output_text or "(Codex produced no final response.)",
                returncode=completed.returncode,
                stderr=completed.stderr.strip(),
            )
