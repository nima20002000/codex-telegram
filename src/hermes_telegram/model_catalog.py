from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    slug: str
    display_name: str
    reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str


FALLBACK_MODELS = (
    ModelChoice("gpt-5.5", "GPT-5.5", ("low", "medium", "high", "xhigh"), "medium"),
    ModelChoice("gpt-5.4", "GPT-5.4", ("low", "medium", "high", "xhigh"), "medium"),
    ModelChoice("gpt-5.4-mini", "GPT-5.4-Mini", ("low", "medium", "high", "xhigh"), "medium"),
)


class CodexModelCatalog:
    def __init__(self, codex_command: str, *, timeout_seconds: int = 10):
        self._codex_command = codex_command
        self._timeout_seconds = timeout_seconds

    def list_models(self) -> tuple[ModelChoice, ...]:
        try:
            completed = subprocess.run(
                [self._codex_command, "debug", "models"],
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except Exception:
            return FALLBACK_MODELS
        if completed.returncode != 0:
            return FALLBACK_MODELS
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return FALLBACK_MODELS
        raw_models = payload.get("models", [])
        if not isinstance(raw_models, list):
            return FALLBACK_MODELS

        choices: list[ModelChoice] = []
        for item in raw_models:
            if not isinstance(item, dict) or item.get("visibility") != "list":
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            display_name = item.get("display_name")
            if not isinstance(display_name, str) or not display_name:
                display_name = slug
            efforts = []
            for level in item.get("supported_reasoning_levels", []):
                if isinstance(level, dict) and isinstance(level.get("effort"), str):
                    efforts.append(level["effort"])
            default_effort = item.get("default_reasoning_level")
            if not isinstance(default_effort, str) or default_effort not in efforts:
                default_effort = efforts[0] if efforts else "medium"
            choices.append(ModelChoice(slug, display_name, tuple(efforts or ("medium",)), default_effort))
        return tuple(choices) or FALLBACK_MODELS

    def get_model(self, slug: str) -> ModelChoice | None:
        for model in self.list_models():
            if model.slug == slug:
                return model
        return None
