from __future__ import annotations

import json
import subprocess
import time
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
    def __init__(self, codex_command: str, *, timeout_seconds: int = 10, cache_ttl_seconds: int = 300):
        self._codex_command = codex_command
        self._timeout_seconds = timeout_seconds
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: tuple[ModelChoice, ...] | None = None
        self._cache_loaded_at: float | None = None
        self._authoritative = False

    def _fallback_models(self) -> tuple[ModelChoice, ...]:
        self._authoritative = False
        return self._cache or FALLBACK_MODELS

    def list_models(self) -> tuple[ModelChoice, ...]:
        now = time.monotonic()
        if (
            self._cache is not None
            and self._cache_loaded_at is not None
            and now - self._cache_loaded_at < self._cache_ttl_seconds
        ):
            return self._cache
        try:
            completed = subprocess.run(
                [self._codex_command, "debug", "models"],
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except Exception:
            return self._fallback_models()
        if completed.returncode != 0:
            return self._fallback_models()
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return self._fallback_models()
        raw_models = payload.get("models", [])
        if not isinstance(raw_models, list):
            return self._fallback_models()

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
        self._cache = tuple(choices) or FALLBACK_MODELS
        self._cache_loaded_at = now
        self._authoritative = True
        return self._cache

    def get_model(self, slug: str) -> ModelChoice | None:
        for model in self.list_models():
            if model.slug == slug:
                return model
        return None

    def is_authoritative(self) -> bool:
        self.list_models()
        return self._authoritative
