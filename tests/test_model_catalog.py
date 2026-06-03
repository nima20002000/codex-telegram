from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from codex_telegram.model_catalog import CodexModelCatalog


def _models_payload(*slugs: str) -> str:
    return json.dumps(
        {
            "models": [
                {
                    "slug": slug,
                    "display_name": slug.upper(),
                    "visibility": "list",
                    "default_reasoning_level": "medium",
                    "supported_reasoning_levels": [{"effort": "medium"}],
                }
                for slug in slugs
            ]
        }
    )


class CodexModelCatalogTests(unittest.TestCase):
    def test_successful_lookup_is_cached_until_ttl_expires(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=_models_payload(f"model-{len(calls)}"), stderr="")

        with patch("time.monotonic", side_effect=[10.0, 11.0, 16.0]):
            with patch("subprocess.run", side_effect=fake_run):
                catalog = CodexModelCatalog("codex", cache_ttl_seconds=5)

                self.assertEqual([model.slug for model in catalog.list_models()], ["model-1"])
                self.assertEqual([model.slug for model in catalog.list_models()], ["model-1"])
                self.assertEqual([model.slug for model in catalog.list_models()], ["model-2"])

        self.assertEqual(len(calls), 2)

    def test_failed_refresh_keeps_cache_but_marks_catalog_non_authoritative(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 0, stdout=_models_payload("gpt-5.5"), stderr="")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")

        with patch("time.monotonic", side_effect=[10.0, 11.0, 16.0, 17.0]):
            with patch("subprocess.run", side_effect=fake_run):
                catalog = CodexModelCatalog("codex", cache_ttl_seconds=5)

                self.assertEqual([model.slug for model in catalog.list_models()], ["gpt-5.5"])
                self.assertTrue(catalog.is_authoritative())
                self.assertEqual([model.slug for model in catalog.list_models()], ["gpt-5.5"])
                self.assertFalse(catalog.is_authoritative())


if __name__ == "__main__":
    unittest.main()
