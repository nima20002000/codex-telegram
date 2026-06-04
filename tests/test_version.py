from __future__ import annotations

import unittest
from pathlib import Path

import codex_telegram


def read_pyproject_version(path: Path) -> str:
    in_project = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = False
        if in_project and stripped.startswith("version"):
            _key, value = stripped.split("=", 1)
            return value.strip().strip('"')
    raise AssertionError("Missing [project] version in pyproject.toml")


class VersionTests(unittest.TestCase):
    def test_package_version_matches_pyproject(self):
        self.assertEqual(codex_telegram.__version__, read_pyproject_version(Path("pyproject.toml")))


if __name__ == "__main__":
    unittest.main()
