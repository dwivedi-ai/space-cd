"""Intelligence profiles are read from ``config/agents/<name>/intelligence.json``.

The file lists the setups (model + effort) a request can run with, plus the
fallback ``default``. It is checked on load, re-read when it changes, and an
agent without a usable file simply runs without profiles.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.intelligence import profiles

ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "medium", "high"],
    "profiles": [
        {"id": "light", "model": None, "effort": "low", "use_when": "questions; no code changes"},
        {"id": "deep", "model": "claude-opus-5-5[1m]", "effort": "high", "use_when": "work across the codebase"},
    ],
}


def _variant(**changes) -> dict:
    document = copy.deepcopy(VALID)
    document.update(changes)
    return document


def _with_profile(**fields) -> dict:
    document = copy.deepcopy(VALID)
    document["profiles"][0].update(fields)
    return document


class ShippedConfigTests(unittest.TestCase):
    def test_the_claude_code_profiles_load(self) -> None:
        profiles._cache.clear()
        config = profiles.load("claude_code")
        self.assertIsNotNone(config)
        self.assertEqual(config.profile_ids, ("light", "standard", "deep", "research"))
        self.assertEqual(config.efforts, ("low", "medium", "high", "xhigh", "max"))

    def test_the_default_is_the_current_configuration(self) -> None:
        # No flags at all: Claude Code's own settings decide, exactly as before.
        config = profiles.load("claude_code")
        self.assertEqual(config.default, profiles.Setup(model=None, effort=None))

    def test_profiles_vary_effort_not_model(self) -> None:
        config = profiles.load("claude_code")
        self.assertEqual({p.model for p in config.profiles}, {None})
        self.assertEqual(
            {p.id: p.effort for p in config.profiles},
            {"light": "low", "standard": "medium", "deep": "high", "research": "medium"},
        )


class ParseTests(unittest.TestCase):
    def test_a_valid_document(self) -> None:
        config = profiles.parse(VALID, sha256="abc")
        self.assertEqual(config.sha256, "abc")
        self.assertEqual(config.profile("deep").setup,
                         profiles.Setup(model="claude-opus-5-5[1m]", effort="high"))
        self.assertIsNone(config.profile("missing"))

    def test_rejections(self) -> None:
        cases = {
            "not an object": [],
            "wrong schema": _variant(schema=2),
            "no efforts": _variant(efforts=[]),
            "efforts not names": _variant(efforts=["low", "--max"]),
            "default missing": {k: v for k, v in VALID.items() if k != "default"},
            "default effort unknown": _variant(default={"model": None, "effort": "max"}),
            "default model is a flag": _variant(default={"model": "--dangerous", "effort": None}),
            "no profiles": _variant(profiles=[]),
            "profile not an object": _variant(profiles=["light"]),
            "bad id": _with_profile(id="Light Mode"),
            "reserved id": _with_profile(id="unknown"),
            "duplicate id": _with_profile(id="deep"),
            "effort unknown": _with_profile(effort="max"),
            "model with a space": _with_profile(model="opus 5"),
            "model is a flag": _with_profile(model="-p"),
            "empty use_when": _with_profile(use_when="  "),
            "long use_when": _with_profile(use_when="x" * 301),
            "too many profiles": _variant(profiles=[
                {"id": f"p{i}", "model": None, "effort": "low", "use_when": "x"} for i in range(120)
            ]),
        }
        for name, document in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(profiles.ProfileError):
                    profiles.parse(document)


class LoadTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.agents = Path(tmp.name)
        patcher = patch.object(profiles, "_AGENTS_DIR", self.agents)
        patcher.start()
        self.addCleanup(patcher.stop)
        profiles._cache.clear()
        self.addCleanup(profiles._cache.clear)

    def write(self, document, agent: str = "sample_agent", mtime: int | None = None) -> Path:
        path = self.agents / agent / "intelligence.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_an_agent_without_the_file_has_no_profiles(self) -> None:
        self.assertIsNone(profiles.load("sample_agent"))

    def test_an_unsafe_agent_name_is_never_looked_up(self) -> None:
        for name in ("../sample_agent", "a/b", "", "x" * 65):
            with self.subTest(name=name):
                self.assertIsNone(profiles.config_path(name))
                self.assertIsNone(profiles.load(name))

    def test_an_invalid_file_is_logged_and_treated_as_missing(self) -> None:
        self.write("{not json")
        with self.assertLogs(profiles.log, "WARNING"):
            self.assertIsNone(profiles.load("sample_agent"))
        self.write(_variant(schema=9), mtime=1_000)
        with self.assertLogs(profiles.log, "WARNING"):
            self.assertIsNone(profiles.load("sample_agent"))

    def test_an_invalid_file_is_logged_once_per_version(self) -> None:
        self.write(_variant(schema=9))
        with self.assertLogs(profiles.log, "WARNING") as logs:
            profiles.load("sample_agent")
            profiles.load("sample_agent")
        self.assertEqual(len(logs.records), 1)

    def test_the_file_is_reread_when_it_changes(self) -> None:
        self.write(VALID, mtime=1_000)
        first = profiles.load("sample_agent")
        self.assertEqual(first.profile_ids, ("light", "deep"))
        self.assertIs(profiles.load("sample_agent"), first)

        changed = copy.deepcopy(VALID)
        changed["profiles"].pop()
        self.write(changed, mtime=2_000)
        second = profiles.load("sample_agent")
        self.assertEqual(second.profile_ids, ("light",))
        self.assertNotEqual(first.sha256, second.sha256)

    def test_a_removed_file_turns_profiles_off(self) -> None:
        path = self.write(VALID)
        self.assertIsNotNone(profiles.load("sample_agent"))
        path.unlink()
        self.assertIsNone(profiles.load("sample_agent"))


class ModularityTests(unittest.TestCase):
    def test_core_intelligence_code_names_no_agent(self) -> None:
        package = ROOT / "services" / "cowork_agent" / "intelligence"
        for path in sorted(package.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for name in ("claude_code", "openclaw", "hermes"):
                with self.subTest(file=path.name, agent=name):
                    self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
