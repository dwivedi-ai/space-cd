"""A large project gets a category list: 12–40 functional areas of its code (step 6a).

Only git repos with at least XO_CONTEXT_MIN_FILES code files qualify. The list
is drafted once by the agent's tool-free ``oneshot`` capability from the file
list and README, validated, and kept in the project's runtime home, where a
person may edit it. It is never redrafted unless asked.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.adapters.claude_code import oneshot
from services.cowork_agent.intelligence import categories
from utils.commands import CommandResult


def areas(n: int = 12) -> list[dict]:
    return [{"id": f"area_{i}", "description": f"what area {i} does; not area {i + 1}"} for i in range(n)]


def git_repo(path: Path, files: int) -> Path:
    path.mkdir(parents=True)
    for i in range(files):
        (path / "src" / f"m{i % 3}").mkdir(parents=True, exist_ok=True)
        (path / "src" / f"m{i % 3}" / f"f{i}.py").write_text("x = 1\n")
    (path / "README.md").write_text("# Demo\nA demo repository.\n")
    (path / "notes.txt").write_text("not code\n")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    return path


class ParseTests(unittest.TestCase):
    def test_json_is_found_inside_prose_or_fences(self) -> None:
        for reply in (json.dumps({"categories": areas()}),
                      "Here you go:\n```json\n" + json.dumps({"categories": areas()}) + "\n```"):
            self.assertEqual(len(categories.parse(reply)), 12)

    def test_rejections(self) -> None:
        cases = {
            "no json": "I cannot do that.",
            "too few": json.dumps({"categories": areas(3)}),
            "too many": json.dumps({"categories": areas(41)}),
            "bad id": json.dumps({"categories": areas(11) + [{"id": "Chat API", "description": "x"}]}),
            "duplicate": json.dumps({"categories": areas(11) + [areas(1)[0]]}),
            "no description": json.dumps({"categories": areas(11) + [{"id": "zzz", "description": " "}]}),
            "no list": json.dumps({"areas": areas()}),
        }
        for name, reply in cases.items():
            with self.subTest(case=name), self.assertRaises(categories.CategoryError):
                categories.parse(reply)


class _Repo(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.base / "state"),
                                      "XO_PROJECTS_ROOT": str(self.base / "projects"),
                                      categories.ENV_MIN_FILES: "6"})
        env.start()
        self.addCleanup(env.stop)


class EligibilityTests(_Repo):
    def test_a_large_git_repo_qualifies(self) -> None:
        repo = git_repo(self.base / "projects" / "big", 8)
        ok, reason, files = asyncio.run(categories.eligible(repo))
        self.assertTrue(ok)
        self.assertEqual((reason, len(files)), ("8 code files", 8))
        self.assertNotIn("notes.txt", files)

    def test_small_or_not_git_does_not(self) -> None:
        small = git_repo(self.base / "projects" / "small", 3)
        self.assertEqual(asyncio.run(categories.eligible(small))[:2], (False, "3 code files, fewer than 6"))
        plain = self.base / "projects" / "plain"
        plain.mkdir(parents=True)
        self.assertEqual(asyncio.run(categories.eligible(plain))[:2], (False, "not a git repository"))

    def test_the_default_threshold(self) -> None:
        with patch.dict(os.environ, {categories.ENV_MIN_FILES: ""}):
            self.assertEqual(categories.min_files(), 150)

    def test_the_prompt_has_paths_and_readme_only(self) -> None:
        repo = git_repo(self.base / "projects" / "big", 8)
        files = asyncio.run(categories.code_files(repo))
        prompt = categories.draft_prompt(repo, files)
        self.assertIn("src/m0/f0.py", prompt)
        self.assertIn("src/m1/  (3)", prompt)
        self.assertIn("A demo repository.", prompt)
        self.assertNotIn("x = 1", prompt)  # never file contents


class EnsureTests(_Repo):
    def setUp(self) -> None:
        super().setUp()
        git_repo(self.base / "projects" / "big", 8)
        self.replies: list[str | None] = []
        self.prompts: list[str] = []

        class _OneShot:
            @staticmethod
            async def complete(prompt: str):
                self.prompts.append(prompt)
                return self.replies.pop(0)

        patcher = patch.object(categories, "try_load_capability", return_value=_OneShot)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ensure(self, project="big", **kw):
        return asyncio.run(categories.ensure(project, "sample_agent", **kw))

    def test_drafted_once_then_kept(self) -> None:
        self.replies = [json.dumps({"categories": areas(14)})]
        document, what = self.ensure()
        self.assertEqual(what, "drafted 14 areas from 8 code files")
        stored = json.loads(categories.path_for("big").read_text())
        self.assertEqual(list(stored)[:2], ["schema", "source"])
        self.assertEqual((stored["schema"], stored["code_files"], len(stored["categories"])), (1, 8, 14))
        self.assertTrue(stored["ts"].endswith("Z"))
        self.assertEqual(self.ensure(), (document, "kept the existing list"))
        self.assertEqual(len(self.prompts), 1)

    def test_force_redrafts(self) -> None:
        self.replies = [json.dumps({"categories": areas(14)}), json.dumps({"categories": areas(20)})]
        self.ensure()
        document, _ = self.ensure(force=True)
        self.assertEqual(len(document["categories"]), 20)

    def test_a_failed_or_bad_draft_is_reported_not_raised(self) -> None:
        self.replies = [None, "no json here"]
        self.assertEqual(self.ensure(), (None, "draft failed: the drafting run failed; see the server log"))
        self.assertEqual(self.ensure()[1], "draft failed: the reply holds no JSON object")
        self.assertFalse(categories.path_for("big").exists())

    def test_an_ineligible_project_is_never_drafted(self) -> None:
        git_repo(self.base / "projects" / "small", 2)
        self.assertEqual(self.ensure("small"), (None, "not eligible: 2 code files, fewer than 6"))
        self.assertEqual(self.prompts, [])

    def test_a_hand_edited_list_is_loaded_and_checked(self) -> None:
        self.replies = [json.dumps({"categories": areas(14)})]
        self.ensure()
        path = categories.path_for("big")
        document = json.loads(path.read_text())
        document["categories"] = areas(2)  # broken by hand
        path.write_text(json.dumps(document))
        with self.assertLogs(categories.log, "WARNING"):
            self.assertIsNone(categories.load("big"))


class OneShotTests(unittest.TestCase):
    def test_a_tool_free_run_with_no_transcript_and_the_prompt_on_stdin(self) -> None:
        seen = {}

        async def fake_run(argv, **kw):
            seen.update(argv=argv, **kw)
            return CommandResult(argv=argv, returncode=0, duration_seconds=1.0,
                                 output=json.dumps({"is_error": False, "result": "the reply"}))

        with patch.object(oneshot, "run", fake_run):
            self.assertEqual(asyncio.run(oneshot.complete("a long prompt")), "the reply")
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertIn("--no-session-persistence", argv)
        self.assertNotIn("a long prompt", argv)
        self.assertEqual(seen["input"], b"a long prompt")
        self.assertIn("xo-oneshot-", seen["cwd"])

    def test_failures_are_none(self) -> None:
        for result in (CommandResult(argv=[], returncode=1, output="", duration_seconds=0),
                       CommandResult(argv=[], returncode=0, output="not json", duration_seconds=0),
                       CommandResult(argv=[], returncode=0, output=json.dumps({"is_error": True, "result": "x"}),
                                     duration_seconds=0)):
            async def fake_run(argv, _result=result, **kw):
                return _result
            with self.subTest(output=result.output), patch.object(oneshot, "run", fake_run), \
                 self.assertLogs(oneshot.log, "WARNING"):
                self.assertIsNone(asyncio.run(oneshot.complete("p")))


if __name__ == "__main__":
    unittest.main()
