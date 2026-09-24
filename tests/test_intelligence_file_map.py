"""Each code file is tagged with its areas once, by Sage, from its path and outline (step 6b).

Incremental: unchanged files are not re-tagged, deleted ones are dropped, and
a new category list starts over. A 402 stops the run with what it has saved,
and the next run resumes. Test files are skipped, and file contents never
reach Sage beyond the outline.
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

from services.cowork_agent.intelligence import file_map
from services.levanto.client import SageResult

AREAS = {"ts": "2026-09-24T10:00:00Z", "categories": [
    {"id": "chat", "description": "the chat API"}, {"id": "storage", "description": "files on disk"}]}


def sage_tags(chat: float, storage: float) -> SageResult:
    return SageResult(ok=True, status=200, data={"result": {"tags": [
        {"id": "chat", "probability": chat, "applies": chat > 0.5},
        {"id": "storage", "probability": storage, "applies": None if 0.4 < storage < 0.6 else storage > 0.5},
        {"id": "made_up", "probability": 0.9, "applies": True},
    ]}})


class OutlineTests(unittest.TestCase):
    def test_python(self) -> None:
        text = '"""The chat route."""\nimport os\nSECRET = "hunter2"\n\nasync def chat_prompt():\n    pass\n\nclass Router:\n    pass\n'
        self.assertEqual(file_map.outline(text, "chat.py"), "The chat route.; chat_prompt; Router")

    def test_javascript(self) -> None:
        text = "// The toolbar\nconst key = 'abc';\nexport function render() {}\nclass Toolbar {}\n"
        self.assertEqual(file_map.outline(text, "ui/toolbar.js"), "The toolbar; render; Toolbar")

    def test_tests_are_not_indexed(self) -> None:
        for path in ("tests/test_x.py", "pkg/test_y.py", "a/b_test.go", "ui/x.test.js", "src/__tests__/a.ts"):
            self.assertFalse(file_map.is_indexed(path), path)
        for path in ("routers/chat.py", "ui/app.js", "README.md") [:2]:
            self.assertTrue(file_map.is_indexed(path), path)


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.repo = base / "repo"
        self.target = base / "state" / "file_map.json"
        for rel, text in {"chat.py": '"""Chat."""\ndef chat(): SECRET = 1\n', "store.py": "def save(): pass\n",
                          "tests/test_chat.py": "def test(): pass\n", "notes.md": "# notes\n"}.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text)
        self.git("init", "-q")
        self.git("add", ".")
        self.sent: list[str] = []
        self.answers: list[SageResult] = []
        env = patch.dict(os.environ, {"LEVANTO_API_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *args], check=True)

    def index(self, areas=AREAS, limit=None) -> file_map.IndexRun:
        async def fake_decide(content, question, *, timeout):
            self.sent.append(content)
            return self.answers.pop(0) if self.answers else sage_tags(0.9, 0.1)
        with patch.object(file_map.client, "decide", fake_decide):
            return asyncio.run(file_map.index(self.repo, areas, self.target, limit=limit))

    def saved(self) -> dict:
        return json.loads(self.target.read_text())

    def test_first_run_tags_code_files_only(self) -> None:
        done = self.index()
        self.assertEqual((done.tagged, done.units, done.unchanged, done.stopped), (2, 2, 0, None))
        files = self.saved()["files"]
        self.assertEqual(sorted(files), ["chat.py", "store.py"])
        self.assertEqual(files["chat.py"]["tags"], {"chat": [0.9, True], "storage": [0.1, False]})
        self.assertEqual(self.saved()["schema"], 1)
        self.assertEqual(sorted(self.sent), ["File: chat.py\nOutline: Chat.; chat", "File: store.py\nOutline: save"])

    def test_unsure_stays_unsure(self) -> None:
        self.answers = [sage_tags(0.9, 0.5), sage_tags(0.9, 0.5)]
        self.index()
        self.assertEqual(self.saved()["files"]["chat.py"]["tags"]["storage"], [0.5, None])

    def test_only_changed_files_are_retagged_and_deleted_ones_dropped(self) -> None:
        self.index()
        self.sent.clear()
        (self.repo / "chat.py").write_text("def chat_v2(): pass\n")
        self.git("add", "chat.py")
        self.git("rm", "-q", "-f", "store.py")
        done = self.index()
        self.assertEqual((done.tagged, done.unchanged, done.dropped), (1, 0, 1))
        self.assertEqual(self.sent, ["File: chat.py\nOutline: chat_v2"])
        self.assertEqual(sorted(self.saved()["files"]), ["chat.py"])

    def test_a_new_category_list_retags_everything(self) -> None:
        self.index()
        done = self.index(areas={**AREAS, "ts": "2026-09-25T10:00:00Z"})
        self.assertEqual(done.tagged, 2)

    def test_a_402_stops_the_run_and_the_next_one_resumes(self) -> None:
        with patch.object(file_map, "CONCURRENCY", 1):
            self.answers = [sage_tags(0.9, 0.1), SageResult(ok=False, status=402, detail="Insufficient balance")]
            first = self.index()
            self.assertEqual((first.tagged, first.stopped), (1, "balance"))
            self.assertEqual(len(self.saved()["files"]), 1)
            second = self.index()
        self.assertEqual((second.tagged, second.unchanged, second.stopped), (1, 1, None))

    def test_other_failures_are_counted_and_retried_next_run(self) -> None:
        self.answers = [SageResult(ok=False, status=503, detail="loading"), sage_tags(0.9, 0.1)]
        done = self.index()
        self.assertEqual((done.tagged, done.failed, done.stopped), (1, 1, None))
        self.assertEqual(self.index().tagged, 1)

    def test_limit_and_cap(self) -> None:
        done = self.index(limit=1)
        self.assertEqual((done.tagged, done.skipped_over_cap), (1, 1))
        with patch.dict(os.environ, {file_map.ENV_MAX_FILES: "1"}):
            self.assertEqual(self.index().tagged, 0)  # the one tagged file already fills the cap

    def test_not_a_git_repo(self) -> None:
        plain = self.repo.parent / "plain"
        plain.mkdir()
        with patch.object(file_map.client, "decide") as decide:
            done = asyncio.run(file_map.index(plain, AREAS, self.target))
        self.assertEqual(done.stopped, "not a git repository")
        decide.assert_not_called()


if __name__ == "__main__":
    unittest.main()
