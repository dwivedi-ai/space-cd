"""Every new session's decision is one line in the machine-local decision log.

``XO_INTELLIGENCE_MODE=shadow`` classifies each new session with Levanto Sage
in the background and logs the decision, while the session runs exactly as it
would have: the reply never waits for Sage, and a failed or missing answer is
logged as the default. ``off`` sends and writes nothing, and a resumed turn is
never re-decided.

- a project's chats: ``~/.quirq/projects/<pid>/intelligence/decisions.jsonl``
- chats with no project: ``~/.quirq/sessions/intelligence/decisions.jsonl``
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.cowork_agent import chat
from services.cowork_agent.engine.chat_state import active_streams
from services.cowork_agent.intelligence import classify, decision_log, decisions, mode, profiles, selection
from services.levanto.client import SageResult

PID = "7deb4a22-0789-497d-9399-a2272579fa06"
CONFIG = profiles.parse({
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "high"],
    "profiles": [
        {"id": "light", "model": None, "effort": "low", "use_when": "questions"},
        {"id": "deep", "model": "claude-opus-5-5", "effort": "high", "use_when": "big work"},
    ],
}, sha256="cfg-sha")


def sage_picks(chosen: str | None) -> classify.Decision:
    return classify.Decision(profile=chosen if chosen not in (None, "unknown") else None,
                             reason=classify.SAGE_CHOICE if chosen else classify.SAGE_UNSURE,
                             sage={"choice": {"chosen": chosen}, "tags": None, "units": 2, "errors": []})


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.state = base / "state"
        self.projects = base / "projects"
        env = patch.dict(os.environ, {
            "QUIRQ_STATE_ROOT": str(self.state),
            "XO_PROJECTS_ROOT": str(self.projects),
            mode.ENV_MODE: "shadow",
        })
        env.start()
        self.addCleanup(env.stop)

    def make_project(self, name: str = "demo", pid: str | None = PID) -> None:
        xo = self.projects / name / ".xo"
        xo.mkdir(parents=True)
        if pid:
            (xo / "project.json").write_text(json.dumps({"schema": 1, "pid": pid, "name": name}), encoding="utf-8")

    def decide(self, project, request=None, chosen="deep") -> None:
        with patch.object(decisions.classify, "classify", AsyncMock(return_value=sage_picks(chosen))):
            asyncio.run(decisions._decide(CONFIG, "shadow", "sample_agent", "fix the retry test",
                                          "s1", project, request))


class StartTests(_Sandbox):
    def start(self) -> object:
        async def go():
            return decisions.start(agent_name="sample_agent", text="hi", session_id="s1",
                                   project=None, request=None)
        return asyncio.run(go())

    def test_off_decides_nothing(self) -> None:
        for value in ("off", "", "banana"):
            with self.subTest(mode=value), patch.dict(os.environ, {mode.ENV_MODE: value}), \
                 patch.object(profiles, "load", return_value=CONFIG):
                self.assertIsNone(self.start())

    def test_an_agent_without_profiles_decides_nothing(self) -> None:
        with patch.object(profiles, "load", return_value=None):
            self.assertIsNone(self.start())

    def test_shadow_starts_a_task(self) -> None:
        async def go():
            with patch.object(profiles, "load", return_value=CONFIG), \
                 patch.object(decisions, "_decide", AsyncMock(return_value=None)):
                task = decisions.start(agent_name="sample_agent", text="hi", session_id="s1",
                                       project=None, request=None)
                self.assertIn(task, decisions._tasks)
                await task
            return task
        task = asyncio.run(go())
        self.assertNotIn(task, decisions._tasks)


class LogTests(_Sandbox):
    def test_a_projects_decision_lands_in_its_runtime_home(self) -> None:
        self.make_project()
        self.decide("demo")
        [line] = read_lines(self.state / "projects" / PID / "intelligence" / "decisions.jsonl")
        self.assertEqual(list(line)[:3], ["ts", "type", "schema"])
        self.assertEqual((line["type"], line["schema"]), ("intelligence.decision", 1))
        self.assertTrue(line["ts"].endswith("Z"))
        self.assertEqual((line["pid"], line["project_id"], line["session_id"]), (PID, "demo", "s1"))
        self.assertEqual((line["runtime"], line["mode"]), ("sample_agent", "shadow"))
        self.assertEqual(line["request"]["preview"], "fix the retry test")
        self.assertEqual(len(line["request"]["sha256"]), 64)
        self.assertEqual(line["profiles"], {"sha256": "cfg-sha"})
        self.assertEqual(line["decision"], {"profile": "deep", "reason": "sage_choice",
                                            "model": "claude-opus-5-5", "effort": "high"})
        # Shadow: nothing was applied.
        self.assertEqual(line["applied"], {"profile": None, "model": None, "effort": None, "source": None})

    def test_no_decision_means_the_default(self) -> None:
        self.decide(None, chosen=None)
        [line] = read_lines(self.state / "sessions" / "intelligence" / "decisions.jsonl")
        self.assertEqual(line["decision"], {"profile": None, "reason": "sage_unsure", "model": None, "effort": None})
        self.assertNotIn("pid", line)

    def test_applied_records_what_the_request_chose(self) -> None:
        self.decide(None, request=selection.RequestChoice(profile="light"))
        [line] = read_lines(self.state / "sessions" / "intelligence" / "decisions.jsonl")
        self.assertEqual(line["applied"], {"profile": "light", "model": None, "effort": "low", "source": "request"})

    def test_a_project_without_a_folder_yet_uses_its_name_keyed_home(self) -> None:
        self.decide("brand-new")
        self.assertTrue((self.state / "projects" / "brand-new" / "intelligence" / "decisions.jsonl").is_file())

    def test_a_write_failure_is_logged_not_raised(self) -> None:
        with patch.object(decision_log, "append_jsonl", side_effect=OSError("disk full")), \
             self.assertLogs(decision_log.log, "ERROR"):
            self.decide(None)

    def test_a_long_profile_list_drops_per_option_probabilities(self) -> None:
        sage = {"choice": {"chosen": "p1", "options": {f"p{i}": 0.5 for i in range(800)}},
                "tags": None, "units": 1, "errors": []}
        line = decision_log.build_line(
            identity={}, session_id="s1", runtime="a", mode="shadow", request={}, profiles_sha256="x",
            sage=sage, decision={}, applied={}, latency_ms=1.0)
        self.assertNotIn("options", line["sage"]["choice"])
        self.assertLess(len(json.dumps(line)), decision_log.MAX_LINE_BYTES)


class ShadowRouteTests(_Sandbox):
    """End to end through ``POST /api/chat/prompt`` with a stubbed Sage."""

    def setUp(self) -> None:
        super().setUp()
        self.release = threading.Event()
        self.calls: list[str] = []

        async def slow_decide(content, question, *, timeout):
            self.calls.append(question["kind"])
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            if question["kind"] == "choice":
                return SageResult(ok=True, status=200, data={"result": {
                    "chosen": "deep", "probability": 0.8, "probabilities": []}})
            return SageResult(ok=True, status=200, data={"result": {"tags": []}})

        for patcher in (patch.object(classify.client, "decide", slow_decide),
                        patch.object(chat, "_resolve_user_id", AsyncMock(return_value=None))):
            patcher.start()
            self.addCleanup(patcher.stop)
        before = set(active_streams)
        self.addCleanup(lambda: [active_streams.pop(k, None) for k in set(active_streams) - before])
        app = FastAPI()
        app.include_router(chat.router)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.log = self.state / "sessions" / "intelligence" / "decisions.jsonl"

    def wait_for_log(self) -> list[dict]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.log.is_file():
                return read_lines(self.log)
            time.sleep(0.02)
        self.fail("no decision was logged")

    def test_a_new_session_is_decided_without_waiting(self) -> None:
        response = self.client.post("/api/chat/prompt", json={"text": "fix it", "agent_name": "claude_code"})
        # The prompt answered while Sage was still thinking.
        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.log.exists())
        self.release.set()
        [line] = self.wait_for_log()
        self.assertEqual(line["session_id"], response.json()["session_id"])
        self.assertEqual(line["decision"]["profile"], "deep")
        self.assertEqual(line["decision"]["effort"], "high")
        self.assertEqual(sorted(self.calls), ["choice", "tags"])

    def test_a_resumed_turn_is_not_decided_again(self) -> None:
        self.release.set()
        response = self.client.post("/api/chat/prompt", json={
            "text": "and the other test", "agent_name": "claude_code", "session_id": "existing"})
        self.assertEqual(response.status_code, 200)
        time.sleep(0.1)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.log.exists())

    def test_off_sends_nothing(self) -> None:
        self.release.set()
        with patch.dict(os.environ, {mode.ENV_MODE: "off"}):
            self.client.post("/api/chat/prompt", json={"text": "fix it", "agent_name": "claude_code"})
        time.sleep(0.1)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
