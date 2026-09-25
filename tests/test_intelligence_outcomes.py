"""Every finished turn adds one ``intelligence.turn`` line to the decision log.

The adapter reads what the turn cost and did from Claude Code's ``result``
event and hands it on with its ``done`` event (``outcome``); the stream
records it, with the setup the turn ran with, beside the session's decision
line. A session's outcome is the sum of its turn lines, joined on
``session_id``. Nothing is written in ``off``.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from routers.cowork_agent import chat
from services.cowork_agent.adapters.claude_code import adapter as adapter_mod
from services.cowork_agent.adapters.claude_code.streaming import parse_stream_line, turn_outcome
from services.cowork_agent.engine import dispatcher as dispatcher_mod
from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import decision_log, decisions, mode, outcomes, profiles, selection

# Field names as Claude Code 2.1.281 prints them (`claude --print --output-format json`).
RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "duration_ms": 3315,
    "duration_api_ms": 3213, "num_turns": 1, "result": "ok", "session_id": "n1",
    "total_cost_usd": 0.0240249, "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 1200,
              "cache_creation_input_tokens": 300, "service_tier": "standard"},
    "modelUsage": {"claude-haiku-4-5-20251001": {}},
}
OUTCOME = {
    "turns": 1, "duration_ms": 3315, "api_duration_ms": 3213, "cost_usd": 0.0240249,
    "is_error": False, "stop": "success", "models": ["claude-haiku-4-5-20251001"],
    "tokens": {"input": 10, "output": 5, "cache_read": 1200, "cache_write": 300},
}
CONFIG = profiles.parse({
    "schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low", "high"],
    "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "questions"}],
})


def _lines(*events: dict):
    async def gen():
        for event in events:
            yield (json.dumps(event) + "\n").encode()
    return gen()


class _Process:
    def __init__(self, *events: dict) -> None:
        self.stdout = _lines(*events)
        self.returncode = 0

    async def wait(self) -> int:
        return 0


class ParseTests(unittest.TestCase):
    def test_the_outcome_of_a_turn(self) -> None:
        self.assertEqual(turn_outcome(RESULT), OUTCOME)
        self.assertEqual(parse_stream_line(json.dumps(RESULT).encode())["outcome"], OUTCOME)

    def test_a_failed_turn_keeps_its_outcome(self) -> None:
        failed = {**RESULT, "subtype": "error_max_turns", "is_error": True, "result": "max turns"}
        event = parse_stream_line(json.dumps(failed).encode())
        self.assertEqual(event["type"], "error")
        self.assertEqual((event["outcome"]["stop"], event["outcome"]["is_error"]), ("error_max_turns", True))

    def test_missing_fields_are_none(self) -> None:
        outcome = turn_outcome({"type": "result", "num_turns": True, "usage": "?"})
        self.assertIsNone(outcome["turns"])
        self.assertEqual(outcome["tokens"], {"input": None, "output": None, "cache_read": None, "cache_write": None})


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(base / "s"), "XO_PROJECTS_ROOT": str(base / "p")})
        env.start()
        self.addCleanup(env.stop)

    def stream(self, *events: dict) -> list[dict]:
        async def spawn(*_cmd, **_kw):
            return _Process(*events)

        async def consume():
            return [e async for e in adapter_mod.Adapter({}).stream(
                "hi", None, our_session_id="s1", is_new_session=True)]

        with patch.object(adapter_mod.asyncio, "create_subprocess_exec", spawn):
            return asyncio.run(consume())

    def test_the_done_event_carries_the_outcome(self) -> None:
        events = self.stream({"type": "system", "subtype": "init", "session_id": "n1"}, RESULT)
        self.assertEqual(events[-1], {"done": True, "native_session_id": "n1", "outcome": OUTCOME})

    def test_a_failed_turn_reports_its_outcome_too(self) -> None:
        failed = {**RESULT, "is_error": True, "subtype": "error_during_execution", "result": "boom"}
        events = self.stream(failed)
        [error] = [e for e in events if e.get("type") == "error"]
        self.assertNotIn("outcome", error)  # the SSE error event stays as it was
        self.assertEqual(events[-1]["outcome"]["stop"], "error_during_execution")

    def test_no_result_means_no_outcome(self) -> None:
        self.assertIsNone(self.stream({"type": "system", "subtype": "init", "session_id": "n1"})[-1]["outcome"])


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.state = base / "state"
        env = patch.dict(os.environ, {
            "QUIRQ_STATE_ROOT": str(self.state), "XO_PROJECTS_ROOT": str(base / "projects"),
            mode.ENV_MODE: "shadow",
        })
        env.start()
        self.addCleanup(env.stop)
        patcher = patch.object(profiles, "load", return_value=CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)
        for cache in (decisions._session_projects, decisions._session_setups):
            cache.clear()
            self.addCleanup(cache.clear)
        self.root_log = self.state / "sessions" / "intelligence" / "decisions.jsonl"

    def after_turn(self, info: dict, applied=None, done=None, agent_error=False):
        async def go():
            task = outcomes.after_turn(info, applied, done, agent_error=agent_error)
            if task is not None:
                await task
            return task
        return asyncio.run(go())

    def lines(self, path: Path) -> list[dict]:
        return [json.loads(x) for x in path.read_text().splitlines()]


class TurnLineTests(_Sandbox):
    INFO = {"agent_name": "sample_agent", "our_session_id": "s1", "is_new_session": True, "agent_id": None}

    def test_a_turn_line(self) -> None:
        applied = {"profile": "light", "model": None, "effort": "low", "source": "request"}
        self.after_turn(self.INFO, applied, {"done": True, "outcome": OUTCOME})
        [line] = self.lines(self.root_log)
        self.assertEqual(list(line)[:3], ["ts", "type", "schema"])
        self.assertEqual((line["type"], line["schema"]), ("intelligence.turn", 1))
        self.assertTrue(line["ts"].endswith("Z"))
        self.assertEqual({k: line[k] for k in ("session_id", "runtime", "mode", "new_session", "agent_error")},
                         {"session_id": "s1", "runtime": "sample_agent", "mode": "shadow",
                          "new_session": True, "agent_error": False})
        self.assertEqual((line["applied"], line["outcome"]), (applied, OUTCOME))

    def test_a_turn_without_an_outcome(self) -> None:
        self.after_turn(self.INFO, None, None, agent_error=True)
        [line] = self.lines(self.root_log)
        self.assertEqual(line["applied"], {"profile": None, "model": None, "effort": None, "source": None})
        self.assertIsNone(line["outcome"])
        self.assertTrue(line["agent_error"])

    def test_off_writes_nothing(self) -> None:
        with patch.dict(os.environ, {mode.ENV_MODE: "off"}):
            self.assertIsNone(self.after_turn(self.INFO, None, {"done": True, "outcome": OUTCOME}))
        self.assertFalse(self.root_log.exists())

    def test_an_agent_without_profiles_writes_nothing(self) -> None:
        with patch.object(profiles, "load", return_value=None):
            self.assertIsNone(self.after_turn(self.INFO, None, {"done": True, "outcome": OUTCOME}))

    def test_a_resumed_turn_finds_its_sessions_project(self) -> None:
        project = self.state.parent / "projects" / "demo"
        (project / ".xo").mkdir(parents=True)
        sessions_io.write_session_row("demo", "sample:demo:web:abcd1234", {
            "sessionId": "s2", "nativeSessionId": "n2", "directory": str(project),
            "backend": "sample_agent", "updatedAt": 1})
        info = {"agent_name": "sample_agent", "our_session_id": "s2", "is_new_session": False, "agent_id": None}
        with patch.object(decisions, "_session_project", wraps=decisions._session_project) as lookup:
            self.after_turn(info, None, {"done": True, "outcome": OUTCOME})
            self.after_turn(info, None, {"done": True, "outcome": OUTCOME})
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(len(self.lines(self.state / "projects" / "demo" / "intelligence" / "decisions.jsonl")), 2)
        self.assertFalse(self.root_log.exists())

    def test_decision_lookups_skip_turn_lines(self) -> None:
        self.after_turn(self.INFO, None, {"done": True, "outcome": OUTCOME})
        self.assertIsNone(decision_log.find(None, "s1"))


class _Dispatcher:
    def __init__(self, agent_name: str) -> None:
        pass

    async def stream(self, question, session_id=None, **kwargs):
        yield {"type": "token", "token": "ok"}
        yield {"done": True, "native_session_id": "n1", "outcome": OUTCOME}


class StreamTests(_Sandbox):
    def test_the_stream_records_the_turn_after_the_reply(self) -> None:
        calls = []
        info = {"agent_name": "sample_agent", "question": "hi", "our_session_id": "s1",
                "is_new_session": True, "intelligence_request": selection.RequestChoice(profile="light")}

        async def consume():
            return [chunk async for chunk in chat._dispatcher_sse(info)]

        with patch.object(dispatcher_mod, "AgentDispatcher", _Dispatcher), \
             patch.object(outcomes, "after_turn", side_effect=lambda *a, **k: calls.append((a, k))):
            chunks = asyncio.run(consume())
        self.assertIn("event: done", chunks[-1])
        [(args, kwargs)] = calls
        self.assertIs(args[0], info)
        self.assertEqual(args[1], {"profile": "light", "model": None, "effort": "low", "source": "request"})
        self.assertEqual(args[2]["outcome"], OUTCOME)
        self.assertEqual(kwargs, {"agent_error": False, "context": None})


if __name__ == "__main__":
    unittest.main()
