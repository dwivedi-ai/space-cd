"""XO can hand Claude Code per-turn context through a ``UserPromptSubmit`` hook.

``XO_INTELLIGENCE_CONTEXT=note`` adds a fixed, harmless note to every turn of
an agent with profiles (plan step 5: prove the pipe). The claude_code adapter
writes a private settings layer whose hook ``cat``s a JSON file holding the
text, and passes it with ``--settings``. The text never reaches a command line,
and the files are gone when the turn ends. The turn line records the
context's kind, size and hash, never the text.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from routers.cowork_agent import chat
from services.cowork_agent.adapters.claude_code import adapter as adapter_mod
from services.cowork_agent.adapters.claude_code import context_hook
from services.cowork_agent.engine import dispatcher as dispatcher_mod
from services.cowork_agent.intelligence import context, decision_log, decisions, mode, outcomes, profiles, where_to_look

CONFIG = profiles.parse({
    "schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low"],
    "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "questions"}],
})
HOSTILE = "it's \"quoted\" $(touch /tmp/pwned) `id`; rm -rf ~ \n and a newline"


class SwitchTests(unittest.TestCase):
    def test_off_unless_set(self) -> None:
        for value, expected in (("", "off"), ("off", "off"), ("note", "note"), ("NOTE", "note"), ("map", "map")):
            with self.subTest(value=value), patch.dict(os.environ, {mode.ENV_CONTEXT: value}):
                self.assertEqual(mode.context_mode(), expected)

    def test_a_typo_adds_nothing(self) -> None:
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "notes"}), self.assertLogs(mode.log, "WARNING"):
            self.assertEqual(mode.context_mode(), "off")

    def test_turn_context(self) -> None:
        info = {"agent_name": "sample_agent"}
        with patch.object(profiles, "load", return_value=CONFIG):
            self.assertIsNone(asyncio.run(context.turn_context(info)).text)  # off by default (tests/__init__.py)
            with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}):
                self.assertEqual(asyncio.run(context.turn_context(info)).text, context.NOTE)
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}), patch.object(profiles, "load", return_value=None):
            self.assertIsNone(asyncio.run(context.turn_context(info)).text)

    def test_the_log_keeps_size_and_hash_not_text(self) -> None:
        self.assertIsNone(context.record(None))
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}):
            recorded = context.record("hello")
        self.assertEqual((recorded["kind"], recorded["chars"], len(recorded["sha256"])), ("note", 5, 64))
        self.assertNotIn("hello", json.dumps(recorded))


class _Pending:
    """A new session's decision as the first turn sees it (decisions.PendingDecision)."""

    def __init__(self, areas, started_ago: float = 0.0) -> None:
        self.areas = areas
        self.started = time.monotonic() - started_ago


NEVER = object()  # the areas answer never arrives


class MapTests(unittest.TestCase):
    """XO_INTELLIGENCE_CONTEXT=map: where to look on a new session's first turn."""

    def turn(self, areas=None, *, new=True, handed=None, started_ago=0.0, decided=True):
        """Run the first turn's context; ``areas`` is what the decision task resolved."""
        handed = handed or where_to_look.Handover(
            text="XO Space, from this project's map: `a.py`.",
            record={"kind": "map", "areas": ["chat"], "files": ["a.py"], "also_edited": []})

        async def go():
            future = asyncio.get_running_loop().create_future()
            if areas is not NEVER:
                future.set_result(areas)
            info = {"agent_name": "sample_agent", "agent_id": "demo", "question": "fix the stream retry",
                    "is_new_session": new,
                    "intelligence_decision": _Pending(future, started_ago) if decided else None}
            return await context.turn_context(info)

        with patch.object(profiles, "load", return_value=CONFIG), \
             patch.dict(os.environ, {mode.ENV_CONTEXT: "map"}), \
             patch.object(where_to_look, "build", return_value=handed) as build:
            out = asyncio.run(go())
        self.build = build
        return out

    def test_a_confident_hand_over_is_sent_and_logged(self) -> None:
        out = self.turn({"areas": {"chat": [0.95, True]}, "units": 1})
        self.assertEqual(out.text, "XO Space, from this project's map: `a.py`.")
        self.assertEqual((out.record["kind"], out.record["files"], out.record["areas"]), ("map", ["a.py"], ["chat"]))
        self.assertEqual(out.record["chars"], len(out.text))
        self.assertNotIn("XO Space", json.dumps(out.record))
        self.build.assert_called_once_with("demo", "fix the stream retry", {"chat": [0.95, True]})

    def test_a_withheld_hand_over_sends_nothing_and_says_why(self) -> None:
        withheld = where_to_look.Handover(text=None, record={"kind": "map", "withheld": "no confident area"})
        out = self.turn({"areas": {}, "units": 1}, handed=withheld)
        self.assertEqual((out.text, out.record), (None, {"kind": "map", "withheld": "no confident area"}))

    def test_late_areas_mean_nothing_is_sent(self) -> None:
        # Deciding started almost the whole deadline ago: the first turn waits the rest, then gives up.
        started = time.monotonic()
        out = self.turn(NEVER, started_ago=decisions.ON_WAIT_S - 0.05)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual((out.text, out.record), (None, {"kind": "map", "withheld": "late"}))

    def test_a_project_that_is_not_indexed_adds_and_logs_nothing(self) -> None:
        out = self.turn(None)
        self.assertEqual((out.text, out.record), (None, None))

    def test_a_failed_areas_call_is_logged(self) -> None:
        out = self.turn({"areas": None, "units": 0, "error": "balance"})
        self.assertEqual(out.record, {"kind": "map", "withheld": "no areas (balance)"})

    def test_later_turns_and_sessions_without_a_decision_add_nothing(self) -> None:
        self.assertEqual(self.turn({"areas": {"chat": [0.95, True]}}, new=False).text, None)
        self.assertEqual(self.turn(decided=False).text, None)
        self.build.assert_not_called()


class HookFileTests(unittest.TestCase):
    def test_no_context_no_files(self) -> None:
        self.assertIsNone(context_hook.write_turn_context(None))
        self.assertIsNone(context_hook.write_turn_context(""))

    def test_the_hook_prints_the_context_and_nothing_reaches_a_shell(self) -> None:
        settings = context_hook.write_turn_context(HOSTILE)
        self.addCleanup(context_hook.cleanup_turn_context, settings)
        document = json.loads(settings.read_text())
        [[hook]] = [entry["hooks"] for entry in document["hooks"]["UserPromptSubmit"]]
        self.assertEqual(hook["type"], "command")
        self.assertNotIn("pwned", hook["command"])
        argv = shlex.split(hook["command"])
        self.assertEqual(argv[0], "cat")
        output = json.loads(subprocess.run(argv, capture_output=True, check=True, text=True).stdout)
        self.assertEqual(output, {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit", "additionalContext": HOSTILE}})
        self.assertFalse(Path("/tmp/pwned").exists())

    def test_files_are_private_and_removed(self) -> None:
        settings = context_hook.write_turn_context("note")
        folder = settings.parent
        self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)
        for name in ("settings.json", "context.json"):
            self.assertEqual(stat.S_IMODE((folder / name).stat().st_mode), 0o600)
        context_hook.cleanup_turn_context(settings)
        self.assertFalse(folder.exists())

    def test_cleanup_only_removes_its_own_folders(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "settings.json"
            target.write_text("{}")
            context_hook.cleanup_turn_context(target)
            self.assertTrue(target.exists())


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(base / "s"), "XO_PROJECTS_ROOT": str(base / "p")})
        env.start()
        self.addCleanup(env.stop)
        self.seen: dict = {}

        async def no_lines():
            return
            yield

        class _Process:
            stdout = no_lines()
            returncode = 0

            async def wait(self):
                return 0

        async def spawn(*cmd, **_kw):
            self.seen["argv"] = list(cmd)
            if "--settings" in cmd:
                path = Path(cmd[cmd.index("--settings") + 1])
                self.seen["settings"] = path
                self.seen["existed"] = path.is_file()
            return _Process()

        patcher = patch.object(adapter_mod.asyncio, "create_subprocess_exec", spawn)
        patcher.start()
        self.addCleanup(patcher.stop)

    def stream(self, **kwargs) -> None:
        async def consume():
            async for _ in adapter_mod.Adapter({}).stream("hi", None, our_session_id="s1",
                                                          is_new_session=True, **kwargs):
                pass
        asyncio.run(consume())

    def test_the_turn_gets_its_settings_and_they_are_removed(self) -> None:
        self.stream(context="note for this turn")
        argv = self.seen["argv"]
        self.assertTrue(self.seen["existed"])
        self.assertLess(argv.index("--settings"), argv.index("-p"))
        self.assertFalse(self.seen["settings"].parent.exists())

    def test_no_context_no_settings_flag(self) -> None:
        self.stream()
        self.assertNotIn("--settings", self.seen["argv"])


class _Dispatcher:
    calls: list = []

    def __init__(self, agent_name: str) -> None:
        pass

    async def stream(self, question, session_id=None, **kwargs):
        _Dispatcher.calls.append(kwargs)
        yield {"done": True, "native_session_id": "n1", "outcome": None}


class RouteAndLogTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / "state"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.state), mode.ENV_CONTEXT: "note",
                                      mode.ENV_MODE: "shadow"})
        env.start()
        self.addCleanup(env.stop)
        for patcher in (patch.object(profiles, "load", return_value=CONFIG),
                        patch.object(dispatcher_mod, "AgentDispatcher", _Dispatcher)):
            patcher.start()
            self.addCleanup(patcher.stop)
        _Dispatcher.calls = []
        for cache in (decisions._session_setups, decisions._session_projects):
            cache.clear()
            self.addCleanup(cache.clear)

    def run_turn(self) -> None:
        info = {"agent_name": "sample_agent", "question": "hi", "our_session_id": "s1",
                "is_new_session": True, "intelligence_request": None}

        async def consume():
            chunks = [c async for c in chat._dispatcher_sse(info)]
            await asyncio.gather(*outcomes._tasks)
            return chunks
        asyncio.run(consume())

    def test_the_note_reaches_the_adapter_and_the_log_records_it(self) -> None:
        self.run_turn()
        [kwargs] = _Dispatcher.calls
        self.assertEqual(kwargs["context"], context.NOTE)
        [line] = [json.loads(x) for x in (self.state / "sessions" / "intelligence" / "decisions.jsonl").read_text().splitlines()]
        self.assertEqual(line["type"], "intelligence.turn")
        self.assertEqual((line["context"]["kind"], line["context"]["chars"]), ("note", len(context.NOTE)))
        self.assertNotIn("context channel", json.dumps(line))

    def test_off_hands_over_nothing(self) -> None:
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "off"}):
            self.run_turn()
        self.assertNotIn("context", _Dispatcher.calls[0])


if __name__ == "__main__":
    unittest.main()
