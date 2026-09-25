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
import unittest
from pathlib import Path
from unittest.mock import patch

from routers.cowork_agent import chat
from services.cowork_agent.adapters.claude_code import adapter as adapter_mod
from services.cowork_agent.adapters.claude_code import context_hook
from services.cowork_agent.engine import dispatcher as dispatcher_mod
from services.cowork_agent.intelligence import context, decision_log, decisions, mode, outcomes, profiles

CONFIG = profiles.parse({
    "schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low"],
    "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "questions"}],
})
HOSTILE = "it's \"quoted\" $(touch /tmp/pwned) `id`; rm -rf ~ \n and a newline"


class SwitchTests(unittest.TestCase):
    def test_off_unless_set(self) -> None:
        for value, expected in (("", "off"), ("off", "off"), ("note", "note"), ("NOTE", "note")):
            with self.subTest(value=value), patch.dict(os.environ, {mode.ENV_CONTEXT: value}):
                self.assertEqual(mode.context_mode(), expected)

    def test_a_typo_adds_nothing(self) -> None:
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "notes"}), self.assertLogs(mode.log, "WARNING"):
            self.assertEqual(mode.context_mode(), "off")

    def test_turn_context(self) -> None:
        info = {"agent_name": "sample_agent"}
        with patch.object(profiles, "load", return_value=CONFIG):
            self.assertIsNone(context.turn_context(info))  # off by default (tests/__init__.py)
            with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}):
                self.assertEqual(context.turn_context(info), context.NOTE)
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}), patch.object(profiles, "load", return_value=None):
            self.assertIsNone(context.turn_context(info))

    def test_the_log_keeps_size_and_hash_not_text(self) -> None:
        self.assertIsNone(context.record(None))
        with patch.dict(os.environ, {mode.ENV_CONTEXT: "note"}):
            recorded = context.record("hello")
        self.assertEqual((recorded["kind"], recorded["chars"], len(recorded["sha256"])), ("note", 5, 64))
        self.assertNotIn("hello", json.dumps(recorded))


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
