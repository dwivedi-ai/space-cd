"""``POST /api/chat/prompt`` takes an optional profile / effort / model.

The route checks them against the agent's profiles (400 on an unknown
value), keeps them with the stream, and the stream hands the resolved setup to
the adapter as the ``intelligence`` keyword. A body without the fields, or an
agent without profiles, streams exactly as before.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.cowork_agent import chat
from services.cowork_agent.engine import dispatcher as dispatcher_mod
from services.cowork_agent.engine.chat_state import active_streams
from services.cowork_agent.intelligence import decisions, profiles, selection


class _Route(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(chat.router)
        self.client = TestClient(app)
        patcher = patch.object(chat, "_resolve_user_id", AsyncMock(return_value=None))
        patcher.start()
        self.addCleanup(patcher.stop)
        before = set(active_streams)
        self.addCleanup(lambda: [active_streams.pop(k, None) for k in set(active_streams) - before])

    def prompt(self, **body):
        return self.client.post("/api/chat/prompt", json={"text": "fix the retry test", **body})


class PromptTests(_Route):
    def test_a_profile_is_kept_with_the_stream(self) -> None:
        response = self.prompt(agent_name="claude_code", profile="deep", effort="xhigh")
        self.assertEqual(response.status_code, 200)
        info = active_streams[response.json()["stream_id"]]
        self.assertEqual(info["intelligence_request"],
                         selection.RequestChoice(profile="deep", effort="xhigh"))

    def test_unknown_values_are_a_400_naming_the_choices(self) -> None:
        response = self.prompt(agent_name="claude_code", profile="turbo")
        self.assertEqual(response.status_code, 400)
        self.assertIn("light, standard, deep, research", response.json()["detail"])
        self.assertEqual(self.prompt(agent_name="claude_code", effort="huge").status_code, 400)

    def test_a_body_without_the_fields_is_unchanged(self) -> None:
        response = self.prompt(agent_name="claude_code", model="claude_code/main")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(active_streams[response.json()["stream_id"]]["intelligence_request"].empty)

    def test_an_agent_without_profiles_ignores_the_fields(self) -> None:
        with patch.object(profiles, "load", return_value=None):
            response = self.prompt(agent_name="claude_code", profile="turbo")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(active_streams[response.json()["stream_id"]]["intelligence_request"])


class _FakeDispatcher:
    calls: list[dict] = []

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    async def stream(self, question, session_id=None, **kwargs):
        _FakeDispatcher.calls.append(kwargs)
        yield {"done": True, "native_session_id": "n1"}


class StreamTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(Path(tmp.name) / "state")})
        env.start()
        self.addCleanup(env.stop)
        _FakeDispatcher.calls = []
        decisions._session_setups.clear()
        self.addCleanup(decisions._session_setups.clear)
        patcher = patch.object(dispatcher_mod, "AgentDispatcher", _FakeDispatcher)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_stream(self, request) -> dict:
        info = {"agent_name": "claude_code", "question": "hi", "our_session_id": "s1",
                "is_new_session": True, "intelligence_request": request}

        async def consume():
            return [chunk async for chunk in chat._dispatcher_sse(info)]

        chunks = asyncio.run(consume())
        self.assertIn("event: done", chunks[-1])
        [kwargs] = _FakeDispatcher.calls
        return kwargs

    def test_the_chosen_setup_reaches_the_adapter(self) -> None:
        kwargs = self.run_stream(selection.RequestChoice(profile="deep"))
        self.assertEqual(kwargs["intelligence"],
                         {"profile": "deep", "model": "claude-opus-5-5", "effort": "high", "source": "request"})

    def test_nothing_chosen_passes_no_keyword(self) -> None:
        self.assertNotIn("intelligence", self.run_stream(selection.RequestChoice()))

    def test_a_stream_without_a_request_passes_no_keyword(self) -> None:
        self.assertNotIn("intelligence", self.run_stream(None))


if __name__ == "__main__":
    unittest.main()
