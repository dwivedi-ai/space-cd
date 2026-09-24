"""``XO_INTELLIGENCE_MODE=on`` applies the decision, and a session never switches setup.

A new session's first turn waits for Sage for at most ``ON_WAIT_S``: a pick in
time is applied, anything else (late, ``unknown``, ``null``, a failure) gets
the default setup. An explicit field in the request wins over both. Every
resumed turn re-applies the setup the session started with (from memory, or
from the decision log after a restart), so its model never changes
mid-session. A session that was never decided keeps running with no flags.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import classify, decisions, mode, outcomes, profiles, selection

CONFIG = profiles.parse({
    "schema": 1,
    "default": {"model": None, "effort": "medium"},
    "efforts": ["low", "medium", "high"],
    "profiles": [
        {"id": "light", "model": None, "effort": "low", "use_when": "questions"},
        {"id": "deep", "model": "claude-opus-5-5", "effort": "high", "use_when": "big work"},
    ],
}, sha256="cfg-sha")
NO_FLAG_DEFAULT = profiles.parse({
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "high"],
    "profiles": [{"id": "deep", "model": None, "effort": "high", "use_when": "big work"}],
})


def answer(profile: str | None, reason: str = classify.SAGE_CHOICE) -> classify.Decision:
    return classify.Decision(profile=profile, reason=reason,
                             sage={"choice": {"chosen": profile}, "tags": None, "units": 2, "errors": []})


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.state = base / "state"
        env = patch.dict(os.environ, {
            "QUIRQ_STATE_ROOT": str(self.state),
            "XO_PROJECTS_ROOT": str(base / "projects"),
            mode.ENV_MODE: "on",
        })
        env.start()
        self.addCleanup(env.stop)
        decisions._session_setups.clear()
        self.addCleanup(decisions._session_setups.clear)
        for patcher in (patch.object(decisions, "ON_WAIT_S", 0.2),
                        patch.object(profiles, "load", return_value=CONFIG)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def sage(self, decision: classify.Decision, delay: float = 0.0, tags_delay: float = 0.0):
        """Sage answers the choice after ``delay`` and the tags ``tags_delay`` later."""
        async def fake(config, content, *, timeout, choice_ready=None):
            await asyncio.sleep(delay)
            if choice_ready is not None and not choice_ready.done():
                choice_ready.set_result(decision.profile)
            await asyncio.sleep(tags_delay)
            return decision
        return patch.object(decisions.classify, "classify", fake)

    def first_turn(self, decision, *, delay=0.0, tags_delay=0.0, request=None,
                   session_id="s1") -> tuple[dict | None, dict]:
        """Run a new session's first turn; return its adapter keyword and its log line."""
        async def go():
            pending = decisions.start(agent_name="sample_agent", text="fix it", session_id=session_id,
                                      project=None, request=request)
            kwargs = await decisions.turn_selection({
                "agent_name": "sample_agent", "our_session_id": session_id, "is_new_session": True,
                "intelligence_request": request, "intelligence_decision": pending,
            })
            await pending.task
            return kwargs
        with self.sage(decision, delay, tags_delay):
            kwargs = asyncio.run(go())
        lines = (self.state / "sessions" / "intelligence" / "decisions.jsonl").read_text().splitlines()
        return kwargs, json.loads(lines[-1])

    @staticmethod
    def same_setup(first: dict) -> dict:
        return {**first, "source": "session"}

    def resumed_turn(self, session_id="s1", request=None) -> dict | None:
        return asyncio.run(decisions.turn_selection({
            "agent_name": "sample_agent", "our_session_id": session_id, "is_new_session": False,
            "intelligence_request": request,
        }))


class SelectTests(unittest.TestCase):
    def test_precedence(self) -> None:
        req = selection.RequestChoice
        session = {"profile": "light", "model": None, "effort": "low"}
        cases = [
            (dict(decided="deep"), selection.Selection("deep", "claude-opus-5-5", "high", "sage")),
            (dict(decided=None, use_default=True), selection.Selection(None, None, "medium", "default")),
            (dict(decided="deep", session=session), selection.Selection("light", None, "low", "session")),
            (dict(request=req(profile="light"), decided="deep"), selection.Selection("light", None, "low", "request")),
            (dict(request=req(effort="low"), decided="deep"), selection.Selection("deep", "claude-opus-5-5", "low", "sage")),
            (dict(decided="turbo", use_default=True), selection.Selection(None, None, "medium", "default")),
        ]
        for kwargs, expected in cases:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                request = kwargs.pop("request", None)
                self.assertEqual(selection.select(CONFIG, request, **kwargs), expected)


class FirstTurnTests(_Sandbox):
    def test_a_pick_in_time_is_applied(self) -> None:
        kwargs, line = self.first_turn(answer("deep"))
        self.assertEqual(kwargs, {"profile": "deep", "model": "claude-opus-5-5", "effort": "high", "source": "sage"})
        self.assertEqual(line["applied"], kwargs)
        self.assertEqual((line["mode"], line["decision"]["reason"]), ("on", "sage_choice"))

    def test_a_late_pick_is_logged_but_the_default_runs(self) -> None:
        kwargs, line = self.first_turn(answer("deep"), delay=0.5)
        self.assertEqual(kwargs, {"profile": None, "model": None, "effort": "medium", "source": "default"})
        self.assertEqual(line["decision"], {"profile": "deep", "reason": "sage_late",
                                            "model": "claude-opus-5-5", "effort": "high"})
        self.assertEqual(line["applied"], kwargs)

    def test_a_quick_choice_is_applied_while_slow_tags_finish(self) -> None:
        # Live 16:09: the choice came in 0.64 s, the tags in 3.7 s. The pick counts.
        kwargs, line = self.first_turn(answer("deep"), tags_delay=0.5)
        self.assertEqual(kwargs["source"], "sage")
        self.assertEqual(line["decision"]["reason"], "sage_choice")

    def test_no_decision_means_the_default(self) -> None:
        for decision in (answer(None, classify.SAGE_UNSURE), answer(None, classify.SAGE_UNKNOWN),
                         answer(None, classify.SAGE_ERROR)):
            with self.subTest(reason=decision.reason):
                kwargs, line = self.first_turn(decision)
                self.assertEqual(kwargs["source"], "default")
                self.assertEqual(line["decision"]["reason"], decision.reason)

    def test_the_request_wins(self) -> None:
        kwargs, _ = self.first_turn(answer("deep"), request=selection.RequestChoice(effort="low"))
        self.assertEqual(kwargs, {"profile": "deep", "model": "claude-opus-5-5", "effort": "low", "source": "sage"})

    def test_a_failed_decision_still_answers_the_turn(self) -> None:
        async def broken(config, content, *, timeout, choice_ready=None):
            raise RuntimeError("boom")

        async def go():
            pending = decisions.start(agent_name="sample_agent", text="fix it", session_id="s1",
                                      project=None, request=None)
            kwargs = await decisions.turn_selection({
                "agent_name": "sample_agent", "our_session_id": "s1", "is_new_session": True,
                "intelligence_request": None, "intelligence_decision": pending})
            await pending.task
            return kwargs

        with patch.object(decisions.classify, "classify", broken), self.assertLogs(decisions.log, "ERROR"):
            kwargs = asyncio.run(go())
        self.assertEqual(kwargs["source"], "default")

    def test_a_default_with_no_flags_passes_nothing(self) -> None:
        with patch.object(profiles, "load", return_value=NO_FLAG_DEFAULT):
            kwargs, line = self.first_turn(answer(None, classify.SAGE_UNSURE))
            self.assertIsNone(kwargs)
            self.assertEqual(line["applied"]["source"], "default")
            # ...and the turn line says so too, though the adapter got nothing.
            async def record():
                await outcomes.after_turn({"agent_name": "sample_agent", "our_session_id": "s1",
                                           "is_new_session": True, "agent_id": None},
                                          kwargs, {"done": True, "outcome": None}, agent_error=False)
            asyncio.run(record())
        turn = json.loads((self.state / "sessions" / "intelligence" / "decisions.jsonl").read_text().splitlines()[-1])
        self.assertEqual((turn["type"], turn["applied"]["source"]), ("intelligence.turn", "default"))


class ResumedTurnTests(_Sandbox):
    def test_a_resumed_turn_keeps_the_first_setup(self) -> None:
        first, _ = self.first_turn(answer("deep"))
        self.assertEqual(self.resumed_turn(), self.same_setup(first))

    def test_after_a_restart_the_setup_comes_from_the_log(self) -> None:
        first, _ = self.first_turn(answer("deep"))
        sessions_io.write_session_row("", "sample::web:abcd1234", {
            "sessionId": "s1", "nativeSessionId": "n1", "directory": "/x", "backend": "sample_agent", "updatedAt": 1})
        decisions._session_setups.clear()
        self.assertEqual(self.resumed_turn(), self.same_setup(first))

    def test_an_explicit_field_still_wins_on_a_resumed_turn(self) -> None:
        self.first_turn(answer("deep"))
        self.assertEqual(self.resumed_turn(request=selection.RequestChoice(effort="low"))["effort"], "low")

    def test_a_session_that_was_never_decided_runs_without_flags(self) -> None:
        self.assertIsNone(self.resumed_turn(session_id="older-session"))

    def test_shadow_keeps_an_explicit_first_setup(self) -> None:
        with patch.dict(os.environ, {mode.ENV_MODE: "shadow"}):
            first, line = self.first_turn(answer("deep"), request=selection.RequestChoice(profile="light"))
            self.assertEqual(first, {"profile": "light", "model": None, "effort": "low", "source": "request"})
            self.assertEqual(self.resumed_turn(), self.same_setup(first))
            # After a restart it comes back from the decision log.
            sessions_io.write_session_row("", "sample::web:abcd1234", {
                "sessionId": "s1", "nativeSessionId": "n1", "directory": "/x", "backend": "sample_agent", "updatedAt": 1})
            decisions._session_setups.clear()
            self.assertEqual(self.resumed_turn(), self.same_setup(first))

    def test_off_keeps_an_explicit_first_setup_until_a_restart(self) -> None:
        with patch.dict(os.environ, {mode.ENV_MODE: "off"}):
            first = asyncio.run(decisions.turn_selection({
                "agent_name": "sample_agent", "our_session_id": "s9", "is_new_session": True,
                "intelligence_request": selection.RequestChoice(effort="high"),
                "intelligence_decision": None}))
            self.assertEqual(first["effort"], "high")
            self.assertEqual(self.resumed_turn("s9")["effort"], "high")
            decisions._session_setups.clear()  # off writes no log to recover it from
            self.assertIsNone(self.resumed_turn("s9"))

    def test_an_old_session_is_looked_up_once(self) -> None:
        with patch.object(decisions, "_setup_from_log", return_value=None) as lookup:
            self.assertIsNone(self.resumed_turn("older-session"))
            self.assertIsNone(self.resumed_turn("older-session"))
        self.assertEqual(lookup.call_count, 1)

    def test_shadow_applies_nothing(self) -> None:
        with patch.dict(os.environ, {mode.ENV_MODE: "shadow"}):
            kwargs, line = self.first_turn(answer("deep"))
            self.assertIsNone(kwargs)
            self.assertIsNone(self.resumed_turn())
        self.assertEqual(line["applied"]["source"], None)


if __name__ == "__main__":
    unittest.main()
