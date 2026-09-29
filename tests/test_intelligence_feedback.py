"""A person's answer to "did this chat still complete your task?" (plan step 4b).

Any session in Agents > Sessions can be rated, routed by XO or not. Answers go
to one machine-local log with the model the session ran on and a preview and
hash of its first prompt; a routed session's answer also names its XO session
id and the layer it ran on, which is how the report joins it to the decision.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import feedback
from services.errors import ServiceError

LIGHT = {"profile": "light", "model": "claude-haiku-4-5-20251001", "effort": "low", "source": "sage"}


class FeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.state = base / "state"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.state), "XO_PROJECTS_ROOT": str(base / "p")})
        env.start()
        self.addCleanup(env.stop)
        (base / "p").mkdir()
        prompt = patch.object(feedback, "_first_prompt", return_value="write a polite email to a client")
        self.first_prompt = prompt.start()
        self.addCleanup(prompt.stop)

    def routed(self, xo_id="xo-1", native="native-1") -> None:
        """A chat XO routed: a session index row and its decision line."""
        sessions_io.write_session_row("", f"claude_code:{xo_id}",
                                      {"sessionId": xo_id, "nativeSessionId": native, "backend": "claude_code"})
        log = self.state / "sessions" / "intelligence" / "decisions.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps({"ts": "2026-09-28T10:00:00Z", "type": "intelligence.decision", "schema": 1,
                                "session_id": xo_id, "runtime": "claude_code", "applied": LIGHT}) + "\n")

    def lines(self) -> list[dict]:
        path = self.state / "sessions" / "intelligence" / "feedback.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def test_a_routed_session_is_found_by_the_agents_own_id(self) -> None:
        self.routed()
        status = feedback.status("claude_code", "native-1")
        self.assertTrue(status["routed"])
        self.assertEqual(status["applied"], LIGHT)
        self.assertIsNone(status["answer"])
        # and by XO's id
        self.assertTrue(feedback.status("claude_code", "xo-1")["routed"])

    def test_a_session_started_elsewhere_is_not_routed(self) -> None:
        status = feedback.status("claude_code", "terminal-7")
        self.assertEqual((status["routed"], status["applied"], status["answer"]), (False, None, None))

    def test_an_answer_is_recorded_with_the_layer_and_the_first_prompt(self) -> None:
        self.routed()
        status = feedback.record("claude_code", "native-1", "no", model=None)
        self.assertEqual(status["answer"], "no")
        [line] = self.lines()
        self.assertEqual(line["type"], "intelligence.feedback")
        self.assertEqual((line["session_id"], line["xo_session_id"], line["project_id"]), ("native-1", "xo-1", None))
        self.assertEqual((line["agent"], line["routed"], line["answer"]), ("claude_code", True, "no"))
        self.assertEqual(line["applied"], LIGHT)
        self.assertEqual(line["model"], LIGHT["model"])
        self.assertEqual(line["prompt"]["preview"], "write a polite email to a client")
        self.assertEqual(len(line["prompt"]["sha256"]), 64)

    def test_an_unrouted_session_keeps_the_model_it_ran_on(self) -> None:
        feedback.record("codex", "terminal-7", "yes", model="gpt-5-codex")
        [line] = self.lines()
        self.assertEqual((line["routed"], line["xo_session_id"], line["applied"]), (False, None, None))
        self.assertEqual((line["agent"], line["model"], line["answer"]), ("codex", "gpt-5-codex", "yes"))

    def test_the_latest_answer_wins(self) -> None:
        self.routed()
        feedback.record("claude_code", "native-1", "no", model=None)
        feedback.record("claude_code", "native-1", "yes", model=None)
        self.assertEqual(feedback.status("claude_code", "native-1")["answer"], "yes")
        self.assertEqual(feedback.latest_answers(), {"xo-1": "yes"})

    def test_a_prompt_that_cannot_be_read_is_left_out(self) -> None:
        self.first_prompt.return_value = None
        feedback.record("claude_code", "terminal-7", "yes", model=None)
        self.assertIsNone(self.lines()[0]["prompt"])

    def test_bad_input_is_refused(self) -> None:
        cases = {
            "invalid_answer": ("claude_code", "s1", "maybe", None),
            "invalid_session": ("claude_code", "../../etc", "yes", None),
            "invalid_agent": ("claude code!", "s1", "yes", None),
            "invalid_model": ("claude_code", "s1", "yes", "--dangerous"),
        }
        for code, args in cases.items():
            with self.subTest(code=code), self.assertRaises(ServiceError) as caught:
                feedback.record(*args[:3], model=args[3])
            self.assertEqual((caught.exception.code, caught.exception.status), (code, 400))
        self.assertEqual(self.lines(), [])


if __name__ == "__main__":
    unittest.main()
