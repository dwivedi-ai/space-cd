"""Levanto Sage classifies a new session's request against the agent's profiles.

One ``choice`` over the profiles (by ``use_when``) plus ``unknown`` picks the
profile; one ``tags`` call records three concrete facts. Sage sees only the
user's text, stripped of the workspace preamble. A pick is used; ``unknown``,
a ``null`` (not sure) or any failure leaves the default in place, and the
record says which.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest.mock import patch

from services.cowork_agent.intelligence import classify, profiles
from services.levanto.client import SageResult

CONFIG = profiles.parse({
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "high"],
    "profiles": [
        {"id": "light", "model": None, "effort": "low", "use_when": "questions; no code changes"},
        {"id": "deep", "model": None, "effort": "high", "use_when": "work across the codebase"},
    ],
}, sha256="cfg")

META = {"model": "levanto-sage-v1.1", "usage": {"billed_input_tokens": 120},
        "reasoning": {"fired": False, "ran": False}}


def choice_answer(chosen, probability=0.8, units_tokens=120) -> SageResult:
    meta = copy.deepcopy(META)
    meta["usage"]["billed_input_tokens"] = units_tokens
    return SageResult(ok=True, status=200, latency_ms=210.0, data={
        "id": "profile", "kind": "choice",
        "result": {"chosen": chosen, "probability": probability if chosen else None,
                   "probabilities": [{"option": "light", "probability": 0.1},
                                     {"option": "deep", "probability": 0.8},
                                     {"option": "unknown", "probability": 0.05}]},
        "meta": meta,
    })


def tags_answer() -> SageResult:
    return SageResult(ok=True, status=200, latency_ms=1200.0, data={
        "id": "needs", "kind": "tags",
        "result": {"tags": [
            {"id": "needs_external_lookup", "probability": 0.02, "applies": False},
            {"id": "needs_code_writing", "probability": 0.97, "applies": True},
            {"id": "needs_search", "probability": 0.55, "applies": None},
        ]},
        "meta": META,
    })


def failure(status: int, detail: str) -> SageResult:
    return SageResult(ok=False, status=status, detail=detail, latency_ms=90.0)


class _Classify(unittest.TestCase):
    def setUp(self) -> None:
        classify._warned_kinds.clear()
        self.addCleanup(classify._warned_kinds.clear)
        self.asked: list[tuple[str, dict]] = []

    def run_with(self, choice: SageResult, tags: SageResult | None = None) -> classify.Decision:
        async def fake_decide(content, question, *, timeout):
            self.asked.append((content, question))
            return choice if question["kind"] == "choice" else (tags or tags_answer())

        with patch.object(classify.client, "decide", fake_decide):
            return asyncio.run(classify.classify(CONFIG, "fix the flaky retry test", timeout=5))


class QuestionTests(unittest.TestCase):
    def test_the_choice_is_over_the_profiles_plus_unknown(self) -> None:
        question = classify.choice_question(CONFIG)
        self.assertEqual(question["kind"], "choice")
        self.assertEqual(question["options"], [
            {"option": "light", "description": "questions; no code changes"},
            {"option": "deep", "description": "work across the codebase"},
            {"option": "unknown", "description": classify.UNKNOWN_DESCRIPTION},
        ])

    def test_the_tags_are_the_three_measured_facts(self) -> None:
        question = classify.tags_question()
        self.assertEqual([t["id"] for t in question["tags"]],
                         ["needs_external_lookup", "needs_code_writing", "needs_search"])
        for tag in question["tags"]:
            self.assertTrue(tag["name"].startswith(f"{tag['id']}: the agent will have to"))

    def test_sage_sees_the_users_text_only(self) -> None:
        text = "Fix the login bug\n\n---\n\n> **Project context**\n> This project has an AGENTS.md…"
        self.assertEqual(classify.prepare_content(text), "Fix the login bug")
        self.assertEqual(len(classify.prepare_content("x" * 10_000)), classify.CONTENT_MAX_CHARS)


class DecisionTests(_Classify):
    def test_sage_picks_a_profile(self) -> None:
        decision = self.run_with(choice_answer("deep"))
        self.assertEqual((decision.profile, decision.reason), ("deep", classify.SAGE_CHOICE))
        self.assertEqual(decision.sage["choice"]["options"], {"light": 0.1, "deep": 0.8, "unknown": 0.05})
        self.assertEqual(decision.sage["choice"]["reasoning_ran"], False)
        self.assertEqual(decision.sage["tags"]["needs_code_writing"], {"p": 0.97, "applies": True})
        self.assertEqual(decision.sage["tags"]["needs_search"], {"p": 0.55, "applies": None})
        self.assertEqual(decision.sage["units"], 2)
        self.assertEqual(decision.sage["errors"], [])
        self.assertEqual({content for content, _ in self.asked}, {"fix the flaky retry test"})

    def test_unknown_and_unsure_leave_the_default(self) -> None:
        self.assertEqual(self.run_with(choice_answer("unknown")).reason, classify.SAGE_UNKNOWN)
        unsure = self.run_with(choice_answer(None))
        self.assertEqual((unsure.profile, unsure.reason), (None, classify.SAGE_UNSURE))

    def test_a_failed_choice_leaves_the_default(self) -> None:
        with self.assertLogs(classify.log, "WARNING"):
            decision = self.run_with(failure(402, "Key is valid but account balance is too low"))
        self.assertEqual((decision.profile, decision.reason), (None, classify.SAGE_ERROR))
        self.assertEqual(decision.sage["errors"], [{"stage": "choice", "kind": "balance", "status": 402,
                                                    "detail": "Key is valid but account balance is too low"}])
        self.assertEqual(decision.sage["units"], 1)  # only the tags call was billed

    def test_an_answer_outside_the_options_is_not_used(self) -> None:
        with self.assertLogs(classify.log, "WARNING"):
            decision = self.run_with(choice_answer("turbo"))
        self.assertEqual(decision.reason, classify.SAGE_ERROR)
        self.assertEqual(decision.sage["errors"][0]["kind"], "bad_response")

    def test_failed_tags_do_not_block_the_choice(self) -> None:
        with self.assertLogs(classify.log, "WARNING"):
            decision = self.run_with(choice_answer("light"), failure(503, "Service is still loading."))
        self.assertEqual((decision.profile, decision.reason), ("light", classify.SAGE_CHOICE))
        self.assertIsNone(decision.sage["tags"])
        self.assertEqual(decision.sage["errors"][0]["stage"], "tags")

    def test_units_follow_billed_tokens(self) -> None:
        decision = self.run_with(choice_answer("deep", units_tokens=9000))
        self.assertEqual(decision.sage["units"], 3 + 1)

    def test_each_failure_kind_warns_once(self) -> None:
        with self.assertLogs(classify.log, "WARNING") as logs:
            for _ in range(3):
                self.run_with(failure(402, "balance"), failure(402, "balance"))
        self.assertEqual(len(logs.records), 1)


if __name__ == "__main__":
    unittest.main()
