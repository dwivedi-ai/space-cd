"""A chat request can choose its profile, effort and model.

``profile`` and ``effort`` are new, optional fields checked against the
agent's ``intelligence.json`` (unknown values are a 400). ``model`` predates
them: a routing id such as ``claude_code/main`` is ignored as it always was, a
bare model id is honoured. An explicit field wins over the profile's value,
and a turn that chooses nothing passes no flag at all.
"""

from __future__ import annotations

import asyncio
import copy
import unittest
from unittest.mock import patch

from services.cowork_agent.intelligence import decisions, profiles, selection

DOCUMENT = {
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "medium", "high"],
    "profiles": [
        {"id": "light", "model": None, "effort": "low", "use_when": "questions"},
        {"id": "deep", "model": "claude-opus-5-5", "effort": "high", "use_when": "big work"},
    ],
}
CONFIG = profiles.parse(copy.deepcopy(DOCUMENT))


class _WithConfig(unittest.TestCase):
    config: profiles.IntelligenceConfig | None = CONFIG

    def setUp(self) -> None:
        patcher = patch.object(selection.profiles, "load", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)


class ParseRequestTests(_WithConfig):
    def test_a_body_without_the_fields_chooses_nothing(self) -> None:
        self.assertTrue(selection.parse_request({"text": "hi"}, "sample_agent").empty)

    def test_known_profile_and_effort(self) -> None:
        choice = selection.parse_request({"profile": "deep", "effort": "low"}, "sample_agent")
        self.assertEqual(choice, selection.RequestChoice(profile="deep", effort="low"))

    def test_blank_values_are_absent(self) -> None:
        choice = selection.parse_request({"profile": " ", "effort": "", "model": None}, "sample_agent")
        self.assertTrue(choice.empty)

    def test_an_unknown_profile_names_the_choices(self) -> None:
        with self.assertRaises(selection.RequestError) as caught:
            selection.parse_request({"profile": "turbo"}, "sample_agent")
        self.assertIn("light, deep", str(caught.exception))

    def test_an_unknown_effort_names_the_choices(self) -> None:
        with self.assertRaises(selection.RequestError) as caught:
            selection.parse_request({"effort": "max"}, "sample_agent")
        self.assertIn("low, medium, high", str(caught.exception))

    def test_the_legacy_model_field(self) -> None:
        cases = {
            "claude_code/main": None,          # a routing id from /api/models
            "openrouter/anthropic/x": None,
            "-p": None,                         # never a flag
            "opus 5": None,
            123: None,
            "claude-opus-5-5": "claude-opus-5-5",
            "sonnet": "sonnet",
            "claude-opus-5-5[1m]": "claude-opus-5-5[1m]",
        }
        for value, expected in cases.items():
            with self.subTest(model=value):
                self.assertEqual(selection.parse_request({"model": value}, "sample_agent").model, expected)


class NoProfilesTests(_WithConfig):
    config = None

    def test_an_agent_without_profiles_ignores_the_fields(self) -> None:
        self.assertIsNone(selection.parse_request({"profile": "turbo", "effort": "huge"}, "sample_agent"))

    def test_and_gets_no_intelligence_keyword(self) -> None:
        info = {"agent_name": "sample_agent", "intelligence_request": None}
        self.assertIsNone(asyncio.run(decisions.turn_selection(info)))


class SelectTests(unittest.TestCase):
    def select(self, **fields) -> selection.Selection:
        return selection.select(CONFIG, selection.RequestChoice(**fields))

    def test_nothing_chosen_passes_no_flag(self) -> None:
        self.assertEqual(selection.select(CONFIG, None), selection.Selection(None, None, None, None))

    def test_a_profile_brings_its_model_and_effort(self) -> None:
        self.assertEqual(self.select(profile="deep"),
                         selection.Selection("deep", "claude-opus-5-5", "high", "request"))

    def test_an_explicit_field_wins_over_the_profile(self) -> None:
        self.assertEqual(self.select(profile="deep", effort="low"),
                         selection.Selection("deep", "claude-opus-5-5", "low", "request"))
        self.assertEqual(self.select(profile="deep", model="sonnet"),
                         selection.Selection("deep", "sonnet", "high", "request"))

    def test_a_model_alone(self) -> None:
        self.assertEqual(self.select(model="sonnet"), selection.Selection(None, "sonnet", None, "request"))


class TurnSelectionTests(_WithConfig):
    def turn(self, request):
        return asyncio.run(decisions.turn_selection(
            {"agent_name": "sample_agent", "our_session_id": "s1", "intelligence_request": request}
        ))

    def test_a_turn_that_chooses_nothing_passes_nothing(self) -> None:
        self.assertIsNone(self.turn(None))
        self.assertIsNone(self.turn(selection.RequestChoice()))

    def test_a_chosen_profile_becomes_the_adapter_keyword(self) -> None:
        self.assertEqual(self.turn(selection.RequestChoice(profile="light")),
                         {"profile": "light", "model": None, "effort": "low", "source": "request"})

    def test_a_failure_falls_back_to_no_flags(self) -> None:
        with patch.object(selection, "select", side_effect=RuntimeError("boom")), \
             self.assertLogs(decisions.log, "ERROR"):
            self.assertIsNone(self.turn(selection.RequestChoice(profile="light")))


if __name__ == "__main__":
    unittest.main()
