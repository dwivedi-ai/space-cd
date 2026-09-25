"""Recalibration from the past record (plan step 4b-2): log the tier correction
the past would make, apply nothing.

When a new session is decided, XO looks at earlier finished sessions in the
same log with the same key (the request's primary area, else its three
``needs_*`` tags) that ran on the same setup. With at least 5 of them and at
least 60% labelled under-powered, the next such request would move one tier
up the agent's ``tiers`` ladder; with 60% over-powered, one tier down. The
default sits above the ladder's top. ``XO_INTELLIGENCE_RECALIBRATE``:
``off`` (default), ``shadow`` (log it), ``on`` (the same as ``shadow`` until
4b-3 applies it).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.intelligence import classify, decisions, mode, profiles, recalibrate, report, selection

CONFIG = profiles.parse({
    "schema": 1,
    "default": {"model": None, "effort": None},
    "efforts": ["low", "medium", "high"],
    "tiers": ["light", "standard", "deep"],
    "profiles": [
        {"id": "light", "model": "claude-haiku-4-5-20251001", "effort": "low", "use_when": "questions"},
        {"id": "standard", "model": "claude-sonnet-5", "effort": "medium", "use_when": "small changes"},
        {"id": "deep", "model": "claude-opus-5-5", "effort": "high", "use_when": "big work"},
        {"id": "research", "model": "claude-sonnet-5", "effort": "medium", "use_when": "outside info"},
    ],
}, sha256="cfg-sha")

CODE_TAGS = {"needs_code_writing": {"p": 0.9, "applies": True},
             "needs_search": {"p": 0.2, "applies": False},
             "needs_external_lookup": {"p": 0.1, "applies": False}}
CODE_KEY = "code=1,search=0,outside=0"
TALK_TAGS = {"needs_code_writing": {"p": 0.1, "applies": False},
             "needs_search": {"p": 0.1, "applies": False},
             "needs_external_lookup": {"p": 0.1, "applies": False}}


def setup_of(profile: str | None) -> dict:
    chosen = CONFIG.profile(profile) if profile else None
    return {"profile": profile, "model": chosen.model if chosen else None,
            "effort": chosen.effort if chosen else None, "source": "sage" if chosen else "default"}


def past_session(sid: str, profile: str | None, *, tags=CODE_TAGS, turns=2, failed=False,
                 project: str | None = None, areas=None) -> list[dict]:
    """One finished session: its decision line and one turn line."""
    identity = {"project_id": project} if project else {}
    decision = {"ts": f"2026-09-24T10:00:{sid[-2:]}Z", "type": "intelligence.decision", "schema": 1, **identity,
                "session_id": sid, "runtime": "sample_agent", "mode": "on",
                "sage": {"choice": {"chosen": profile}, "tags": tags, "units": 2, "errors": []},
                "decision": {"profile": profile, "reason": "sage_choice"}, "applied": setup_of(profile)}
    if areas is not None:
        decision["areas"] = areas
    turn = {"ts": f"2026-09-24T10:01:{sid[-2:]}Z", "type": "intelligence.turn", "schema": 1, **identity,
            "session_id": sid, "runtime": "sample_agent", "mode": "on", "new_session": True,
            "applied": setup_of(profile), "agent_error": False,
            "outcome": {"turns": turns, "cost_usd": 0.02, "is_error": failed}}
    return [decision, turn]


class SwitchTests(unittest.TestCase):
    def test_off_unless_set(self) -> None:
        for raw, expected in (("", "off"), ("off", "off"), ("shadow", "shadow"), ("ON", "on"), ("banana", "off")):
            with self.subTest(raw=raw), patch.dict(os.environ, {mode.ENV_RECALIBRATE: raw}):
                self.assertEqual(mode.recalibrate_mode(), expected)


class TierTests(unittest.TestCase):
    def test_the_ladder_is_read_from_the_config(self) -> None:
        self.assertEqual(CONFIG.tiers, ("light", "standard", "deep"))

    def test_no_ladder_means_no_tiers(self) -> None:
        document = {"schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low"],
                    "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "x"}]}
        self.assertEqual(profiles.parse(document).tiers, ())

    def test_a_bad_ladder_is_rejected(self) -> None:
        base = {"schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low"],
                "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "x"},
                             {"id": "deep", "model": None, "effort": "low", "use_when": "y"}]}
        for name, tiers in {"not a list": "light", "unknown profile": ["light", "huge"],
                            "twice": ["light", "light"], "not names": [1, 2]}.items():
            with self.subTest(case=name), self.assertRaises(profiles.ProfileError):
                profiles.parse({**base, "tiers": tiers})

    def test_neighbours(self) -> None:
        self.assertEqual(recalibrate.neighbour(CONFIG, "light", "up"), "standard")
        self.assertEqual(recalibrate.neighbour(CONFIG, "standard", "down"), "light")
        self.assertIsNone(recalibrate.neighbour(CONFIG, "deep", "up"))
        self.assertIsNone(recalibrate.neighbour(CONFIG, "light", "down"))
        # The default is above the ladder: down lands on its top, and it never goes up.
        self.assertEqual(recalibrate.neighbour(CONFIG, None, "down"), "deep")
        self.assertIsNone(recalibrate.neighbour(CONFIG, None, "up"))
        # A profile off the ladder is never moved.
        self.assertIsNone(recalibrate.neighbour(CONFIG, "research", "up"))


class KeyTests(unittest.TestCase):
    def test_the_requests_own_tags(self) -> None:
        self.assertEqual(recalibrate.key({"tags": {"needs_code_writing": True, "needs_search": False,
                                                   "needs_external_lookup": False}}), CODE_KEY)
        self.assertEqual(recalibrate.key({"tags": {"needs_code_writing": None, "needs_search": True,
                                                   "needs_external_lookup": False}}),
                         "code=?,search=1,outside=0")

    def test_no_tags_no_key(self) -> None:
        self.assertIsNone(recalibrate.key({"tags": {}}))
        self.assertIsNone(recalibrate.key({}))

    def test_a_confident_area_wins_over_the_tags(self) -> None:
        row = {"tags": {"needs_code_writing": True}, "areas": {"chat_dispatch": [0.95, True],
                                                                "adapters": [0.97, True],
                                                                "ui": [0.99, None]}}
        self.assertEqual(recalibrate.key(row), "area=adapters")

    def test_areas_with_none_confident_fall_back_to_the_tags(self) -> None:
        row = {"tags": {"needs_code_writing": True, "needs_search": False, "needs_external_lookup": False},
               "areas": {"ui": [0.5, None]}}
        self.assertEqual(recalibrate.key(row), CODE_KEY)


def labelled(profile: str | None, label: str, key: str = CODE_KEY, sid: str = "x") -> dict:
    tags = dict(zip(("needs_code_writing", "needs_search", "needs_external_lookup"),
                    (v == "1" if v != "?" else None for v in (p.split("=")[1] for p in key.split(",")))))
    return {"session_id": sid, "applied": setup_of(profile), "tags": tags, "label": label}


class RuleTests(unittest.TestCase):
    def rule(self, profile, rows, key=CODE_KEY):
        return recalibrate.rule(CONFIG, profile=profile, key=key, rows=rows)

    def test_too_little_evidence(self) -> None:
        rows = [labelled("light", "under")] * 4
        self.assertEqual(self.rule("light", rows), {"key": CODE_KEY, "evidence": 4, "result": "insufficient"})

    def test_mostly_under_moves_one_tier_up(self) -> None:
        rows = [labelled("light", "under")] * 3 + [labelled("light", "right")] * 2
        self.assertEqual(self.rule("light", rows), {"from": "light", "to": "standard", "key": CODE_KEY,
                                                   "evidence": 5, "rate": 0.6, "direction": "up"})

    def test_mostly_over_moves_one_tier_down(self) -> None:
        rows = [labelled(None, "over")] * 4 + [labelled(None, "right")]
        self.assertEqual(self.rule(None, rows), {"from": "(default)", "to": "deep", "key": CODE_KEY,
                                                "evidence": 5, "rate": 0.8, "direction": "down"})

    def test_below_the_rate_nothing_moves(self) -> None:
        rows = [labelled("light", "under")] * 2 + [labelled("light", "right")] * 3
        self.assertEqual(self.rule("light", rows), {"key": CODE_KEY, "evidence": 5, "result": "none"})

    def test_only_the_same_setup_and_key_count(self) -> None:
        rows = ([labelled("light", "under")] * 5
                + [labelled("standard", "under")] * 5
                + [labelled("light", "under", key="code=0,search=0,outside=0")] * 5
                + [labelled("light", None)] * 5)  # unfinished sessions have no label
        self.assertEqual(self.rule("standard", rows)["evidence"], 5)
        self.assertEqual(self.rule("light", rows)["to"], "standard")
        self.assertEqual(self.rule("light", rows)["evidence"], 5)

    def test_off_the_ladder_or_at_its_end_nothing_moves(self) -> None:
        self.assertEqual(self.rule("research", [labelled("research", "under")] * 5)["result"], "untiered")
        self.assertEqual(self.rule("deep", [labelled("deep", "under")] * 5)["result"], "none")

    def test_no_key(self) -> None:
        self.assertEqual(self.rule("light", [labelled("light", "under")] * 5, key=None), {"result": "no_key"})


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.state = base / "state"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.state),
                                      "XO_PROJECTS_ROOT": str(base / "projects"),
                                      mode.ENV_MODE: "on", mode.ENV_RECALIBRATE: "shadow"})
        env.start()
        self.addCleanup(env.stop)
        self.log = self.state / "sessions" / "intelligence" / "decisions.jsonl"
        self.log.parent.mkdir(parents=True)
        load = patch.object(profiles, "load", return_value=CONFIG)
        load.start()
        self.addCleanup(load.stop)

    def write(self, *sessions: list[dict]) -> None:
        with self.log.open("a") as f:
            for lines in sessions:
                for line in lines:
                    f.write(json.dumps(line) + "\n")

    def lines(self) -> list[dict]:
        return [json.loads(raw) for raw in self.log.read_text().splitlines()]

    def decide(self, chosen: str | None = "light", tags=CODE_TAGS, request=None, current_mode="on") -> dict:
        picked = classify.Decision(profile=chosen, reason=classify.SAGE_CHOICE if chosen else classify.SAGE_UNKNOWN,
                                   sage={"choice": {"chosen": chosen or "unknown"}, "tags": tags,
                                         "units": 2, "errors": []})
        async def fake_classify(_config, _content, *, timeout, choice_ready=None):
            if choice_ready is not None and not choice_ready.done():
                choice_ready.set_result(chosen)
            return picked

        with patch.object(decisions.classify, "classify", fake_classify):
            asyncio.run(decisions._decide(CONFIG, current_mode, "sample_agent", "fix the retry test",
                                          "new-session", None, request))
        return self.lines()[-1]

    def under_light(self, n: int = 5) -> None:
        self.write(*(past_session(f"s{i:02d}", "light", failed=True) for i in range(n)))


class DecisionLineTests(_Sandbox):
    def test_a_would_be_correction_is_logged_and_not_applied(self) -> None:
        self.under_light(5)
        line = self.decide("light")
        self.assertEqual(line["correction"], {"mode": "shadow", "from": "light", "to": "standard", "key": CODE_KEY,
                                              "evidence": 5, "rate": 1.0, "direction": "up", "applied": False})
        self.assertNotIn("recalibrate", line)
        self.assertEqual(line["applied"]["profile"], "light")
        self.assertEqual(decisions.remembered("new-session")["profile"], "light")

    def test_on_only_logs_too_until_4b3(self) -> None:
        self.under_light(5)
        with patch.dict(os.environ, {mode.ENV_RECALIBRATE: "on"}):
            line = self.decide("light")
        self.assertEqual((line["correction"]["mode"], line["correction"]["applied"]), ("on", False))
        self.assertEqual(line["applied"]["profile"], "light")

    def test_shadow_routing_corrects_the_profile_sage_picked(self) -> None:
        # Shadow routing runs every session on the default; the correction is about Sage's pick.
        self.under_light(5)
        line = self.decide("light", current_mode="shadow")
        self.assertEqual(line["correction"]["from"], "light")
        self.assertIsNone(line["applied"]["profile"])

    def test_nothing_moves_on_too_little_evidence(self) -> None:
        self.under_light(4)
        line = self.decide("light")
        self.assertNotIn("correction", line)
        self.assertEqual(line["recalibrate"], {"mode": "shadow", "key": CODE_KEY, "evidence": 4,
                                               "result": "insufficient"})

    def test_off_adds_nothing(self) -> None:
        self.under_light(5)
        with patch.dict(os.environ, {mode.ENV_RECALIBRATE: "off"}):
            line = self.decide("light")
        self.assertNotIn("correction", line)
        self.assertNotIn("recalibrate", line)

    def test_an_explicit_request_is_never_recalibrated(self) -> None:
        self.under_light(5)
        line = self.decide("light", request=selection.RequestChoice(profile="light"))
        self.assertNotIn("correction", line)
        self.assertEqual(line["recalibrate"], {"mode": "shadow", "result": "request"})

    def test_other_projects_do_not_count(self) -> None:
        self.write(*(past_session(f"s{i:02d}", "light", failed=True, project="other") for i in range(5)))
        other = self.state / "projects" / "other" / "intelligence" / "decisions.jsonl"
        other.parent.mkdir(parents=True)
        self.log.rename(other)
        line = self.decide("light")
        self.assertEqual(line["recalibrate"]["evidence"], 0)

    def test_a_failure_still_writes_the_decision(self) -> None:
        self.under_light(5)
        with patch.object(recalibrate, "rule", side_effect=RuntimeError("boom")), \
             self.assertLogs(recalibrate.log, "ERROR"):
            line = self.decide("light")
        self.assertEqual(line["type"], "intelligence.decision")
        self.assertEqual(line["recalibrate"], {"mode": "shadow", "result": "error"})

    def test_only_the_tail_of_the_log_is_read(self) -> None:
        self.under_light(5)
        with patch.object(recalibrate, "READ_LIMIT", 4):  # two sessions' lines
            line = self.decide("light")
        self.assertEqual(line["recalibrate"]["evidence"], 2)


class ReportTests(_Sandbox):
    def test_the_report_shows_would_be_corrections(self) -> None:
        self.under_light(5)
        self.decide("light")
        rows = report.session_rows()
        self.assertEqual(rows[-1]["correction"]["to"], "standard")
        self.assertIsNone(rows[0]["correction"])
        self.assertEqual(report.summary(rows)["would_correct"], {"light→standard": 1})

    def test_the_table_has_a_correction_column(self) -> None:
        import contextlib
        import io
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import intelligence_report

        self.under_light(5)
        self.decide("light")
        out = io.StringIO()
        with patch.object(sys, "argv", ["intelligence_report.py"]), contextlib.redirect_stdout(out):
            self.assertEqual(intelligence_report.main(), 0)
        header, *rows = out.getvalue().splitlines()
        self.assertIn("correction", header)
        self.assertEqual(sum("↑standard" in r for r in rows), 1)


if __name__ == "__main__":
    unittest.main()
