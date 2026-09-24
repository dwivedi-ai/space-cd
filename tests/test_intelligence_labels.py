"""Each finished session is labelled right-sized, under- or over-powered (plan step 4b).

Under: failed turns, a later request that raised the effort, or (once a setup
has 5+ sessions) more turns or cost than 80% of that setup's sessions. Over:
the top tier for one agent turn on a request that writes no code. Labels are
computed when the log is read, so the rules can change without rewriting it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.intelligence import labels, report

LIGHT = {"profile": "light", "model": "claude-haiku-4-5-20251001", "effort": "low", "source": "sage"}
DEEP = {"profile": "deep", "model": "claude-opus-5-5", "effort": "high", "source": "sage"}
DEFAULT = {"profile": None, "model": None, "effort": None, "source": "default"}


def row(applied=LIGHT, turns=2, cost=0.05, failed=0, lines=1, **extra) -> dict:
    return {"applied": applied, "agent_turns": turns, "cost_usd": cost, "failed_turns": failed,
            "turn_lines": lines, "tags": {"needs_code_writing": True}, "top_tier": False,
            "raised_by_request": False, **extra}


class LabelRuleTests(unittest.TestCase):
    def label(self, r, rows=None):
        return labels.label(r, labels.thresholds(rows or [r]))

    def test_no_finished_turn_no_label(self) -> None:
        self.assertEqual(self.label(row(lines=0)), (None, []))

    def test_right_sized(self) -> None:
        self.assertEqual(self.label(row()), ("right", []))

    def test_failed_turns_are_under(self) -> None:
        self.assertEqual(self.label(row(failed=1)), ("under", ["failed turns"]))

    def test_a_raised_effort_is_under(self) -> None:
        self.assertEqual(self.label(row(raised_by_request=True))[0], "under")

    def test_percentiles_need_enough_sessions(self) -> None:
        four = [row(turns=1), row(turns=1), row(turns=2), row(turns=30)]
        self.assertEqual(self.label(four[-1], four), ("right", []))
        six = [row(turns=t, cost=0.05) for t in (1, 1, 1, 2, 2)] + [row(turns=30, cost=0.9)]
        verdict, reasons = self.label(six[-1], six)
        self.assertEqual(verdict, "under")
        self.assertEqual(reasons, ["more agent turns than 80% of light sessions",
                                   "costlier than 80% of light sessions"])
        self.assertEqual(self.label(six[0], six), ("right", []))

    def test_top_tier_for_a_trivial_chat_is_over(self) -> None:
        trivial = {"turns": 1, "tags": {"needs_code_writing": False}, "top_tier": True}
        self.assertEqual(self.label(row(applied=DEFAULT, **trivial))[0], "over")
        self.assertEqual(self.label(row(applied=DEEP, **trivial))[0], "over")
        # Not when Sage was unsure whether it writes code, or it took more turns.
        self.assertEqual(self.label(row(applied=DEFAULT, **{**trivial, "tags": {"needs_code_writing": None}}))[0], "right")
        self.assertEqual(self.label(row(applied=DEFAULT, **{**trivial, "turns": 3}))[0], "right")

    def test_the_top_tier_is_never_under(self) -> None:
        # Nothing stronger to move to: a long or failing top-tier session is hard work.
        six = [row(applied=DEFAULT, top_tier=True, turns=t) for t in (1, 2, 2, 3, 3)]
        six.append(row(applied=DEFAULT, top_tier=True, turns=30, failed=1))
        self.assertEqual(self.label(six[-1], six), ("right", []))

    def test_counts_by_setup(self) -> None:
        rows = [row(), row(failed=1), row(applied=DEFAULT, turns=1, top_tier=True,
                                          tags={"needs_code_writing": False})]
        labels.apply(rows)
        self.assertEqual(labels.counts(rows), {"(default)": {"over": 1}, "light": {"right": 1, "under": 1}})


class ReportTests(unittest.TestCase):
    """The report computes top tier and raised effort from the agent's own profiles."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        state = Path(tmp.name) / "state"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(state), "XO_PROJECTS_ROOT": str(Path(tmp.name) / "p")})
        env.start()
        self.addCleanup(env.stop)
        self.log = state / "sessions" / "intelligence" / "decisions.jsonl"
        self.log.parent.mkdir(parents=True)

    def write(self, *lines: dict) -> None:
        with self.log.open("a") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

    def session(self, sid, applied, turns, *, code=True):
        decision = {"ts": "2026-09-24T10:00:00Z", "type": "intelligence.decision", "schema": 1,
                    "session_id": sid, "runtime": "claude_code", "mode": "on",
                    "sage": {"tags": {"needs_code_writing": {"p": 0.9, "applies": code}}},
                    "decision": {"profile": applied.get("profile"), "reason": "sage_choice"}, "applied": applied}
        lines = [decision]
        for i, (turn_applied, agent_turns) in enumerate(turns):
            lines.append({"ts": f"2026-09-24T10:0{i + 1}:00Z", "type": "intelligence.turn", "schema": 1,
                          "session_id": sid, "runtime": "claude_code", "mode": "on", "new_session": i == 0,
                          "applied": turn_applied, "agent_error": False,
                          "outcome": {"turns": agent_turns, "cost_usd": 0.02, "is_error": False}})
        self.write(*lines)

    def test_a_later_request_for_more_effort_marks_the_session_under(self) -> None:
        raised = {"profile": None, "model": None, "effort": "high", "source": "request"}
        self.session("s1", LIGHT, [(LIGHT, 1), (raised, 4)])
        [r] = report.session_rows()
        self.assertTrue(r["raised_by_request"])
        self.assertEqual((r["label"], r["label_reasons"]), ("under", ["a later request raised the effort"]))

    def test_deep_and_default_are_the_top_tier(self) -> None:
        self.session("s1", DEEP, [(DEEP, 1)], code=False)
        self.session("s2", DEFAULT, [(DEFAULT, 1)], code=False)
        self.session("s3", LIGHT, [(LIGHT, 1)], code=False)
        rows = {r["session_id"]: r for r in report.session_rows()}
        self.assertEqual({k: (r["top_tier"], r["label"]) for k, r in rows.items()},
                         {"s1": (True, "over"), "s2": (True, "over"), "s3": (False, "right")})
        summary = report.summary(list(rows.values()))
        self.assertEqual(summary["labels_by_setup"], {"(default)": {"over": 1}, "deep": {"over": 1}, "light": {"right": 1}})
        self.assertEqual(summary["label_thresholds"], {})


if __name__ == "__main__":
    unittest.main()
