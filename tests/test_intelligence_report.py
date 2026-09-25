"""The report joins each session's decision to its turns, read-only.

One row per session: what Sage decided and why, what ran, and the summed
outcome of its turns (agent turns, cost, time, tokens, failures), plus the
files it edited from the watcher's stats. The summary says how many decisions
have an outcome yet, and the typical outcome per decided profile.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "quirq-state"
PID = "7deb4a22-0789-497d-9399-a2272579fa06"
sys.path.insert(0, str(ROOT / "scripts"))
import intelligence_report  # noqa: E402


def decision(session_id: str, ts: str, chosen: str | None, reason: str, profile: str | None) -> dict:
    return {"ts": ts, "type": "intelligence.decision", "schema": 1, "pid": PID, "project_id": "demo",
            "session_id": session_id, "runtime": "sample_agent", "mode": "shadow",
            "sage": {"choice": {"chosen": chosen, "probability": 0.9}, "units": 2, "errors": [],
                     "tags": {"needs_code_writing": {"p": 0.9, "applies": True},
                              "needs_search": {"p": 0.4, "applies": None}}},
            "decision": {"profile": profile, "reason": reason, "model": None, "effort": None},
            "applied": {"profile": None, "model": None, "effort": None, "source": None}}


def turn(session_id: str, ts: str, cost: float, turns: int, *, error: bool = False) -> dict:
    return {"ts": ts, "type": "intelligence.turn", "schema": 1, "pid": PID, "project_id": "demo",
            "session_id": session_id, "runtime": "sample_agent", "mode": "shadow", "new_session": False,
            "applied": {"profile": None, "model": None, "effort": None, "source": None},
            "outcome": {"turns": turns, "duration_ms": 60000, "cost_usd": cost, "is_error": error,
                        "tokens": {"input": 10, "output": 20, "cache_read": 1000, "cache_write": 5}},
            "agent_error": False}


class _State(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.state = self.base / "state"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.state),
                                      "XO_PROJECTS_ROOT": str(self.base / "projects")})
        env.start()
        self.addCleanup(env.stop)


class FixtureTests(_State):
    def setUp(self) -> None:
        super().setUp()
        shutil.copytree(FIXTURE, self.state, ignore=shutil.ignore_patterns("README.md"))

    def test_the_sample_reads_back(self) -> None:
        rows = {r["session_id"]: r for r in report.session_rows()}
        self.assertEqual(set(rows), {"33333333-3333-4333-8333-333333333333",
                                     "44444444-4444-4444-8444-444444444444"})
        project = rows["33333333-3333-4333-8333-333333333333"]
        self.assertEqual((project["sage_chosen"], project["reason"], project["turn_lines"]),
                         ("deep", "sage_choice", 1))
        self.assertEqual((project["agent_turns"], project["cost_usd"]), (14, 0.4213))
        root = rows["44444444-4444-4444-8444-444444444444"]
        self.assertEqual((root["reason"], root["sage_errors"], root["turn_lines"]),
                         ("sage_error", ["balance", "balance"], 0))


class JoinTests(_State):
    def setUp(self) -> None:
        super().setUp()
        runtime = self.state / "projects" / PID
        (runtime / "intelligence").mkdir(parents=True)
        lines = [
            decision("s1", "2026-09-24T10:00:00Z", "deep", "sage_choice", "deep"),
            turn("s1", "2026-09-24T10:05:00Z", 0.5, 12),
            decision("s2", "2026-09-24T11:00:00Z", None, "sage_unsure", None),
            turn("s1", "2026-09-24T12:00:00Z", 0.25, 3, error=True),
            decision("s3", "2026-09-23T09:00:00Z", "light", "sage_choice", "light"),
        ]
        (runtime / "intelligence" / "decisions.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in lines) + "{torn line\n", encoding="utf-8")
        (runtime / "stats.json").write_text(json.dumps({"schema": 1, "by_session": {
            "native-1": {"files": ["a.py", "b.py", "c.py"]}}}), encoding="utf-8")
        shard = runtime / "sessions" / "sessionslist.d"
        shard.mkdir(parents=True)
        (shard / "x.json").write_text(json.dumps({"k1": {"sessionId": "s1", "nativeSessionId": "native-1"}}))

    def test_a_session_adds_up_its_turns(self) -> None:
        rows = {r["session_id"]: r for r in report.session_rows()}
        s1 = rows["s1"]
        self.assertEqual((s1["turn_lines"], s1["agent_turns"], s1["cost_usd"], s1["duration_ms"]), (2, 15, 0.75, 120000))
        self.assertEqual((s1["tokens_in"], s1["tokens_out"], s1["failed_turns"]), (20, 40, 1))
        self.assertEqual(s1["files_edited"], 3)
        self.assertEqual(s1["tags"], {"needs_code_writing": True, "needs_search": None})
        self.assertEqual(rows["s2"]["turn_lines"], 0)
        self.assertIsNone(rows["s2"]["files_edited"])

    def test_rows_are_oldest_first_and_since_filters(self) -> None:
        self.assertEqual([r["session_id"] for r in report.session_rows()], ["s3", "s1", "s2"])
        self.assertEqual([r["session_id"] for r in report.session_rows(since="2026-09-24")], ["s1", "s2"])

    def test_the_summary(self) -> None:
        summary = report.summary(report.session_rows())
        self.assertEqual((summary["with_decision"], summary["decisions_with_outcome"]), (3, 1))
        self.assertEqual(summary["reasons"], {"sage_choice": 2, "sage_unsure": 1})
        self.assertEqual(summary["decided_profiles"], {"deep": 1, "(default)": 1, "light": 1})
        self.assertEqual(summary["by_decided_profile"],
                         {"deep": {"sessions": 1, "median_cost_usd": 0.75, "median_agent_turns": 15,
                                   "median_duration_s": 120.0}})

    def test_the_script_prints_json_csv_and_a_table(self) -> None:
        for flag in ("--json", "--csv", None):
            with self.subTest(flag=flag):
                out = io.StringIO()
                with patch.object(sys, "argv", ["intelligence_report.py"] + ([flag] if flag else [])), \
                     contextlib.redirect_stdout(out):
                    self.assertEqual(intelligence_report.main(), 0)
                text = out.getvalue()
                if flag == "--json":
                    self.assertEqual(json.loads(text)["summary"]["with_decision"], 3)
                else:
                    self.assertIn("s1" if flag else "sage_choice", text)


if __name__ == "__main__":
    unittest.main()
