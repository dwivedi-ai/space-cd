"""A new request's areas in an indexed project (plan step 6c).

A third Sage call per new session, beside the routing choice and tags, only in
projects with a category list and a file map built from it. The areas land on
the decision line; recalibration groups by the primary one, and "where to
look" (6d) shortlists files by the confident ones.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.cowork_agent.intelligence import categories, classify, decisions, file_map, mode, profiles, request_areas
from services.levanto.client import SageResult

PID = "7deb4a22-0789-497d-9399-a2272579fa06"
CONFIG = profiles.parse({
    "schema": 1, "default": {"model": None, "effort": None}, "efforts": ["low", "high"],
    "tiers": ["light", "deep"],
    "profiles": [{"id": "light", "model": None, "effort": "low", "use_when": "questions"},
                 {"id": "deep", "model": "claude-opus-5-5", "effort": "high", "use_when": "big work"}],
}, sha256="cfg-sha")
AREAS = [{"id": f"area_{i}", "description": f"area {i}"} for i in range(12)]


def areas_answer(**p: float) -> SageResult:
    return SageResult(ok=True, status=200, data={"result": {"tags": [
        {"id": a["id"], "probability": p.get(a["id"], 0.05), "applies": p.get(a["id"], 0.05) > 0.5}
        for a in AREAS] + [{"id": "made_up", "probability": 0.99, "applies": True}]}})


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.state = base / "state"
        self.projects = base / "projects"
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(self.state), "XO_PROJECTS_ROOT": str(self.projects),
                                      mode.ENV_MODE: "shadow", mode.ENV_RECALIBRATE: "shadow",
                                      "LEVANTO_API_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)
        xo = self.projects / "demo" / ".xo"
        xo.mkdir(parents=True)
        (xo / "project.json").write_text(json.dumps({"schema": 1, "pid": PID, "name": "demo"}))

    def index_demo(self, files: dict | None = None, ts: str | None = None) -> None:
        target = categories.path_for("demo")
        categories.save(target, AREAS, code_file_count=200, source="drafted")
        doc = categories.load("demo")
        file_map_path = file_map.path_for("demo")
        file_map_path.write_text(json.dumps({"schema": 1, "categories_ts": ts or doc["ts"],
                                             "files": files if files is not None else
                                             {"a.py": {"blob": "1", "tags": {"area_1": [1.0, True]}}}}))


class IndexedTests(_Sandbox):
    def test_only_a_project_with_a_list_and_a_matching_map(self) -> None:
        self.assertIsNone(request_areas.indexed("demo"))  # nothing yet
        self.assertIsNone(request_areas.indexed(None))
        self.index_demo(files={})
        self.assertIsNone(request_areas.indexed("demo"))  # an empty map
        self.index_demo(ts="2020-01-01T00:00:00Z")
        self.assertIsNone(request_areas.indexed("demo"))  # a map from an older list
        self.index_demo()
        self.assertEqual(len(request_areas.indexed("demo")["categories"]), 12)


class AskTests(unittest.TestCase):
    def ask(self, result: SageResult) -> dict:
        with patch.object(request_areas.client, "decide", AsyncMock(return_value=result)) as decide:
            out = asyncio.run(request_areas.ask("fix the chat retry", {"categories": AREAS}, timeout=3))
        question = decide.call_args.args[1]
        self.assertEqual(question["kind"], "tags")
        self.assertIn("change", question["instructions"])
        return out

    def test_the_areas_the_task_would_change(self) -> None:
        out = self.ask(areas_answer(area_3=0.95, area_7=0.6))
        self.assertEqual(out["areas"]["area_3"], [0.95, True])
        self.assertNotIn("made_up", out["areas"])
        self.assertEqual(out["units"], 1)
        self.assertEqual(request_areas.confident(out["areas"]), ["area_3"])  # 0.6 is not confident

    def test_a_failure_is_recorded_not_raised(self) -> None:
        out = self.ask(SageResult(ok=False, status=402, detail="Insufficient balance"))
        self.assertEqual((out["areas"], out["error"]), (None, "balance"))

    def test_confident_areas_strongest_first(self) -> None:
        areas = {"a": [0.85, True], "b": [0.99, True], "c": [0.9, None], "d": [0.95, False]}
        self.assertEqual(request_areas.confident(areas), ["b", "a"])
        self.assertEqual(request_areas.confident(None), [])


class DecisionTests(_Sandbox):
    def decide(self, project: str | None, answer: SageResult) -> list[dict]:
        sage = classify.Decision(profile="deep", reason=classify.SAGE_CHOICE,
                                 sage={"choice": {"chosen": "deep"}, "tags": None, "units": 2, "errors": []})
        with patch.object(decisions.classify, "classify", AsyncMock(return_value=sage)), \
             patch.object(request_areas.client, "decide", AsyncMock(return_value=answer)) as decide:
            pending = {}

            async def go():
                areas = asyncio.get_running_loop().create_future()
                await decisions._decide(CONFIG, "shadow", "sample_agent", "fix the chat retry", "s1",
                                        project, None, None, areas)
                pending["areas"] = areas.result() if areas.done() else "unresolved"
            asyncio.run(go())
            self.calls = decide.call_count
        self.resolved = pending["areas"]
        log = decisions.decision_log.existing_path(project)
        return [json.loads(l) for l in log.read_text().splitlines()]

    def test_an_indexed_projects_decision_carries_the_requests_areas(self) -> None:
        self.index_demo()
        [line] = self.decide("demo", areas_answer(area_3=0.95))
        self.assertEqual(line["areas"]["area_3"], [0.95, True])
        self.assertEqual(line["areas_call"]["units"], 1)
        self.assertEqual(self.resolved["areas"]["area_3"], [0.95, True])  # handed to the first turn
        # recalibration now groups by the request's primary area
        self.assertEqual(line["recalibrate"]["key"], "area=area_3")

    def test_a_project_that_is_not_indexed_gets_no_third_call(self) -> None:
        [line] = self.decide("demo", areas_answer(area_3=0.95))
        self.assertEqual(self.calls, 0)
        self.assertNotIn("areas", line)
        self.assertIsNone(self.resolved)

    def test_a_failed_areas_call_is_logged_and_the_decision_stands(self) -> None:
        self.index_demo()
        [line] = self.decide("demo", SageResult(ok=False, status=503, detail="loading"))
        self.assertNotIn("areas", line)
        self.assertEqual(line["areas_call"]["error"], "server")
        self.assertEqual(line["decision"]["profile"], "deep")


if __name__ == "__main__":
    unittest.main()
