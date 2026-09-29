"""Where to look: a short, confident file list for a new request (plan step 6d).

Files in the request's confident areas are ranked by how well the request's
words match their path and outline, plus past sessions in the same areas
that edited them. At most five are handed over, and only when some file
matches and the front-runners stand out; otherwise nothing is sent and the
reason is logged.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import categories, decision_log, file_map, where_to_look

PID = "7deb4a22-0789-497d-9399-a2272579fa06"
AREAS = [{"id": "chat_engine", "description": "chat"}, {"id": "storage", "description": "disk"}] + [
    {"id": f"area_{i}", "description": f"area {i}"} for i in range(10)]


def entry(area: str, outline: str, applies=True) -> dict:
    return {"blob": "1", "tags": {area: [0.99 if applies else 0.5, applies]}, "outline": outline}


FILES = {
    "routers/chat.py": entry("chat_engine", "Chat prompt API; chat_prompt; chat_stream"),
    "services/engine/dispatcher.py": entry("chat_engine", "Dispatch a prompt to the agent; dispatch"),
    "services/engine/retry.py": entry("chat_engine", "Retry a failed stream; retry_stream; backoff"),
    "services/engine/unsure.py": entry("chat_engine", "Stream retry helpers; retry_later", applies=None),
    "services/storage/disk.py": entry("storage", "Write files atomically; retry_write"),
}


class RankTests(unittest.TestCase):
    def test_only_files_in_the_confident_areas_ranked_by_the_requests_words(self) -> None:
        ranked = where_to_look.rank("the stream retry backoff is too short", ["chat_engine"], FILES, Counter())
        paths = [p for p, _score in ranked]
        self.assertNotIn("services/storage/disk.py", paths)  # another area, though it says "retry"
        self.assertIn("services/engine/unsure.py", paths)  # unsure for the area stays in
        self.assertEqual(paths[0], "services/engine/retry.py")

    def test_past_edits_in_the_same_areas_lift_a_file(self) -> None:
        ranked = dict(where_to_look.rank("dispatch", ["chat_engine"], FILES, Counter({"routers/chat.py": 5})))
        self.assertGreater(ranked["routers/chat.py"], 0)
        plain = dict(where_to_look.rank("dispatch", ["chat_engine"], FILES, Counter()))
        self.assertEqual(plain["routers/chat.py"], 0)


class _Project(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        env = patch.dict(os.environ, {"QUIRQ_STATE_ROOT": str(base / "state"), "XO_PROJECTS_ROOT": str(base / "p")})
        env.start()
        self.addCleanup(env.stop)
        xo = base / "p" / "demo" / ".xo"
        xo.mkdir(parents=True)
        (xo / "project.json").write_text(json.dumps({"schema": 1, "pid": PID, "name": "demo"}))
        categories.save(categories.path_for("demo"), AREAS, code_file_count=200, source="drafted")
        self.write_map(FILES)

    def write_map(self, files: dict) -> None:
        doc = categories.load("demo")
        file_map.path_for("demo").write_text(json.dumps({"schema": 1, "categories_ts": doc["ts"], "files": files}))

    def build(self, text="the stream retry backoff is too short"):
        return where_to_look.build("demo", text)


class BuildTests(_Project):
    def test_a_confident_hand_over(self) -> None:
        handed = self.build()
        self.assertEqual(handed.record["kind"], "map")
        self.assertEqual(handed.record["areas"][0], "chat_engine")
        self.assertEqual(handed.record["files"][0], "services/engine/retry.py")
        self.assertLessEqual(len(handed.record["files"]), where_to_look.MAX_FILES)
        self.assertIn("`services/engine/retry.py`", handed.text)
        self.assertIn("chat engine", handed.text)
        self.assertIn("starting point, not a constraint", handed.text)
        self.assertLessEqual(len(handed.text), where_to_look.TEXT_MAX)

    def test_the_areas_come_from_the_files_own_tags(self) -> None:
        # No decision model: the areas named are the ones the front-runners are tagged with.
        self.assertEqual(where_to_look.areas_of(["services/engine/retry.py", "routers/chat.py",
                                                 "services/storage/disk.py"], FILES), ["chat_engine", "storage"])
        # A file only unsure for an area does not name it.
        self.assertEqual(where_to_look.areas_of(["services/engine/unsure.py"], FILES), [])

    def test_withheld_when_no_file_matches(self) -> None:
        handed = self.build(text="rename the logo")
        self.assertEqual((handed.text, handed.record["withheld"]), (None, "no file matches the request"))

    def test_withheld_when_nothing_stands_out(self) -> None:
        # Twelve files that all match the request equally well.
        self.write_map({f"services/engine/stream_{i}.py": entry("chat_engine", "stream helpers; stream")
                        for i in range(12)})
        handed = self.build(text="fix the stream")
        self.assertEqual((handed.text, handed.record["withheld"]), (None, "no file stands out"))

    def test_withheld_when_the_project_is_not_indexed(self) -> None:
        file_map.path_for("demo").unlink()
        self.assertEqual(self.build().record["withheld"], "project not indexed")


class PastWorkTests(_Project):
    def test_files_edited_by_past_sessions_in_the_same_areas(self) -> None:
        runtime = decision_log.existing_path("demo").parent.parent
        (runtime / "stats.json").write_text(json.dumps({"by_session": {
            "native-a": {"files": ["routers/chat.py", "services/engine/retry.py"]},
            "native-b": {"files": ["services/storage/disk.py"]},
            "native-c": {"files": ["routers/chat.py"]},
        }}))
        for native, xo in (("native-a", "xo-a"), ("native-b", "xo-b"), ("native-c", "xo-c")):
            sessions_io.write_session_row("demo", f"claude_code:{xo}", {"sessionId": xo, "nativeSessionId": native})
        log = decision_log.existing_path("demo")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            for xo, area in (("xo-a", "chat_engine"), ("xo-b", "storage"), ("xo-c", "chat_engine")):
                f.write(json.dumps({"type": "intelligence.decision", "session_id": xo,
                                    "areas": {area: [0.95, True]}}) + "\n")
        edits = where_to_look.past_edits("demo", {"chat_engine"})
        self.assertEqual(edits, Counter({"routers/chat.py": 2, "services/engine/retry.py": 1}))
        handed = self.build()
        self.assertIn("routers/chat.py", handed.record["files"])


if __name__ == "__main__":
    unittest.main()
