from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.cowork_agent.bff import session_feedback as routes
from services.cowork_agent.intelligence import feedback
from services.errors import ServiceError

STATUS = {"session_id": "native-1", "agent": "claude_code", "routed": True,
          "applied": {"profile": "light", "model": "claude-haiku-4-5-20251001", "effort": "low", "source": "sage"},
          "answer": None, "answered_at": None}


def client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


class SessionFeedbackRoutesTests(unittest.TestCase):
    def test_get_returns_the_routing_and_the_answer(self) -> None:
        with patch.object(feedback, "status", return_value=STATUS) as st:
            r = client().get("/api/intelligence/sessions/claude_code/native-1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), STATUS)
        self.assertEqual(r.headers["cache-control"], "no-store")
        st.assert_called_once_with("claude_code", "native-1")

    def test_post_records_the_answer(self) -> None:
        answered = {**STATUS, "answer": "no", "answered_at": "2026-09-28T10:00:00Z"}
        with patch.object(feedback, "record", return_value=answered) as rec:
            r = client().post("/api/intelligence/sessions/codex/s-9/feedback",
                              json={"answer": "no", "model": "gpt-5-codex"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "no")
        rec.assert_called_once_with("codex", "s-9", "no", model="gpt-5-codex")

    def test_the_model_is_optional(self) -> None:
        with patch.object(feedback, "record", return_value=STATUS) as rec:
            client().post("/api/intelligence/sessions/claude_code/native-1/feedback", json={"answer": "yes"})
        rec.assert_called_once_with("claude_code", "native-1", "yes", model=None)

    def test_an_unknown_key_is_a_422(self) -> None:
        with patch.object(feedback, "record") as rec:
            r = client().post("/api/intelligence/sessions/claude_code/native-1/feedback",
                              json={"answer": "yes", "comment": "x"})
        self.assertEqual(r.status_code, 422)
        rec.assert_not_called()

    def test_service_errors_become_their_status_and_code(self) -> None:
        with patch.object(feedback, "record", side_effect=ServiceError("invalid_answer", "yes or no")):
            r = client().post("/api/intelligence/sessions/claude_code/native-1/feedback", json={"answer": "maybe"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], {"code": "invalid_answer", "message": "yes or no"})


if __name__ == "__main__":
    unittest.main()
