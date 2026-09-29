"""BFF routes for the "routing check" on a session (plan step 4b).

  GET  /api/intelligence/sessions/{agent}/{session_id}           {session_id, agent, routed, applied, answer, answered_at}
  POST /api/intelligence/sessions/{agent}/{session_id}/feedback  body {answer: "yes"|"no", model?}; the same shape back

Asked on the session detail in Agents > Sessions, for any session: routed by
XO or not. ``session_id`` is the id the session view has (the agent's own; XO's
works too). Declarative over ``services.cowork_agent.intelligence.feedback``;
typed errors become HTTP through ``bff/errors.py``. Plain ``def`` handlers:
the service does file I/O, so FastAPI runs them in its threadpool. No
os/pathlib in this module (BFF rule P2). Bodies are strict: an unknown key is
a 422.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.cowork_agent.intelligence import feedback
from services.errors import ServiceError

from routers.cowork_agent.bff.errors import ForbidExtra, http_error

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}


class FeedbackBody(ForbidExtra):
    answer: str
    model: Optional[str] = None


@router.get("/api/intelligence/sessions/{agent}/{session_id}")
def session_routing(agent: str, session_id: str):
    try:
        return JSONResponse(feedback.status(agent, session_id), headers=_NO_STORE)
    except ServiceError as exc:
        raise http_error(exc) from exc


@router.post("/api/intelligence/sessions/{agent}/{session_id}/feedback")
def session_feedback(agent: str, session_id: str, body: FeedbackBody):
    try:
        return JSONResponse(feedback.record(agent, session_id, body.answer, model=body.model), headers=_NO_STORE)
    except ServiceError as exc:
        raise http_error(exc) from exc
