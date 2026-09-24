"""After every turn: one ``intelligence.turn`` line saying what it ran with and cost.

A session never really ends (it can be resumed days later), so its outcome
is recorded per turn and added up when the log is read: the session's
decision line and its turn lines share ``session_id``. The numbers are the
agent's own, from the adapter's ``done`` event (``outcome``).

Written in ``shadow`` and ``on`` only, for agents with profiles, in the
background after the turn's reply has been streamed; a failure is logged and
never reaches the chat.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from services.cowork_agent.intelligence import decision_log, decisions, mode, profiles

log = logging.getLogger(__name__)

_NOTHING_APPLIED = {"profile": None, "model": None, "effort": None, "source": None}

# Strong references: the event loop keeps only weak ones to running tasks.
_tasks: set[asyncio.Task] = set()


def after_turn(
    stream_info: dict,
    applied: dict[str, Any] | None,
    done_event: dict[str, Any] | None,
    *,
    agent_error: bool,
) -> asyncio.Task | None:
    """Record a finished turn in the background. Returns at once."""
    try:
        current = mode.mode()
        agent_name = stream_info.get("agent_name") or ""
        if current == mode.OFF or profiles.load(agent_name) is None:
            return None
        outcome = (done_event or {}).get("outcome")
        task = asyncio.get_running_loop().create_task(asyncio.to_thread(
            _record, stream_info, current, applied or dict(_NOTHING_APPLIED),
            outcome if isinstance(outcome, dict) else None, agent_error,
        ))
    except Exception:  # noqa: BLE001 - recording a turn must never cost a reply
        log.exception("intelligence: could not record a turn")
        return None
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def _record(stream_info: dict, current_mode: str, applied: dict, outcome: dict | None, agent_error: bool) -> None:
    session_id = stream_info.get("our_session_id")
    is_new = bool(stream_info.get("is_new_session"))
    project = decisions.session_project(session_id, stream_info.get("agent_id"), new_session=is_new)
    decision_log.record_turn(
        project,
        session_id=session_id,
        runtime=stream_info.get("agent_name") or "",
        mode=current_mode,
        new_session=is_new,
        applied=applied,
        outcome=outcome,
        agent_error=agent_error,
    )
