"""What XO hands the agent this turn, beside the user's own message.

The agent gets it as per-turn context, never inside the user's message and
never as a file in the project. The adapter decides how; for Claude Code it
is a ``UserPromptSubmit`` hook's ``additionalContext``, which our tests showed
can change every turn, keeps the prompt cache and stays in the transcript
(docs: research/context-delivery-tests.md).

Switched by ``XO_INTELLIGENCE_CONTEXT`` (``mode.py``), for agents that ship
intelligence profiles:

- ``note``: a fixed, harmless note on every turn that proves the channel end
  to end (plan step 5).
- ``map``: "where to look" (plan step 6d) on a new session's first turn, in an
  indexed project: at most five files ranked by the request's words, and the
  areas they belong to (``where_to_look.py``), only when sure. No decision
  model is asked, so nothing waits. Later turns add nothing: the note stays
  in the session's history.

The turn line logs the context's kind, size and hash (never the text), and
for ``map`` the areas and files handed over, or why nothing was.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from services.cowork_agent.intelligence import classify, mode, profiles, request_areas, where_to_look

log = logging.getLogger(__name__)

NOTE = (
    "Note from XO Space: this is a check of XO's context channel. "
    "There is nothing to act on; answer the user as you normally would."
)


@dataclass
class TurnContext:
    """``text`` for the agent (``None``: add nothing) and what the turn line logs."""

    text: str | None = None
    record: dict[str, Any] | None = None


async def turn_context(stream_info: dict) -> TurnContext:
    """The context for this turn. Never raises."""
    try:
        current = mode.context_mode()
        if current == mode.OFF or profiles.load(stream_info.get("agent_name") or "") is None:
            return TurnContext()
        if current == mode.CONTEXT_NOTE:
            return TurnContext(NOTE, record(NOTE))
        return await _where_to_look(stream_info)
    except Exception:  # noqa: BLE001 - the reply matters more than the context
        log.exception("intelligence: could not build the turn's context; adding none")
        return TurnContext()


async def _where_to_look(stream_info: dict) -> TurnContext:
    project = stream_info.get("agent_id")
    if not stream_info.get("is_new_session") or not project:
        return TurnContext()
    if await asyncio.to_thread(request_areas.indexed, project) is None:
        return TurnContext()  # the project is not indexed: no hand-over was expected
    request = classify.prepare_content(stream_info.get("question") or "")
    handed = await asyncio.to_thread(where_to_look.build, project, request)
    if not handed.text:
        return TurnContext(None, handed.record)
    return TurnContext(handed.text, {**record(handed.text), **handed.record})


def record(text: str | None) -> dict[str, Any] | None:
    """What the log keeps of a turn's context: its kind, size and hash, not the text."""
    if not text:
        return None
    return {
        "kind": mode.context_mode(),
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
