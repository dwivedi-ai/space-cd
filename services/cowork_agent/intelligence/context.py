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
  indexed project: the request's confident areas and at most five files
  (``where_to_look.py``), only when sure. The first turn waits for the
  request's areas no longer than routing's own deadline; late means nothing
  is sent. Later turns add nothing: the note stays in the session's history.

The turn line logs the context's kind, size and hash (never the text), and
for ``map`` the areas and files handed over, or why nothing was.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from services.cowork_agent.intelligence import classify, decisions, mode, profiles, where_to_look

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
    pending = stream_info.get("intelligence_decision")
    if not stream_info.get("is_new_session") or pending is None or pending.areas is None:
        return TurnContext()
    remaining = max(0.0, decisions.ON_WAIT_S - (time.monotonic() - pending.started))
    try:
        found = await asyncio.wait_for(asyncio.shield(pending.areas), timeout=remaining)
    except asyncio.TimeoutError:
        return TurnContext(None, {"kind": where_to_look.KIND, "withheld": "late"})
    if found is None:
        return TurnContext()  # the project is not indexed: no hand-over was expected
    if found.get("areas") is None:
        return TurnContext(None, {"kind": where_to_look.KIND, "withheld": f"no areas ({found.get('error')})"})
    request = classify.prepare_content(stream_info.get("question") or "")
    handed = await asyncio.to_thread(where_to_look.build, stream_info.get("agent_id"), request, found["areas"])
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
