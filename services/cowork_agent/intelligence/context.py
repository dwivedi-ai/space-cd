"""What XO hands the agent this turn, beside the user's own message.

The agent gets it as per-turn context, never inside the user's message and
never as a file in the project. The adapter decides how; for Claude Code it
is a ``UserPromptSubmit`` hook's ``additionalContext``, which our tests showed
can change every turn, keeps the prompt cache and stays in the transcript
(docs: research/context-delivery-tests.md).

Switched by ``XO_INTELLIGENCE_CONTEXT`` (``mode.py``), for agents that ship
intelligence profiles. Today (plan step 5) the only content is a fixed,
harmless note that proves the channel end to end. Choosing what is actually
relevant (where to look, which house rules apply, similar past work) is
step 6 and replaces :data:`NOTE`.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from services.cowork_agent.intelligence import mode, profiles

log = logging.getLogger(__name__)

NOTE = (
    "Note from XO Space: this is a check of XO's context channel. "
    "There is nothing to act on; answer the user as you normally would."
)


def turn_context(stream_info: dict) -> str | None:
    """The context for this turn, or ``None`` to add nothing. Never raises."""
    try:
        if mode.context_mode() == mode.OFF:
            return None
        if profiles.load(stream_info.get("agent_name") or "") is None:
            return None
        return NOTE
    except Exception:  # noqa: BLE001 - the reply matters more than the context
        log.exception("intelligence: could not build the turn's context; adding none")
        return None


def record(text: str | None) -> dict[str, Any] | None:
    """What the log keeps of a turn's context: its kind, size and hash, not the text."""
    if not text:
        return None
    return {
        "kind": mode.context_mode(),
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
