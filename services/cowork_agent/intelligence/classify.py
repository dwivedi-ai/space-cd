"""Ask Levanto Sage which profile a new session's request needs.

Two questions about the same request, sent at once:

- a ``choice`` over the agent's profiles (each described by its
  ``use_when``) plus ``unknown``. This picks the profile.
- a ``tags`` call for three concrete facts about the work, worded as in our
  routing probes, where Sage scored them at AUC 1.00, 0.98 and 0.78. They
  are logged to explain and sanity-check the choice; they do not decide it.

There is deliberately no "how hard is this?" question: in our tests it
collapsed to a constant.

Sage sees the user's text only: the workspace preamble the frontend appends is
stripped, and the text is cut to one decision unit's worth.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from services.cowork_agent.helpers import strip_workspace_preamble
from services.cowork_agent.intelligence import profiles
from services.levanto import client

log = logging.getLogger(__name__)

#: About 1,000 tokens: well inside one decision unit (4,000 tokens).
CONTENT_MAX_CHARS = 4000

CHOICE_ID = "profile"
CHOICE_INSTRUCTIONS = (
    "The content is a request given to an AI coding agent at the very start of a session. "
    "Which setup fits the work it asks for? Choose unknown when none clearly fits "
    "or the request is too vague to tell."
)
UNKNOWN_DESCRIPTION = "none of the other options clearly fits, or the request is too vague to tell"

TAGS_ID = "needs"
TAGS_INSTRUCTIONS = (
    "The content is a task given to an AI software agent at the very start of a session. "
    "Which of these will the task require?"
)
#: Wording from the routing probes (research/routing-feasibility-probes.md).
TAGS = {
    "needs_external_lookup": "the agent will have to look something up outside this machine, on the web",
    "needs_code_writing": "the agent will have to write or edit code or files",
    "needs_search": "the agent will have to search or grep a codebase to find things",
}

TOKENS_PER_UNIT = 4000  # billed input tokens per decision unit

# Reasons a decision can have.
SAGE_CHOICE = "sage_choice"      # Sage picked a profile
SAGE_UNKNOWN = "sage_unknown"    # Sage said none of the profiles fits
SAGE_UNSURE = "sage_unsure"      # Sage answered null: too close to call
SAGE_ERROR = "sage_error"        # no usable answer (failure, no key, bad response)

_warned_kinds: set[str] = set()


@dataclass
class Decision:
    """What Sage decided. ``profile`` is ``None`` when the default applies."""

    profile: str | None
    reason: str
    #: The record of Sage's answers, as logged.
    sage: dict[str, Any] = field(default_factory=dict)


def prepare_content(text: str) -> str:
    return strip_workspace_preamble(text or "").strip()[:CONTENT_MAX_CHARS]


def choice_question(config: profiles.IntelligenceConfig) -> dict:
    options = [{"option": p.id, "description": p.use_when} for p in config.profiles]
    options.append({"option": profiles.UNKNOWN, "description": UNKNOWN_DESCRIPTION})
    return {"id": CHOICE_ID, "kind": "choice", "instructions": CHOICE_INSTRUCTIONS, "options": options}


def tags_question() -> dict:
    return {
        "id": TAGS_ID,
        "kind": "tags",
        "instructions": TAGS_INSTRUCTIONS,
        "tags": [{"id": tag, "name": f"{tag}: {text}"} for tag, text in TAGS.items()],
    }


def _units(data: dict) -> int:
    """Decision units one successful call cost: max(1, ceil(tokens / 4000))."""
    usage = (data.get("meta") or {}).get("usage") or {}
    tokens = usage.get("billed_input_tokens")
    if isinstance(tokens, (int, float)) and tokens > 0:
        return max(1, math.ceil(tokens / TOKENS_PER_UNIT))
    return 1


def _reasoning_ran(data: dict) -> bool | None:
    reasoning = (data.get("meta") or {}).get("reasoning")
    return reasoning.get("ran") if isinstance(reasoning, dict) else None


def _error(stage: str, result: client.SageResult, detail: str | None = None) -> dict:
    kind = result.kind or "bad_response"
    error = {"stage": stage, "kind": kind}
    if result.status:
        error["status"] = result.status
    error["detail"] = (detail or result.detail)[:200]
    if kind not in _warned_kinds:
        _warned_kinds.add(kind)
        log.warning("intelligence: Sage %s call failed (%s: %s); the default setup applies. "
                    "Logged once per kind.", stage, kind, error["detail"])
    return error


def _read_choice(result: client.SageResult, option_ids: set[str]) -> tuple[dict | None, dict | None]:
    """``(record, error)`` for the choice answer."""
    if not result.ok:
        return None, _error("choice", result)
    answer = result.data.get("result")
    if not isinstance(answer, dict) or "chosen" not in answer:
        return None, _error("choice", result, "the answer has no `chosen`")
    chosen = answer.get("chosen")
    if chosen is not None and chosen not in option_ids:
        return None, _error("choice", result, f"chose {chosen!r}, which was not an option")
    options = {
        item["option"]: item.get("probability")
        for item in answer.get("probabilities") or []
        if isinstance(item, dict) and item.get("option") in option_ids
    }
    return {
        "chosen": chosen,
        "probability": answer.get("probability"),
        "options": options,
        "latency_ms": result.latency_ms,
        "reasoning_ran": _reasoning_ran(result.data),
    }, None


def _read_tags(result: client.SageResult) -> tuple[dict | None, dict | None]:
    if not result.ok:
        return None, _error("tags", result)
    answer = result.data.get("result")
    items = answer.get("tags") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return None, _error("tags", result, "the answer has no `tags`")
    record: dict[str, Any] = {
        item["id"]: {"p": item.get("probability"), "applies": item.get("applies")}
        for item in items
        if isinstance(item, dict) and item.get("id") in TAGS
    }
    record["latency_ms"] = result.latency_ms
    return record, None


async def classify(config: profiles.IntelligenceConfig, content: str, *, timeout: float) -> Decision:
    """Ask both questions at once. Never raises."""
    choice_result, tags_result = await asyncio.gather(
        client.decide(content, choice_question(config), timeout=timeout),
        client.decide(content, tags_question(), timeout=timeout),
    )
    option_ids = set(config.profile_ids) | {profiles.UNKNOWN}
    choice, choice_error = _read_choice(choice_result, option_ids)
    tags, tags_error = _read_tags(tags_result)
    sage = {
        "choice": choice,
        "tags": tags,
        "units": sum(_units(r.data) for r in (choice_result, tags_result) if r.ok),
        "errors": [e for e in (choice_error, tags_error) if e],
    }

    if choice is None:
        return Decision(profile=None, reason=SAGE_ERROR, sage=sage)
    chosen = choice["chosen"]
    if chosen is None:
        return Decision(profile=None, reason=SAGE_UNSURE, sage=sage)
    if chosen == profiles.UNKNOWN:
        return Decision(profile=None, reason=SAGE_UNKNOWN, sage=sage)
    return Decision(profile=chosen, reason=SAGE_CHOICE, sage=sage)
