"""Which setup a chat turn runs with.

A request may name a ``profile`` (an id from the agent's
``intelligence.json``), an ``effort``, and a ``model``. Each field is
optional and an explicit field always wins over the profile's value for that
field. ``model`` predates profiles: a value holding a ``/`` is a routing id
from ``/api/models`` (``<prefix>/<agent>``), not a model, and is ignored as it
always was.

A turn with none of these fields runs exactly as before: no flags.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from services.cowork_agent.intelligence import profiles

log = logging.getLogger(__name__)


class RequestError(ValueError):
    """A request named a profile or effort this agent does not have (HTTP 400)."""


@dataclass(frozen=True)
class RequestChoice:
    """What the request itself asked for; every field optional."""

    profile: str | None = None
    model: str | None = None
    effort: str | None = None

    @property
    def empty(self) -> bool:
        return self.profile is None and self.model is None and self.effort is None


@dataclass(frozen=True)
class Selection:
    """The setup one turn runs with. ``None`` fields pass no flag."""

    profile: str | None
    model: str | None
    effort: str | None
    #: where the profile came from: "request", or None when nothing was chosen
    source: str | None

    def as_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_request(body: dict, agent_name: str) -> RequestChoice | None:
    """Read ``profile`` / ``effort`` / ``model`` from a chat request body.

    Returns ``None`` when the agent has no profiles, so the fields are
    ignored for it. Raises :class:`RequestError` for a profile or effort the
    agent does not have; the legacy ``model`` field never fails a request.
    """
    config = profiles.load(agent_name)
    if config is None:
        return None

    profile = _text(body.get("profile"))
    if profile is not None and config.profile(profile) is None:
        raise RequestError(
            f"unknown profile {profile!r}; choose one of: {', '.join(config.profile_ids)}"
        )

    effort = _text(body.get("effort"))
    if effort is not None and effort not in config.efforts:
        raise RequestError(
            f"unknown effort {effort!r}; choose one of: {', '.join(config.efforts)}"
        )

    model = _text(body.get("model"))
    if model is not None and ("/" in model or not profiles.is_valid_model(model)):
        log.debug("intelligence: request model %r is not a model id; ignored", model)
        model = None

    return RequestChoice(profile=profile, model=model, effort=effort)


def select(config: profiles.IntelligenceConfig, request: RequestChoice | None) -> Selection:
    """The setup for one turn: explicit request fields over the requested profile."""
    request = request or RequestChoice()
    chosen = config.profile(request.profile) if request.profile else None
    base = chosen.setup if chosen else profiles.Setup(model=None, effort=None)
    return Selection(
        profile=chosen.id if chosen else None,
        model=request.model or base.model,
        effort=request.effort or base.effort,
        source=None if request.empty else "request",
    )


async def turn_selection(stream_info: dict) -> dict[str, Any] | None:
    """The ``intelligence`` keyword for the adapter, or ``None`` to pass nothing.

    ``None`` when the agent has no profiles or the turn sets no flag, so an
    agent (or a request) that does not use profiles runs exactly as before.
    Never raises: a failure here falls back to no flags, never to no reply.
    """
    try:
        config = profiles.load(stream_info.get("agent_name") or "")
        if config is None:
            return None
        selection = select(config, stream_info.get("intelligence_request"))
    except Exception:  # noqa: BLE001 - the reply matters more than the setup
        log.exception("intelligence: could not choose a setup; running without flags")
        return None
    if selection.model is None and selection.effort is None:
        return None
    log.info(
        "intelligence: session %s runs with profile=%s model=%s effort=%s (%s)",
        stream_info.get("our_session_id"), selection.profile, selection.model,
        selection.effort, selection.source,
    )
    return selection.as_kwargs()
