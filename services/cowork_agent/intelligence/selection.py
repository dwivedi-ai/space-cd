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


def select(
    config: profiles.IntelligenceConfig,
    request: RequestChoice | None,
    *,
    session: dict[str, Any] | None = None,
    decided: str | None = None,
    use_default: bool = False,
) -> Selection:
    """The setup for one turn.

    Explicit request fields always win, field by field. Under them, the first
    of: the requested profile; the setup the session started with
    (``session``, so a resumed turn never switches); the profile the decision
    model picked (``decided``); the default setup (``use_default``). With none
    of these, nothing is set and no flag is passed.
    """
    request = request or RequestChoice()
    requested = config.profile(request.profile) if request.profile else None
    picked = config.profile(decided) if decided else None
    if requested is not None:
        profile, base, source = requested.id, requested.setup, "request"
    elif session is not None:
        profile = session.get("profile")
        base = profiles.Setup(model=session.get("model"), effort=session.get("effort"))
        source = "session"
    elif picked is not None:
        profile, base, source = picked.id, picked.setup, "sage"
    elif use_default:
        profile, base, source = None, config.default, "default"
    else:
        profile, base = None, profiles.Setup(model=None, effort=None)
        source = None if request.empty else "request"
    return Selection(
        profile=profile,
        model=request.model or base.model,
        effort=request.effort or base.effort,
        source=source,
    )
