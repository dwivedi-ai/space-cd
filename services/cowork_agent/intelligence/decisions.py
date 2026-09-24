"""Decide a new session's profile with Levanto Sage, log it, and (in ``on``) apply it.

:func:`start` is called once per new session, from the chat route, and
returns at once: the decision runs as a task beside the agent.

- ``shadow``: nothing waits for the decision. The session runs with what the
  request itself chose, if anything, and the decision is only logged.
- ``on``: the stream waits for the decision for at most :data:`ON_WAIT_S`.
  An answer in time is applied; a late one, ``unknown``, a ``null`` or a
  failure gets the default setup. The late answer is still logged
  (``sage_late``).

Either way, an explicit profile, model or effort in the request wins, and a
session keeps the setup it started with: :func:`turn_selection` re-applies it
on every resumed turn (remembered in memory, and found again in the decision
log after a restart), so the model never changes mid-session.

Every new session's outcome is one line in the decision log (``decision_log``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from services.cowork_agent.engine import sessions_io
from services.cowork_agent.intelligence import classify, decision_log, mode, profiles, selection

log = logging.getLogger(__name__)

#: How long a new session's first turn waits for the decision in ``on`` mode.
ON_WAIT_S = 2.0
#: The decision was made, but after the first turn had to start.
SAGE_LATE = "sage_late"

# Strong references: the event loop keeps only weak ones to running tasks.
_tasks: set[asyncio.Task] = set()

# session_id -> the setup it started with (Selection.as_kwargs()), newest last.
_session_setups: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SESSION_SETUPS_MAX = 2048


@dataclass
class PendingDecision:
    """A new session's decision in progress.

    ``setup`` resolves to the :class:`selection.Selection` the first turn runs
    with, by the ``on`` deadline at the latest (``None`` if deciding failed);
    ``task`` finishes once the decision is logged.
    """

    task: asyncio.Task
    setup: asyncio.Future


def start(
    *,
    agent_name: str,
    text: str,
    session_id: str | None,
    project: str | None,
    request: selection.RequestChoice | None,
) -> PendingDecision | None:
    """Start deciding a new session's profile. ``None`` when there is nothing to do:
    the switch is off, or the agent has no profiles."""
    current = mode.mode()
    if current == mode.OFF:
        return None
    config = profiles.load(agent_name)
    if config is None:
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no running loop: nothing to decide on
        return None
    setup = loop.create_future()
    task = loop.create_task(
        _decide(config, current, agent_name, text, session_id, project, request, setup),
        name=f"intelligence-decision-{session_id}",
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return PendingDecision(task=task, setup=setup)


def decided_setup(config: profiles.IntelligenceConfig, decision: classify.Decision) -> profiles.Setup:
    """The setup a decision stands for: its profile's, or the default."""
    chosen = config.profile(decision.profile) if decision.profile else None
    return chosen.setup if chosen else config.default


def _resolve(future: asyncio.Future | None, value: Any) -> None:
    if future is not None and not future.done():
        future.set_result(value)


def remember(session_id: str | None, applied: dict[str, Any]) -> None:
    if not session_id:
        return
    _session_setups[session_id] = applied
    _session_setups.move_to_end(session_id)
    while len(_session_setups) > _SESSION_SETUPS_MAX:
        _session_setups.popitem(last=False)


async def _decide(
    config: profiles.IntelligenceConfig,
    current_mode: str,
    agent_name: str,
    text: str,
    session_id: str | None,
    project: str | None,
    request: selection.RequestChoice | None,
    setup: asyncio.Future | None = None,
) -> classify.Decision | None:
    started = time.perf_counter()
    try:
        content = classify.prepare_content(text)
        deciding = asyncio.ensure_future(classify.classify(config, content, timeout=mode.sage_timeout_s()))
        on_time = True
        if current_mode == mode.ON:
            try:
                await asyncio.wait_for(asyncio.shield(deciding), timeout=ON_WAIT_S)
            except asyncio.TimeoutError:
                on_time = False
            picked = deciding.result().profile if on_time else None
            applied = selection.select(config, request, decided=picked, use_default=True)
        else:
            applied = selection.select(config, request)
        remember(session_id, applied.as_kwargs())
        _resolve(setup, applied)

        decision = await deciding
        setup_decided = decided_setup(config, decision)
        await asyncio.to_thread(
            decision_log.record,
            project,
            session_id=session_id,
            runtime=agent_name,
            mode=current_mode,
            request=decision_log.request_record(content),
            profiles_sha256=config.sha256,
            sage=decision.sage,
            decision={"profile": decision.profile,
                      "reason": decision.reason if on_time else SAGE_LATE,
                      "model": setup_decided.model, "effort": setup_decided.effort},
            applied=applied.as_kwargs(),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return decision
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a failed decision is only a missing log line
        log.exception("intelligence: deciding session %s failed", session_id)
        return None
    finally:
        _resolve(setup, None)


def _session_project(session_id: str) -> tuple[bool, str | None]:
    """``(found, project)`` for a session, from the session indexes. Blocking I/O."""
    for project, _directory, index in sessions_io.iter_project_session_indexes():
        if any(isinstance(row, dict) and row.get("sessionId") == session_id for row in index.values()):
            return True, project or None
    return False, None


def _setup_from_log(session_id: str) -> dict[str, Any] | None:
    found, project = _session_project(session_id)
    if not found:
        return None
    line = decision_log.find(project, session_id)
    applied = line.get("applied") if line else None
    return applied if isinstance(applied, dict) else None


async def session_setup(session_id: str | None) -> dict[str, Any] | None:
    """The setup a session started with, or ``None`` if it was never decided."""
    if not session_id:
        return None
    cached = _session_setups.get(session_id)
    if cached is not None:
        return cached
    applied = await asyncio.to_thread(_setup_from_log, session_id)
    if applied is not None:
        remember(session_id, applied)
    return applied


async def _first_turn_setup(pending: PendingDecision | None) -> selection.Selection | None:
    if pending is None:
        return None
    try:
        # The task resolves this by its own deadline; the margin only guards a stuck task.
        return await asyncio.wait_for(asyncio.shield(pending.setup), timeout=ON_WAIT_S + 1.0)
    except Exception:  # noqa: BLE001 - no setup in time means the default
        return None


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
        request = stream_info.get("intelligence_request")
        if mode.mode() != mode.ON:
            chosen = selection.select(config, request)
        elif stream_info.get("is_new_session"):
            chosen = (await _first_turn_setup(stream_info.get("intelligence_decision"))
                      or selection.select(config, request, use_default=True))
        else:
            # A resumed turn keeps the session's first setup; a session that was
            # never decided keeps running as it started, with no flags.
            session = await session_setup(stream_info.get("our_session_id"))
            chosen = selection.select(config, request, session=session)
    except Exception:  # noqa: BLE001 - the reply matters more than the setup
        log.exception("intelligence: could not choose a setup; running without flags")
        return None
    if chosen.model is None and chosen.effort is None:
        return None
    log.info(
        "intelligence: session %s runs with profile=%s model=%s effort=%s (%s)",
        stream_info.get("our_session_id"), chosen.profile, chosen.model, chosen.effort, chosen.source,
    )
    return chosen.as_kwargs()
