"""Decide a new session's profile with Levanto Sage, in the background, and log it.

:func:`start` is called once per new session, from the chat route, and
returns at once: the decision runs as a task beside the agent. In ``shadow``
mode nothing waits for it, so the reply is never delayed. Every outcome,
including a failed or missing Sage answer, is one line in the decision log
(``decision_log``). The default setup is what a session gets whenever Sage does
not make a decision.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from services.cowork_agent.intelligence import classify, decision_log, mode, profiles, selection

log = logging.getLogger(__name__)

# Strong references: the event loop keeps only weak ones to running tasks.
_tasks: set[asyncio.Task] = set()


def start(
    *,
    agent_name: str,
    text: str,
    session_id: str | None,
    project: str | None,
    request: selection.RequestChoice | None,
) -> asyncio.Task | None:
    """Start deciding a new session's profile. ``None`` when there is nothing to do:
    the switch is off, or the agent has no profiles."""
    current = mode.mode()
    if current == mode.OFF:
        return None
    config = profiles.load(agent_name)
    if config is None:
        return None
    try:
        task = asyncio.get_running_loop().create_task(
            _decide(config, current, agent_name, text, session_id, project, request),
            name=f"intelligence-decision-{session_id}",
        )
    except RuntimeError:  # no running loop: nothing to decide on
        return None
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


def decided_setup(config: profiles.IntelligenceConfig, decision: classify.Decision) -> profiles.Setup:
    """The setup a decision stands for: its profile's, or the default."""
    chosen = config.profile(decision.profile) if decision.profile else None
    return chosen.setup if chosen else config.default


async def _decide(
    config: profiles.IntelligenceConfig,
    current_mode: str,
    agent_name: str,
    text: str,
    session_id: str | None,
    project: str | None,
    request: selection.RequestChoice | None,
) -> classify.Decision | None:
    started = time.perf_counter()
    try:
        content = classify.prepare_content(text)
        decision = await classify.classify(config, content, timeout=mode.sage_timeout_s())
        setup = decided_setup(config, decision)
        # In shadow the session runs with what the request itself chose, if anything.
        applied: dict[str, Any] = selection.select(config, request).as_kwargs()
        await asyncio.to_thread(
            decision_log.record,
            project,
            session_id=session_id,
            runtime=agent_name,
            mode=current_mode,
            request=decision_log.request_record(content),
            profiles_sha256=config.sha256,
            sage=decision.sage,
            decision={"profile": decision.profile, "reason": decision.reason,
                      "model": setup.model, "effort": setup.effort},
            applied=applied,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return decision
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a failed decision is only a missing log line
        log.exception("intelligence: deciding session %s failed", session_id)
        return None
