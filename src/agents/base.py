"""Common helpers shared by all V2 agents.

Every agent follows the same skeleton:

    def run(state) -> dict:
        log = start_log("AgentName")
        try:
            ... do work ...
            return {<my section>: ..., "logs": [finish_log(log, "success")]}
        except Exception as e:
            log_failure(log, e)
            return {"logs": [log], "errors": [f"AgentName: {e}"]}

Centralizing this pattern keeps each agent file short and consistent.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from src.schemas import AgentLog

logger = logging.getLogger(__name__)


def start_log(agent_name: str) -> AgentLog:
    """Open a fresh `AgentLog` in `running` state."""
    return AgentLog(
        agent_name=agent_name,
        started_at=datetime.now(),
        status="running",
    )


def finish_log(
    log: AgentLog,
    status: str = "success",
    *,
    tokens_used: int = 0,
    api_calls: int = 0,
    error: Optional[str] = None,
) -> AgentLog:
    """Close an AgentLog and return it (caller appends it to `state["logs"]`)."""
    log.completed_at = datetime.now()
    log.status = status  # type: ignore[assignment]
    log.tokens_used = tokens_used
    log.api_calls = api_calls
    log.error = error
    return log
