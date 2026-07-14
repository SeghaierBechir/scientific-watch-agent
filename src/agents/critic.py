"""Critic agent — Reflexion pattern (Phase 6).

The Critic reads the current Synthesis and evaluates its quality on 4 axes:
    1. Fidelity    — every claim is grounded in the summaries (no hallucination)
    2. Completeness — important findings from summaries are not omitted
    3. Specificity  — concrete numbers/methods cited, not vague generalities
    4. Consistency  — no internal contradictions

If the quality is below the configured threshold (REFLEXION_MIN_QUALITY),
it sets `needs_revision=True` and provides actionable issues + suggestions.
The Synthesizer will then receive this feedback and revise.

Inputs  (from state): synthesis, summaries, synthesis_iteration
Outputs (to state)  : critic_feedbacks (appended via operator.add)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.config import MAX_REFLEXION_ITERATIONS, REFLEXION_MIN_QUALITY
from src.llm.base import LLMClient, LLMError, Message
from src.llm.factory import get_llm_for_task
from src.schemas import ArticleSummary, CriticFeedback, Synthesis

logger = logging.getLogger(__name__)

AGENT_NAME = "Critic"

# Quality levels in ascending order — used for threshold comparison.
_QUALITY_ORDER = ["poor", "acceptable", "good", "excellent"]

SYSTEM_PROMPT = """You are a rigorous scientific peer-reviewer evaluating a literature synthesis.

You receive:
    - The synthesis to evaluate.
    - The article summaries it was built from.

Evaluate the synthesis on 4 criteria:
    1. Fidelity      : every claim is traceable to a summary (no invented facts).
    2. Completeness  : important findings from summaries are represented.
    3. Specificity   : concrete metrics, method names, dataset names are cited.
    4. Consistency   : no contradictions between statements.

Set overall_quality to one of: "poor", "acceptable", "good", "excellent".

Set needs_revision to True if overall_quality is "poor" or "acceptable".
Set needs_revision to False if overall_quality is "good" or "excellent".

For issues: list only real problems, one per item (max 5).
For suggestions: provide one concrete fix per issue (same order, max 5).

If the synthesis is already good, set issues=[] and suggestions=[].
"""


def run(state: WatchState, llm: Optional[LLMClient] = None) -> dict:
    log = start_log(AGENT_NAME)
    synthesis: Optional[Synthesis] = state.get("synthesis")
    summaries: list[ArticleSummary] = state.get("summaries", [])
    iteration: int = state.get("synthesis_iteration", 1)

    if synthesis is None:
        logger.warning("[%s] no synthesis to critique", AGENT_NAME)
        return {
            "critic_feedbacks": [],
            "logs": [finish_log(log, "success")],
        }

    # If we have already reached the iteration limit, approve without calling LLM.
    if iteration >= MAX_REFLEXION_ITERATIONS:
        logger.info(
            "[%s] max iterations (%d) reached — approving synthesis as-is",
            AGENT_NAME, MAX_REFLEXION_ITERATIONS,
        )
        forced_approval = CriticFeedback(
            target="synthesis",
            iteration=iteration,
            overall_quality="acceptable",
            issues=[],
            suggestions=[],
            needs_revision=False,   # force stop
        )
        return {
            "critic_feedbacks": [forced_approval],
            "logs": [finish_log(log, "success")],
        }

    llm = llm or get_llm_for_task("critic")

    payload = {
        "synthesis": synthesis.model_dump(mode="json"),
        "summaries": [s.model_dump(mode="json") for s in summaries],
    }
    messages = [
        Message(
            role="user",
            content=(
                f"Iteration: {iteration}\n\n"
                f"Synthesis + summaries (JSON):\n{json.dumps(payload, indent=2)}"
            ),
        )
    ]

    try:
        feedback, resp = llm.chat_structured(
            system=SYSTEM_PROMPT,
            messages=messages,
            schema=CriticFeedback,
            temperature=0.1,      # low temperature = deterministic evaluation
            max_tokens=1024,
        )
        # Stamp the iteration number — LLM output may not have it right.
        feedback.iteration = iteration
        feedback.target = "synthesis"

        # Override needs_revision based on configured quality threshold.
        feedback.needs_revision = _below_threshold(
            feedback.overall_quality, REFLEXION_MIN_QUALITY
        )

        logger.info(
            "[%s] iteration=%d quality=%s needs_revision=%s",
            AGENT_NAME, iteration, feedback.overall_quality, feedback.needs_revision,
        )
        return {
            "critic_feedbacks": [feedback],
            "logs": [
                finish_log(
                    log, "success",
                    tokens_used=resp.input_tokens + resp.output_tokens,
                    api_calls=1,
                )
            ],
        }
    except LLMError as exc:
        logger.exception("[%s] LLM call failed", AGENT_NAME)
        # On failure: approve synthesis so the pipeline is not blocked.
        fallback = CriticFeedback(
            target="synthesis",
            iteration=iteration,
            overall_quality="acceptable",
            issues=[],
            suggestions=[],
            needs_revision=False,
        )
        return {
            "critic_feedbacks": [fallback],
            "logs": [finish_log(log, "failed", error=str(exc))],
            "errors": [f"{AGENT_NAME}: {exc} — synthesis approved as fallback"],
        }


def _below_threshold(quality: str, threshold: str) -> bool:
    """Return True if quality is strictly below threshold."""
    try:
        return _QUALITY_ORDER.index(quality) < _QUALITY_ORDER.index(threshold)
    except ValueError:
        return False
