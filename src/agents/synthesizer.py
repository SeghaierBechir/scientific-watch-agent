"""Synthesizer agent: builds a global Synthesis from all article summaries.

Phase 6 update — Reflexion support:
    If `critic_feedbacks` contains at least one feedback with needs_revision=True,
    the Synthesizer receives its previous synthesis + the Critic's issues and
    suggestions, and produces a revised version.

Inputs  (from state): topic, summaries, synthesis (previous, may be None),
                      critic_feedbacks (may be empty)
Outputs (to state)  : synthesis, synthesis_iteration (incremented)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.llm.base import LLMClient, LLMError, Message
from src.llm.factory import get_llm_for_task
from src.schemas import ArticleSummary, CriticFeedback, Synthesis

logger = logging.getLogger(__name__)

AGENT_NAME = "Synthesizer"


SYSTEM_PROMPT = """You are a senior researcher writing a literature synthesis.

You will receive a JSON payload containing:
    - topic: the research topic
    - summaries: structured summaries of scientific articles

Produce a global synthesis with:
    - overview: 4 to 8 sentences describing the field's current state.
    - main_approaches: 3 to 6 distinct families of methods observed.
    - common_datasets: datasets mentioned in 2+ articles (may be empty).
    - key_findings: 3 to 7 concrete findings supported by multiple articles.
    - article_count: must equal the number of summaries provided.

Ground every claim in the summaries. Do not introduce outside knowledge.
Prefer specific, quantitative phrasing over vague generalities.
"""

REVISION_SYSTEM_PROMPT = """You are a senior researcher revising a literature synthesis.

You produced a first synthesis that was reviewed by a Critic. Your task is to
address the Critic's issues and suggestions to produce an improved version.

Rules:
    - Fix every issue listed by the Critic.
    - Apply every suggestion where applicable.
    - Keep what was already correct in the previous synthesis.
    - Do not invent facts not present in the summaries.
    - Maintain the same output structure (overview, main_approaches, etc.).
"""


def run(state: WatchState, llm: Optional[LLMClient] = None) -> dict:
    log = start_log(AGENT_NAME)
    topic: str = state["topic"]
    narrative: bool = state.get("narrative_mode", False)
    summaries = (
        state.get("narrative_summaries", [])
        if narrative
        else state.get("summaries", [])
    )
    previous_synthesis: Optional[Synthesis] = state.get("synthesis")
    feedbacks: list[CriticFeedback] = state.get("critic_feedbacks", [])
    iteration: int = state.get("synthesis_iteration", 0)

    if not summaries:
        logger.warning("[%s] no summaries to synthesize", AGENT_NAME)
        return {
            "synthesis": None,
            "synthesis_iteration": iteration,
            "logs": [finish_log(log, "success")],
        }

    llm = llm or get_llm_for_task("synthesize")

    # Decide if this is a first run or a revision.
    last_feedback = feedbacks[-1] if feedbacks else None
    is_revision = (
        last_feedback is not None
        and last_feedback.needs_revision
        and previous_synthesis is not None
    )

    messages = _build_messages(topic, summaries, previous_synthesis, last_feedback, is_revision)
    system = REVISION_SYSTEM_PROMPT if is_revision else SYSTEM_PROMPT

    try:
        synthesis, resp = llm.chat_structured(
            system=system,
            messages=messages,
            schema=Synthesis,
            temperature=0.2,
            max_tokens=2048,
        )
        synthesis.topic = topic
        synthesis.article_count = len(summaries)
        new_iteration = iteration + 1

        logger.info(
            "[%s] iteration=%d %s (%d tokens, $%.4f)",
            AGENT_NAME, new_iteration,
            "REVISION" if is_revision else "first run",
            resp.input_tokens + resp.output_tokens,
            resp.cost_usd,
        )
        return {
            "synthesis": synthesis,
            "synthesis_iteration": new_iteration,
            "logs": [
                finish_log(
                    log, "success",
                    tokens_used=resp.input_tokens + resp.output_tokens,
                    api_calls=1,
                )
            ],
        }
    except LLMError as exc:
        logger.exception("[%s] failed", AGENT_NAME)
        return {
            "synthesis": previous_synthesis,   # keep previous if revision failed
            "synthesis_iteration": iteration,
            "logs": [finish_log(log, "failed", error=str(exc))],
            "errors": [f"{AGENT_NAME}: {exc}"],
        }


def _build_messages(
    topic: str,
    summaries: list[ArticleSummary],
    previous: Optional[Synthesis],
    feedback: Optional[CriticFeedback],
    is_revision: bool,
) -> list[Message]:
    """Construct the user message(s) for the LLM call."""
    payload = {
        "topic": topic,
        "n_articles": len(summaries),
        "summaries": [s.model_dump(mode="json") for s in summaries],
    }

    if is_revision and previous is not None and feedback is not None:
        content = (
            f"Topic: {topic}\n\n"
            f"Article summaries (JSON):\n{json.dumps(payload, indent=2)}\n\n"
            f"--- PREVIOUS SYNTHESIS (iteration {feedback.iteration}) ---\n"
            f"{previous.model_dump_json(indent=2)}\n\n"
            f"--- CRITIC FEEDBACK ---\n"
            f"Overall quality: {feedback.overall_quality}\n"
            f"Issues:\n" + "\n".join(f"  - {i}" for i in feedback.issues) + "\n"
            f"Suggestions:\n" + "\n".join(f"  - {s}" for s in feedback.suggestions)
        )
    else:
        content = (
            f"Topic: {topic}\n\n"
            f"Article summaries (JSON):\n{json.dumps(payload, indent=2)}"
        )

    return [Message(role="user", content=content)]
