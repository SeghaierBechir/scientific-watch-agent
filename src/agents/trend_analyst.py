"""TrendAnalyst agent: identifies trends, gaps, and future perspectives.

Single LLM call. Consumes the Synthesis + the per-article summaries (for
evidence article IDs).

Inputs:
    - topic, summaries, synthesis
Outputs:
    - trend_analysis: TrendAnalysis | None
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.llm.base import LLMClient, LLMError, LLMStructuredOutputError, Message
from src.llm.factory import get_llm_for_task
from src.schemas import ArticleSummary, Synthesis, TrendAnalysis

logger = logging.getLogger(__name__)

AGENT_NAME = "TrendAnalyst"


SYSTEM_PROMPT = """You are a research strategist identifying scientific trends.

Inputs you receive:
    - A synthesis of a research field.
    - The list of structured article summaries that fed that synthesis.

Produce a TrendAnalysis with:
    - trends: 3 to 6 named trends, each with:
        * name: short label
        * description: 1-3 sentences
        * evidence_article_ids: ids of summaries that support this trend
        * maturity: one of "emerging", "established", "declining"
    - gaps: 2 to 5 research gaps, each with description, importance
        ("low" | "medium" | "high"), and 1-3 suggested directions.
    - future_perspectives: 3 to 5 forward-looking research directions
      worded as actionable propositions.

Ground every trend in concrete article IDs. Do not invent IDs.
"""


def run(state: WatchState, llm: Optional[LLMClient] = None) -> dict:
    log = start_log(AGENT_NAME)
    topic = state["topic"]
    narrative: bool = state.get("narrative_mode", False)
    summaries = (
        state.get("narrative_summaries", [])
        if narrative
        else state.get("summaries", [])
    )
    synthesis: Optional[Synthesis] = state.get("synthesis")

    if not summaries or synthesis is None:
        logger.warning("[%s] missing summaries or synthesis", AGENT_NAME)
        return {"trend_analysis": None, "logs": [finish_log(log, "success")]}

    llm = llm or get_llm_for_task("trend_analysis")

    payload = {
        "topic": topic,
        "synthesis": synthesis.model_dump(mode="json"),
        "summaries": [s.model_dump(mode="json") for s in summaries],
    }
    messages = [
        Message(
            role="user",
            content=(
                f"Topic: {topic}\n\n"
                f"Synthesis + summaries (JSON):\n{json.dumps(payload, indent=2)}"
            ),
        )
    ]

    total_tokens = 0
    total_calls = 0

    def _call(msgs: list[Message]) -> tuple[TrendAnalysis, int]:
        """Single LLM call; returns (result, tokens_used)."""
        nonlocal total_calls
        total_calls += 1
        ta, resp = llm.chat_structured(
            system=SYSTEM_PROMPT,
            messages=msgs,
            schema=TrendAnalysis,
            temperature=0.3,
            max_tokens=2048,
        )
        return ta, resp.input_tokens + resp.output_tokens, resp.cost_usd

    try:
        trends, tokens, cost = _call(messages)
        total_tokens += tokens

        # ── Retry if key lists are empty (LLM omitted a required section) ──
        # The schema now accepts empty lists (default_factory=list), so
        # Pydantic won't raise — but we detect the omission here and ask
        # the LLM to complete the answer before accepting the result.
        if not trends.future_perspectives:
            logger.warning(
                "[%s] future_perspectives missing — retrying with explicit reminder",
                AGENT_NAME,
            )
            retry_messages = messages + [
                Message(
                    role="assistant",
                    content=str(trends.model_dump(mode="json")),
                ),
                Message(
                    role="user",
                    content=(
                        "Your response is missing the 'future_perspectives' field. "
                        "Please provide a complete TrendAnalysis that includes "
                        "future_perspectives: a list of 3-5 forward-looking research "
                        "directions worded as actionable propositions."
                    ),
                ),
            ]
            try:
                trends, tokens2, _ = _call(retry_messages)
                total_tokens += tokens2
            except LLMStructuredOutputError as retry_exc:
                # Retry also failed — keep the first (partial) result.
                logger.warning(
                    "[%s] retry also failed (%s) — using partial result",
                    AGENT_NAME, retry_exc,
                )

        logger.info(
            "[%s] identified %d trends, %d gaps, %d perspectives (%d tokens)",
            AGENT_NAME,
            len(trends.trends),
            len(trends.gaps),
            len(trends.future_perspectives),
            total_tokens,
        )
        return {
            "trend_analysis": trends,
            "logs": [
                finish_log(
                    log,
                    "success",
                    tokens_used=total_tokens,
                    api_calls=total_calls,
                )
            ],
        }

    except LLMError as exc:
        logger.exception("[%s] failed", AGENT_NAME)
        return {
            "trend_analysis": None,
            "logs": [finish_log(log, "failed", error=str(exc))],
            "errors": [f"{AGENT_NAME}: {exc}"],
        }
