"""Summarizer agent: produces a structured ArticleSummary per top article.

This is the most cost-sensitive agent: 1 LLM call per article × top_n articles.
We use Claude Haiku (cheapest Claude) by default, with prompt caching on the
system prompt — every call after the first hits cached input pricing.

Inputs:
    - topic, top_articles
Outputs:
    - summaries: list[ArticleSummary]
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.llm.base import LLMClient, LLMError, LLMResponse, Message
from src.llm.factory import get_llm_for_task
from src.schemas import Article, ArticleStatus, ArticleSummary, NarrativeSummary

logger = logging.getLogger(__name__)

AGENT_NAME = "Summarizer"


SYSTEM_PROMPT = "You are a helpful assistant. Summarize the scientific article."

SYSTEM_PROMPT_NARRATIVE = """\
You are a scientific paper summarizer. Write a concise narrative summary of \
the scientific article in 150-250 words. Cover the research problem, \
methodology, key quantitative results (include specific numbers, statistics, \
or percentages when present), and the authors' main conclusions. Write in \
past tense throughout. Use a single flowing paragraph — no bullet points, \
no headers, no markdown. Style your output like a scientific abstract.\
"""


def run(state: WatchState, llm: Optional[LLMClient] = None) -> dict:
    log = start_log(AGENT_NAME)
    top_articles: list[Article] = state.get("top_articles", [])
    narrative: bool = state.get("narrative_mode", False)

    if not top_articles:
        logger.warning("[%s] no top_articles to summarize", AGENT_NAME)
        empty_key = "narrative_summaries" if narrative else "summaries"
        return {empty_key: [], "logs": [finish_log(log, "success")]}

    llm = llm or get_llm_for_task("summarize")
    logger.info("[%s] mode=%s", AGENT_NAME, "narrative" if narrative else "structured")

    summaries: list = []
    total_tokens = 0
    api_calls = 0
    failures: list[str] = []

    for art in top_articles:
        try:
            if narrative:
                summary, resp = _summarize_one_narrative(llm, art)
            else:
                summary, resp = _summarize_one(llm, art)
            summaries.append(summary)
            total_tokens += resp.input_tokens + resp.output_tokens
            api_calls += 1
            art.status = ArticleStatus.SUMMARIZED
        except LLMError as exc:
            logger.warning("[%s] failed for article %s: %s", AGENT_NAME, art.id, exc)
            failures.append(f"{art.id}: {exc}")

    output_key = "narrative_summaries" if narrative else "summaries"
    out: dict = {
        output_key: summaries,
        "logs": [
            finish_log(
                log,
                "success" if summaries else "failed",
                tokens_used=total_tokens,
                api_calls=api_calls,
                error="; ".join(failures) if failures else None,
            )
        ],
    }
    if failures:
        out["errors"] = [f"{AGENT_NAME}: {len(failures)} article(s) failed"]
    return out


def _summarize_one(
    llm: LLMClient, article: Article
) -> tuple[ArticleSummary, LLMResponse]:
    """One LLM call for one article — structured mode (6 fields)."""
    user_text = (
        f"Title: {article.title}\n\n"
        f"Abstract:\n{article.abstract or '[no abstract available]'}\n"
    )
    messages = [Message(role="user", content=user_text)]

    parsed, resp = llm.chat_structured(
        system=SYSTEM_PROMPT,
        messages=messages,
        schema=ArticleSummary,
        temperature=0.0,
        max_tokens=1024,
    )
    # ID is deterministic from the source article, not LLM output.
    parsed.article_id = article.id
    return parsed, resp


def _summarize_one_narrative(
    llm: LLMClient, article: Article
) -> tuple[NarrativeSummary, LLMResponse]:
    """One LLM call for one article — narrative mode (prose paragraph)."""
    user_text = (
        f"Title: {article.title}\n\n"
        f"Abstract:\n{article.abstract or '[no abstract available]'}\n"
    )
    messages = [Message(role="user", content=user_text)]

    resp = llm.chat(
        system=SYSTEM_PROMPT_NARRATIVE,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    summary = NarrativeSummary(
        article_id=article.id,
        text=resp.content.strip(),
    )
    return summary, resp
