"""Searcher agent: fetches raw articles from OpenAlex.

V2 update: now reads `expanded_queries` from state (set by QueryExpander).
If the list has N queries, it distributes the n_raw budget across them,
then deduplicates results by article ID.

Fallback: if expanded_queries is empty or missing, searches with topic only.

Inputs  (from state): expanded_queries (or topic), config
Outputs (to state)  : raw_articles
"""

from __future__ import annotations

import logging

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.schemas import Article
from src.sources.openalex import OpenAlexClient, OpenAlexError

logger = logging.getLogger(__name__)

AGENT_NAME = "Searcher"

# Minimum articles to fetch per query regardless of budget split.
MIN_PER_QUERY = 10


def run(state: WatchState) -> dict:
    log = start_log(AGENT_NAME)
    cfg = state.get("config", {})
    topic: str = state["topic"]
    n_raw: int = cfg.get("n_raw", 30)
    from_year: int = cfg.get("from_year", 2020)
    require_abstract: bool = cfg.get("require_abstract", True)

    # Use expanded queries if available, otherwise fall back to topic.
    queries: list[str] = state.get("expanded_queries") or [topic]
    per_query = max(MIN_PER_QUERY, n_raw // len(queries))

    client = OpenAlexClient()
    all_articles, failed_queries = _search_all(
        client, queries, per_query, from_year, require_abstract
    )

    all_failed = len(failed_queries) == len(queries)
    status = "failed" if all_failed else "success"
    error_msg = f"All {len(queries)} queries failed" if all_failed else None

    logger.info(
        "[%s] fetched %d unique articles across %d queries (%d failed)",
        AGENT_NAME, len(all_articles), len(queries), len(failed_queries),
    )

    out: dict = {
        "raw_articles": all_articles,
        "logs": [finish_log(log, status, api_calls=len(queries), error=error_msg)],
    }
    if all_failed:
        out["errors"] = [f"{AGENT_NAME}: {error_msg}"]
    return out


def _search_all(
    client: OpenAlexClient,
    queries: list[str],
    per_query: int,
    from_year: int,
    require_abstract: bool,
) -> tuple[list[Article], list[str]]:
    """Search all queries and return (deduplicated articles, failed query list).

    Deduplication is by article ID. When the same paper appears in results
    for multiple queries, only the first occurrence is kept.
    Articles are ordered: first query's results come first.
    A failed query is logged as warning but never aborts the others.
    """
    seen_ids: set[str] = set()
    combined: list[Article] = []
    failed: list[str] = []

    for query in queries:
        try:
            results = client.search(
                query=query,
                n_results=per_query,
                from_year=from_year,
                require_abstract=require_abstract,
            )
            new_articles = [a for a in results if a.id not in seen_ids]
            seen_ids.update(a.id for a in new_articles)
            combined.extend(new_articles)
            logger.debug(
                "[%s] query=%r -> %d results, %d new",
                AGENT_NAME, query, len(results), len(new_articles),
            )
        except OpenAlexError as exc:
            logger.warning("[%s] query=%r failed: %s", AGENT_NAME, query, exc)
            failed.append(query)

    return combined, failed
