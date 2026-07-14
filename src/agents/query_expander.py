"""QueryExpander agent — ReAct pattern (Hybrid Phase 5+6).

WHY ReAct here?
---------------
The old single-shot expander generated N queries blindly, without knowing
whether those queries actually return relevant articles on OpenAlex. With
ReAct, the agent can *observe* the results of each probe and adjust its
next query accordingly:

    THINK  →  ACT (search OpenAlex)  →  OBSERVE (article count + concepts)
      ↑                                          |
      └──────────────── feedback ────────────────┘
                    (until STOP or max iter)

Key difference from Reflexion (used on Synthesizer):
    • Reflexion loops on OUTPUT quality  (is the synthesis good enough?)
    • ReAct    loops on EXPLORATION      (have I covered enough angles?)

How the multi-turn context works:
    Each LLM call receives ONE user message that contains the full history
    of all previous (thought, action, observation) triples.  This avoids
    the tool-use message format problem with multi-turn APIs and keeps the
    prompt compatible with any LLMClient (Claude or OpenAI).

Inputs  (from state): topic, config.n_raw
Outputs (to state)  :
    expanded_queries  — queries that returned ≥1 article (used by Searcher)
    react_steps       — full loop history (for demo + PDF report)

Cost estimate: MAX_REACT_ITERATIONS × (1 LLM call + 1 OpenAlex probe)
               ≈ 4 × ($0.0001 + free) ≈ $0.0004 total for the expander.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.base import finish_log, start_log
from src.agents.state import WatchState
from src.config import MAX_REACT_ITERATIONS, REACT_PROBE_N
from src.llm.base import LLMClient, LLMError, Message
from src.llm.factory import get_llm_for_task
from src.schemas import Article, ReActThoughtAction
from src.sources.openalex import OpenAlexClient

logger = logging.getLogger(__name__)

AGENT_NAME = "QueryExpander"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a scientific literature search expert working in a
ReAct (Reason + Act) loop to find diverse queries for OpenAlex.

Goal: identify 3-4 search queries that together cover a research topic from
different angles (synonyms, sub-tasks, methods, datasets, applications).

Each step you output ONE action:
  action="search", search_query="<2-5 words>"   — probe OpenAlex
  action="stop"                                  — when coverage is sufficient

Rules:
  • Iteration 1: ALWAYS search with the EXACT original topic first.
  • Observe returned concepts and titles to understand what angles exist.
  • If a query returns 0 articles, note it and try a different angle.
  • Stop when you have 3-4 productive queries covering distinct aspects.
  • Keep queries short (2-5 words), no year filters, no venue names.
  • Do NOT repeat a query already tried.

Good example for "fake news detection":
  Iter 1 → search "fake news detection"       (original topic)
  Iter 2 → search "misinformation NLP"        (synonym)
  Iter 3 → search "graph neural network rumor detection"  (method angle)
  Iter 4 → stop   (3 productive queries found, good coverage)
"""


# ── Prompt builder (full history in one user message) ─────────────────────────

def _build_prompt(topic: str, n_raw: int, steps: list[dict]) -> str:
    """Rebuild the full ReAct context as a single user message.

    Pédagogique — why one message instead of true multi-turn?
        The Anthropic API uses tool_use blocks for structured outputs, which
        creates strict constraints on the assistant/user turn format.
        Embedding all history in one user message is fully compatible with
        any LLMClient (Claude, OpenAI) and produces identical reasoning.
    """
    lines = [
        f"Research topic: {topic}",
        f"Target: ~{n_raw} diverse articles total across all queries.",
        "",
    ]

    if not steps:
        lines += [
            "No searches performed yet.",
            "Your first action MUST be: action='search' with the original topic.",
        ]
    else:
        n_productive = sum(1 for s in steps if s["action"] == "search"
                          and s.get("n_found", 0) > 0)
        lines.append(f"ReAct history ({len(steps)} step(s), {n_productive} productive):")
        for s in steps:
            lines.append(f"\n--- Iteration {s['iteration']} ---")
            lines.append(f"Thought : {s['thought']}")
            if s["action"] == "stop":
                lines.append("Action  : STOP")
                lines.append(f"Reason  : {s.get('stop_reason', '')}")
            else:
                lines.append(f"Action  : SEARCH  query=\"{s['query']}\"")
                lines.append(f"Observation: {s['observation']}")

        # Hint when approaching the limit
        n_searches = sum(1 for s in steps if s["action"] == "search")
        if n_searches >= MAX_REACT_ITERATIONS - 1:
            lines += [
                "",
                f"NOTE: {n_searches}/{MAX_REACT_ITERATIONS} searches used.",
                "Consider stopping now unless a critical angle is missing.",
            ]

    lines += ["", "Your next thought and action:"]
    return "\n".join(lines)


# ── Observation formatter ─────────────────────────────────────────────────────

def _format_observation(
    query: str,
    articles: list[Article],
    n_new: int,
    total_seen: int,
) -> str:
    """Convert search results into a text observation for the LLM."""
    n = len(articles)
    if n == 0:
        return f'0 articles found for "{query}". This angle is not well-indexed.'

    # Top concepts (weighted by frequency)
    counts: dict[str, int] = {}
    for art in articles:
        for c in (art.concepts or [])[:4]:
            counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])[:5]
    concepts_str = ", ".join(f"{c}({cnt})" for c, cnt in top) or "none"

    # Top articles by citation count
    ranked = sorted(articles, key=lambda a: -a.citation_count)[:3]
    samples = "\n".join(
        f"  [{a.year}] {a.title[:65]}  (cited {a.citation_count}x)"
        for a in ranked
    )

    return (
        f"{n} articles found ({n_new} new unique). Total unique seen: {total_seen}\n"
        f"Top concepts: {concepts_str}\n"
        f"Sample titles:\n{samples}"
    )


# ── Probe search (small, for observation only) ────────────────────────────────

def _search_probe(query: str, n: int) -> list[Article]:
    """Fire a small OpenAlex probe for ReAct observation.

    No from_year filter here on purpose: the probe is for *discovery*
    (does this angle exist in OpenAlex at all?).  The final year filter
    is applied by the Searcher agent.

    Failure is silenced: returning [] lets the agent reason that this
    angle produced nothing and try a different one.
    """
    try:
        return OpenAlexClient().search(query=query, n_results=n)
    except Exception as exc:
        logger.debug("[%s] probe failed for '%s': %s", AGENT_NAME, query, exc)
        return []


# ── Main agent function ───────────────────────────────────────────────────────

def run(state: WatchState, llm: Optional[LLMClient] = None) -> dict:
    """ReAct loop: think → search probe → observe → repeat → stop.

    Returns expanded_queries (for Searcher) and react_steps (for display).
    Always succeeds softly: on full LLM failure falls back to [topic].
    """
    log = start_log(AGENT_NAME)
    topic: str = state.get("topic", "")
    n_raw: int = state.get("config", {}).get("n_raw", 30)
    llm = llm or get_llm_for_task("query_expansion")

    steps: list[dict] = []           # full loop history (thought+action+obs)
    queries_found: list[str] = []    # queries that returned ≥1 article
    queries_tried: set[str] = set()  # for duplicate detection
    all_seen_ids: set[str] = set()   # tracks unique articles across probes
    total_tokens = 0
    total_api_calls = 0

    for iteration in range(1, MAX_REACT_ITERATIONS + 1):
        # ── THINK + ACT: LLM chooses the next action ──────────────────────
        try:
            step_action, resp = llm.chat_structured(
                system=SYSTEM_PROMPT,
                messages=[Message(
                    role="user",
                    content=_build_prompt(topic, n_raw, steps),
                )],
                schema=ReActThoughtAction,
                temperature=0.3,
                max_tokens=512,
            )
            total_tokens += resp.input_tokens + resp.output_tokens
            total_api_calls += 1

        except LLMError as exc:
            logger.warning(
                "[%s] LLM error at iteration %d: %s", AGENT_NAME, iteration, exc
            )
            break   # exit loop; use whatever queries were found so far

        # ── STOP branch ───────────────────────────────────────────────────
        if step_action.action == "stop" or step_action.search_query is None:
            steps.append({
                "iteration":   iteration,
                "thought":     step_action.thought,
                "action":      "stop",
                "stop_reason": step_action.stop_reason or "Coverage deemed sufficient.",
                "n_found":     0,
                "observation": "",
            })
            logger.info(
                "[%s] iter=%d STOP. %d queries found so far.",
                AGENT_NAME, iteration, len(queries_found),
            )
            break

        # ── SEARCH branch: execute probe + observe ────────────────────────
        query = step_action.search_query.strip()
        q_key = query.lower()

        if q_key in queries_tried:
            # Duplicate: feed back a note so the LLM tries a different angle.
            obs = (f'Query "{query}" was already tried. '
                   "Please choose a different angle.")
            n_new = 0
        else:
            queries_tried.add(q_key)
            articles = _search_probe(query, n=REACT_PROBE_N)

            # Count newly discovered articles (dedup by ID)
            n_new = sum(1 for a in articles if a.id not in all_seen_ids)
            all_seen_ids.update(a.id for a in articles)

            if articles:
                queries_found.append(query)

            obs = _format_observation(query, articles, n_new, len(all_seen_ids))

        steps.append({
            "iteration":   iteration,
            "thought":     step_action.thought,
            "action":      "search",
            "query":       query,
            "observation": obs,
            "n_found":     n_new,
        })
        logger.info(
            "[%s] iter=%d search=%r -> %d new unique (total seen: %d)",
            AGENT_NAME, iteration, query, n_new, len(all_seen_ids),
        )

    # ── Build final output ────────────────────────────────────────────────
    # Always include the original topic (in case the LLM skipped it).
    if topic.lower() not in {q.lower() for q in queries_found}:
        queries_found = [topic] + queries_found
    final_queries = queries_found or [topic]

    status = "success" if total_api_calls > 0 else "failed"
    logger.info(
        "[%s] ReAct done: %d step(s), %d queries → %s",
        AGENT_NAME, len(steps), len(final_queries), final_queries,
    )

    out: dict = {
        "expanded_queries": final_queries,
        "react_steps":      steps,
        "logs": [
            finish_log(
                log, status,
                tokens_used=total_tokens,
                api_calls=total_api_calls,
            )
        ],
    }
    if status == "failed":
        out["errors"] = [
            f"{AGENT_NAME}: all LLM calls failed — using original topic as fallback"
        ]
    return out
