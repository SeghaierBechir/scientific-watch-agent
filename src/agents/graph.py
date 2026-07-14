"""LangGraph orchestrator for V2 + Reflexion pipeline (Phase 5 + 6).

Pipeline:

    QueryExpander → Searcher → QualityCritic → Summarizer
                                                    ↓
                                               Synthesizer ←──────────┐
                                                    ↓                 │ needs_revision=True
                                                  Critic ─────────────┘ (max 3 iterations)
                                                    ↓ needs_revision=False
                                               TrendAnalyst → END

Pédagogique — LangGraph conditional edges:
    `add_conditional_edges(source, routing_fn, mapping)` adds a branch after
    `source`. `routing_fn(state) -> str` returns the name of the next node.
    `mapping` is a dict {return_value: node_name} used by LangGraph to
    validate that all branches are declared.

Public API:
    build_graph()                          -> compiled LangGraph
    run_pipeline(topic, **config_kwargs)   -> final WatchState dict
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents import (
    critic,
    quality_critic,
    query_expander,
    searcher,
    summarizer,
    synthesizer,
    trend_analyst,
)
from src.agents.state import RunConfig, WatchState
from src.config import DEFAULT_FROM_YEAR, MAX_REFLEXION_ITERATIONS

logger = logging.getLogger(__name__)


# ============================================================
# Node names — constants to avoid typos in edges
# ============================================================

N_EXPANDER   = "query_expander"
N_SEARCHER   = "searcher"
N_CRITIC_Q   = "quality_critic"
N_SUMMARIZER = "summarizer"
N_SYNTHESIZER = "synthesizer"
N_CRITIC     = "critic"
N_TRENDS     = "trend_analyst"


# ============================================================
# Routing function — the heart of the Reflexion loop
# ============================================================

def route_after_critic(state: WatchState) -> str:
    """Decide what comes after the Critic.

    Returns N_SYNTHESIZER if the synthesis needs revision and we haven't
    hit the iteration cap. Returns N_TRENDS otherwise.

    This function is called by LangGraph after every Critic run.
    """
    feedbacks = state.get("critic_feedbacks", [])
    iteration = state.get("synthesis_iteration", 0)

    if not feedbacks:
        return N_TRENDS

    last = feedbacks[-1]

    if last.needs_revision and iteration < MAX_REFLEXION_ITERATIONS:
        logger.info(
            "[Router] iteration=%d quality=%s → looping back to Synthesizer",
            iteration, last.overall_quality,
        )
        return N_SYNTHESIZER

    logger.info(
        "[Router] iteration=%d quality=%s → proceeding to TrendAnalyst",
        iteration, last.overall_quality,
    )
    return N_TRENDS


# ============================================================
# Graph builder
# ============================================================

def build_graph():
    """Wire all agents into a LangGraph with Reflexion loop and compile it."""
    g: StateGraph = StateGraph(WatchState)

    # Register every node (name → callable).
    g.add_node(N_EXPANDER,    query_expander.run)
    g.add_node(N_SEARCHER,    searcher.run)
    g.add_node(N_CRITIC_Q,    quality_critic.run)
    g.add_node(N_SUMMARIZER,  summarizer.run)
    g.add_node(N_SYNTHESIZER, synthesizer.run)
    g.add_node(N_CRITIC,      critic.run)
    g.add_node(N_TRENDS,      trend_analyst.run)

    # Linear edges (no branching).
    g.set_entry_point(N_EXPANDER)
    g.add_edge(N_EXPANDER,    N_SEARCHER)
    g.add_edge(N_SEARCHER,    N_CRITIC_Q)
    g.add_edge(N_CRITIC_Q,    N_SUMMARIZER)
    g.add_edge(N_SUMMARIZER,  N_SYNTHESIZER)
    g.add_edge(N_SYNTHESIZER, N_CRITIC)

    # Conditional edge — Reflexion loop.
    # After Critic: either loop back to Synthesizer or go to TrendAnalyst.
    g.add_conditional_edges(
        N_CRITIC,
        route_after_critic,
        {
            N_SYNTHESIZER: N_SYNTHESIZER,  # revision needed
            N_TRENDS:      N_TRENDS,       # quality OK
        },
    )

    g.add_edge(N_TRENDS, END)

    return g.compile()


# ============================================================
# Public convenience function
# ============================================================

def run_pipeline(
    topic: str,
    *,
    n_raw: int = 30,
    top_n: int = 10,
    from_year: int = DEFAULT_FROM_YEAR,
    weights: dict[str, float] | None = None,
    require_abstract: bool = True,
    narrative_mode: bool = False,
) -> dict[str, Any]:
    """Run the full V2 + Reflexion pipeline and return the final state.

    Args:
        topic:            research topic to investigate.
        n_raw:            articles to fetch from OpenAlex.
        top_n:            articles to keep after QualityCritic.
        from_year:        earliest publication year.
        weights:          optional QualityScore weight overrides.
        require_abstract: drop articles without abstract.

    Returns:
        Final state dict with all populated fields, including:
            synthesis_iteration  — how many Synthesizer runs happened
            critic_feedbacks     — list of CriticFeedback (one per iteration)
    """
    cfg: RunConfig = {
        "n_raw": n_raw,
        "top_n": top_n,
        "from_year": from_year,
        "require_abstract": require_abstract,
    }
    if weights is not None:
        cfg["weights"] = weights

    initial_state: dict = {
        "topic": topic,
        "config": cfg,
        "narrative_mode": narrative_mode,
        "expanded_queries": [],
        "react_steps": [],          # populated by QueryExpander (ReAct loop)
        "synthesis_iteration": 0,
        "critic_feedbacks": [],
        "logs": [],
        "errors": [],
    }

    graph = build_graph()
    logger.info(
        "Running pipeline on topic=%r (n_raw=%d, top_n=%d, max_reflexion=%d)",
        topic, n_raw, top_n, MAX_REFLEXION_ITERATIONS,
    )
    final = graph.invoke(initial_state)
    logger.info(
        "Pipeline done. summaries=%d synthesis_iterations=%d errors=%d",
        len(final.get("summaries") or []),
        final.get("synthesis_iteration", 0),
        len(final.get("errors") or []),
    )
    return final
