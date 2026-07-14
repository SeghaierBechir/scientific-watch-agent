"""Shared state for the LangGraph multi-agent pipeline (V2).

The state is a `TypedDict` that flows through every node. Each agent reads
what it needs and writes only its own section (Single Writer / Multiple
Readers, see CLAUDE.md §4).

LangGraph note (pédagogique):
    - In LangGraph, each "node" is a callable `state -> dict` that returns a
      *partial* update merged into the global state. We never mutate `state`
      in place; we return new fields.
    - Lists in this state are NOT auto-appended by LangGraph by default — if
      a node returns `{"logs": [new_log]}` it REPLACES the list. To append we
      either copy-extend explicitly or use `Annotated[list, operator.add]`.
      We use `Annotated` for `logs` and `errors` because every agent appends.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from src.schemas import (
    AgentLog,
    Article,
    ArticleSummary,
    CriticFeedback,
    NarrativeSummary,
    QualityScore,
    Synthesis,
    TrendAnalysis,
)


class RunConfig(TypedDict, total=False):
    """Per-run knobs passed in via the initial state."""

    n_raw: int             # how many articles to fetch
    top_n: int             # how many to keep after quality filter
    from_year: int         # earliest publication year
    weights: dict[str, float]  # quality score weights override
    require_abstract: bool


class WatchState(TypedDict, total=False):
    """Global state of one Scientific Watch run.

    Section ownership (single writer):
        - topic, config:        set by the caller, read by everyone
        - raw_articles:         Searcher
        - quality_scores:       QualityCritic
        - top_articles, top_scores: QualityCritic
        - summaries:            Summarizer
        - synthesis:            Synthesizer
        - trend_analysis:       TrendAnalyst
        - logs, errors:         appended by every node (use operator.add)
    """

    # --- Inputs ---
    topic: str
    config: RunConfig

    # --- QueryExpander output (ReAct loop) ---
    expanded_queries: list[str]   # queries that produced results, used by Searcher
    react_steps: list[dict]       # full ReAct history for display/debugging

    # --- Searcher output ---
    raw_articles: list[Article]

    # --- QualityCritic output ---
    quality_scores: list[QualityScore]
    top_articles: list[Article]
    top_scores: list[QualityScore]

    # --- Summarizer output ---
    narrative_mode: bool                         # True = prose paragraph summaries
    summaries: list[ArticleSummary]              # structured mode (6 fields)
    narrative_summaries: list[NarrativeSummary]  # narrative mode (prose paragraph)

    # --- Synthesizer output ---
    synthesis: Optional[Synthesis]
    synthesis_iteration: int          # how many times Synthesizer has run (starts at 0)

    # --- Critic output (Reflexion — Phase 6) ---
    # Annotated with operator.add so each Critic run appends its feedback.
    critic_feedbacks: Annotated[list[CriticFeedback], operator.add]

    # --- TrendAnalyst output ---
    trend_analysis: Optional[TrendAnalysis]

    # --- Cross-cutting (auto-appended thanks to operator.add) ---
    logs: Annotated[list[AgentLog], operator.add]
    errors: Annotated[list[str], operator.add]
