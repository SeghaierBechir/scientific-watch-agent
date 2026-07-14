"""End-to-end demo — V2 + Hybrid ReAct+Reflexion pipeline (Phase 5 + 6).

Now shows:
    - QueryExpander ReAct loop  (THINK -> SEARCH -> OBSERVE -> STOP)
    - Expanded queries after exploration
    - Reflexion loop: each Critic iteration with quality, issues, suggestions
    - Final synthesis iteration count
    - Full agent logs with tokens and cost

Usage:
    python demo_phase5.py "fake news detection"
    python demo_phase5.py "fake news detection" 30 5
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from src.agents.graph import run_pipeline
from src.config import MAX_REACT_ITERATIONS, MAX_REFLEXION_ITERATIONS, REFLEXION_MIN_QUALITY
from src.schemas import AgentLog, CriticFeedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

QUALITY_EMOJI = {
    "poor":       "XX",
    "acceptable": "~~",
    "good":       "OK",
    "excellent":  "**",
}


def _print_react_steps(steps: list[dict]) -> None:
    """Display the QueryExpander ReAct loop (think → search → observe → stop)."""
    _hr(f"QUERY EXPANDER  ReAct LOOP  (max={MAX_REACT_ITERATIONS} iterations)",
        char="-")

    if not steps:
        print("  No ReAct steps recorded.")
        return

    n_productive = sum(
        1 for s in steps if s["action"] == "search" and s.get("n_found", 0) > 0
    )
    print(f"  {len(steps)} iteration(s)  |  {n_productive} productive search(es)\n")

    for s in steps:
        it = s.get("iteration", "?")
        if s["action"] == "stop":
            print(f"  [Iter {it}]  STOP")
            print(f"    Thought  : {s['thought']}")
            print(f"    Reason   : {s.get('stop_reason', 'done')}")
        else:
            n = s.get("n_found", 0)
            result_tag = f"{n} new articles" if n > 0 else "0 results"
            print(f"  [Iter {it}]  SEARCH -> '{s['query']}'  ({result_tag})")
            print(f"    Thought  : {s['thought']}")
            # First line of the observation (article count + concepts)
            obs_first = s.get("observation", "").split("\n")[0]
            print(f"    Observed : {obs_first}")


def _hr(title: str, char: str = "=") -> None:
    print("\n" + char * 70)
    print(title)
    print(char * 70)


def _print_reflexion(feedbacks: list[CriticFeedback], final_iteration: int) -> None:
    """Display the full Reflexion loop history."""
    _hr(f"REFLEXION LOOP  (max={MAX_REFLEXION_ITERATIONS}, min_quality={REFLEXION_MIN_QUALITY})")

    if not feedbacks:
        print("  No Critic feedback recorded.")
        return

    for fb in feedbacks:
        emoji = QUALITY_EMOJI.get(fb.overall_quality, "??")
        revised = "-> REVISION REQUESTED" if fb.needs_revision else "-> APPROVED"
        print(f"\n  Iteration {fb.iteration}  [{emoji}] quality={fb.overall_quality}  {revised}")

        if fb.issues:
            print("    Issues found:")
            for issue in fb.issues:
                print(f"      - {issue}")

        if fb.suggestions:
            print("    Suggestions given to Synthesizer:")
            for sug in fb.suggestions:
                print(f"      + {sug}")

        if not fb.issues:
            print("    No issues — synthesis accepted as-is.")

    print(f"\n  Final: Synthesizer ran {final_iteration} time(s) total.")


def _print_logs(logs: list[AgentLog]) -> None:
    _hr("AGENT LOGS  (tokens | cost)")
    total_tokens = 0
    total_cost = 0.0

    # Estimate cost from tokens (rough, using Sonnet pricing as fallback).
    for log in logs:
        duration = (
            (log.completed_at - log.started_at).total_seconds()
            if log.completed_at else 0
        )
        marker = "OK " if log.status == "success" else "ERR"
        # Show iteration number for Synthesizer and Critic.
        name = log.agent_name
        print(
            f"  [{marker}] {name:<16} "
            f"{duration:>5.2f}s  "
            f"calls={log.api_calls:<3}  "
            f"tokens={log.tokens_used:<6}"
        )
        if log.error:
            print(f"          error: {log.error[:100]}")
        total_tokens += log.tokens_used

    print(f"\n  TOTAL: {total_tokens} tokens across all agents")


def main(topic: str = "fake news detection", n_raw: int = 30, top_n: int = 5):
    _hr(f"Scientific Watch Agent  Hybrid ReAct+Reflexion  |  topic: '{topic}'")
    print(f"  Config: n_raw={n_raw}  top_n={top_n}  "
          f"max_react={MAX_REACT_ITERATIONS}  "
          f"max_reflexion={MAX_REFLEXION_ITERATIONS}  "
          f"min_quality={REFLEXION_MIN_QUALITY}")
    print(f"  Started: {datetime.now().isoformat(timespec='seconds')}")

    final = run_pipeline(topic, n_raw=n_raw, top_n=top_n)

    # ── ReAct loop (QueryExpander) ────────────────────────────
    _print_react_steps(final.get("react_steps", []))

    # ── Expanded queries (result of the ReAct loop) ───────────
    queries = final.get("expanded_queries") or [topic]
    _hr(f"EXPANDED QUERIES  ({len(queries)} found by ReAct)")
    for i, q in enumerate(queries, 1):
        tag = "[original]" if i == 1 else f"[variant {i-1}]"
        print(f"  {i}. {tag}  {q}")

    # ── Top articles ──────────────────────────────────────────
    _hr(f"TOP {len(final.get('top_articles', []))} ARTICLES  (after quality filter)")
    for rank, (art, sc) in enumerate(
        zip(final.get("top_articles", []), final.get("top_scores", [])), 1
    ):
        venue = art.journal_name or "[no venue]"
        if art.is_preprint:
            venue += " (preprint)"
        print(f"  #{rank} [score={sc.final_score:.2f}] {art.title[:85]}")
        print(f"       {venue} | {art.year} | cited {art.citation_count}x")
        print(f"       venue={sc.venue_score:.2f}  "
              f"authors={sc.authors_score:.2f}  "
              f"impact={sc.impact_score:.2f}  "
              f"relevance={sc.relevance_score:.2f}")

    # ── Summaries ─────────────────────────────────────────────
    _hr(f"SUMMARIES  ({len(final.get('summaries', []))} articles summarized)")
    for i, s in enumerate(final.get("summaries", []), 1):
        print(f"\n  [{i}] {s.article_id}")
        print(f"      Problem  : {s.problem}")
        print(f"      Method   : {s.method}")
        print(f"      Dataset  : {s.dataset or 'Not specified'}")
        print(f"      Results  : {s.results[:160]}")
        if s.key_contributions:
            print(f"      Key ideas: {' | '.join(s.key_contributions[:3])}")

    # ── Reflexion loop ────────────────────────────────────────
    _print_reflexion(
        final.get("critic_feedbacks", []),
        final.get("synthesis_iteration", 1),
    )

    # ── Final synthesis ───────────────────────────────────────
    synth = final.get("synthesis")
    iterations = final.get("synthesis_iteration", 1)
    if synth:
        _hr(f"FINAL SYNTHESIS  (after {iterations} iteration(s))")
        print(f"\nOverview:\n  {synth.overview}\n")
        print("Main approaches:")
        for a in synth.main_approaches:
            print(f"  - {a}")
        if synth.common_datasets:
            print("\nCommon datasets:")
            for d in synth.common_datasets:
                print(f"  - {d}")
        print("\nKey findings:")
        for f in synth.key_findings:
            print(f"  - {f}")

    # ── Trend analysis ────────────────────────────────────────
    trends = final.get("trend_analysis")
    if trends:
        _hr("TRENDS, GAPS, PERSPECTIVES")
        print("\nTrends:")
        for t in trends.trends:
            print(f"  [{t.maturity:>11}] {t.name}")
            print(f"               {t.description}")
            if t.evidence_article_ids:
                print(f"               evidence: {', '.join(t.evidence_article_ids[:3])}")
        print("\nResearch gaps:")
        for g in trends.gaps:
            print(f"  [importance={g.importance:<6}] {g.description}")
            for d in g.suggested_directions[:2]:
                print(f"               -> {d}")
        print("\nFuture perspectives:")
        for p in trends.future_perspectives:
            print(f"  - {p}")

    # ── Logs + errors ─────────────────────────────────────────
    _print_logs(final.get("logs", []))

    if final.get("errors"):
        _hr("ERRORS", char="!")
        for err in final["errors"]:
            print(f"  - {err}")

    _hr("DONE")


if __name__ == "__main__":
    topic_arg = sys.argv[1] if len(sys.argv) > 1 else "fake news detection"
    n_raw_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    top_n_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    main(topic_arg, n_raw_arg, top_n_arg)
