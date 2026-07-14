"""Compare V1.5 (keyword/bigram) vs V2 (semantic embeddings) relevance scoring.

Double-Optuna experiment: runs Optuna twice per domain — once with V1.5
keyword/bigram scoring, once with V2 semantic embeddings — holding
everything else constant (same corpus, same oracle, same n_trials).

Output per domain:
  - Full metrics table: NDCG@15, NDCG@10, P@5, P@10, Recall@30, MAP
    for DEFAULT and LEARNED weights, for both V1.5 and V2
  - Full weights table: DEFAULT vs Learned-V1.5 vs Learned-V2 for all 6 features
  - Summary NDCG@15 table across all domains

Thesis context (Chapter 4 — AutoML)
-------------------------------------
Research question: "Does replacing keyword overlap with semantic embeddings
improve the quality of AutoML-optimised article ranking?"

Usage
-----
    # All oracle domains (~15 min for 5 domains)
    python compare_v1_v2.py --all

    # Single domain
    python compare_v1_v2.py --domain fake_news_detection

    # Quick sanity-check with synthetic data (no API call, ~30s)
    python compare_v1_v2.py --synthetic

    # Force Optuna to ignore cached studies and restart
    python compare_v1_v2.py --all --force-rerun

    # Fewer trials for quick testing
    python compare_v1_v2.py --all --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WEIGHTS, OPTUNA_N_TRIALS
from src.schemas import Article, Author, SourceType
from src.scoring.automl_scorer import OptimizationResult, optimize_weights
from src.scoring.metrics import evaluate_all
from src.scoring.quality_scorer import _SEMANTIC_AVAILABLE, score_articles  # noqa: PLC2701

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Ordered metric keys for consistent display
_METRIC_KEYS = ["ndcg@15", "ndcg@10", "p@5", "p@10", "recall@30", "map"]
_WEIGHT_KEYS = ["venue", "authors", "impact", "velocity", "recency", "relevance"]


# ── Result container ──────────────────────────────────────────────────────────

class DomainComparison(NamedTuple):
    """Stores both Optuna runs + full metrics for one domain."""
    domain_id: str
    topic: str
    result_v1: OptimizationResult
    result_v2: OptimizationResult
    metrics_v1_default: dict[str, float]   # DEFAULT weights + V1.5 scoring
    metrics_v1_learned: dict[str, float]   # Learned V1.5 weights + V1.5 scoring
    metrics_v2_default: dict[str, float]   # DEFAULT weights + V2 scoring
    metrics_v2_learned: dict[str, float]   # Learned V2 weights + V2 scoring


# ── Scoring helper ────────────────────────────────────────────────────────────

def _eval_with_weights(
    articles: list[Article],
    gold: dict[str, int],
    topic: str,
    weights: dict[str, float],
    use_semantic: bool,
) -> dict[str, float]:
    """Score articles with given weights + relevance method, return all metrics."""
    scores = score_articles(articles, topic, weights, use_semantic=use_semantic)
    paired = sorted(zip(articles, scores), key=lambda p: p[1].final_score, reverse=True)
    ranked_ids = [a.id for a, _ in paired]
    return evaluate_all(ranked_ids, gold)


# ── Synthetic corpus (no API calls) ──────────────────────────────────────────

def _make_synthetic_corpus() -> tuple[list[Article], dict[str, int], str]:
    """Minimal corpus for quick sanity-check."""
    topic = "fake news detection"
    articles: list[Article] = []
    gold: dict[str, int] = {}

    for i in range(1, 6):
        a = Article(
            id=f"rel2_{i}",
            title=f"Fake News Detection via Transformer Models: Study {i}",
            abstract="We propose a deep learning approach for detecting fake news "
                     "on social media using BERT and attention mechanisms.",
            year=2023, source=SourceType.OPENALEX,
            url=f"https://example.com/rel2_{i}",
            citation_count=200 + i * 50,
            concepts=["fake news", "detection", "NLP", "transformers"],
            journal_name="IEEE Transactions on Neural Networks", quartile="Q1",
            authors=[Author(name=f"Author A{i}", h_index=25 + i, citation_count=5000)],
        )
        articles.append(a); gold[a.id] = 2

    for i in range(1, 11):
        a = Article(
            id=f"rel1_{i}",
            title=f"Misinformation Detection on Social Media: Survey {i}",
            abstract="This paper surveys methods for detecting misinformation "
                     "and rumor propagation in online social networks.",
            year=2022, source=SourceType.OPENALEX,
            url=f"https://example.com/rel1_{i}",
            citation_count=80 + i * 10,
            concepts=["fake news", "misinformation", "social media"],
            journal_name="ACM Computing Surveys", quartile="Q2",
            authors=[Author(name=f"Author B{i}", h_index=12 + i, citation_count=2000)],
        )
        articles.append(a); gold[a.id] = 1

    for i in range(35):
        unrelated = [
            ("U-Net for Medical Image Segmentation", "computer vision"),
            ("Reinforcement Learning for Robotics", "reinforcement learning"),
            ("Graph Neural Networks for Molecules", "graph neural networks"),
        ]
        title, concept = unrelated[i % len(unrelated)]
        a = Article(
            id=f"bg_{i}", title=f"{title} Part {i}",
            abstract=f"We study {concept} with deep neural networks.",
            year=2022, source=SourceType.OPENALEX,
            url=f"https://example.com/bg_{i}",
            citation_count=30 + i * 5,
            concepts=[concept, "deep learning"],
            journal_name="NeurIPS", quartile="Q1",
            authors=[Author(name=f"Author C{i}", h_index=8, citation_count=500)],
        )
        articles.append(a)

    return articles, gold, topic


# ── Oracle loader ─────────────────────────────────────────────────────────────

def _load_oracle(domain_id: str) -> tuple[list[Article], dict[str, int], str]:
    from data.oracle.build_oracle import load_oracle  # noqa: PLC0415
    articles, gold = load_oracle(domain_id)
    config_path = PROJECT_ROOT / "data" / "oracle" / "domains_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    domain_cfg = next(d for d in config["domains"] if d["id"] == domain_id)
    return articles, gold, domain_cfg["topic"]


def _all_domain_ids() -> list[str]:
    config_path = PROJECT_ROOT / "data" / "oracle" / "domains_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return [d["id"] for d in config["domains"]]


# ── Core comparison runner ────────────────────────────────────────────────────

def _run_comparison(
    domain_id: str,
    articles: list[Article],
    gold: dict[str, int],
    topic: str,
    n_trials: int,
    force_rerun: bool,
) -> DomainComparison | None:
    """Run Optuna twice (V1.5 then V2) + compute full metrics for one domain.

    Returns None (and prints a warning) when the oracle gold set is empty —
    which happens for arXiv-survey domains where OpenAlex has no referenced_works.
    """
    n_g2 = sum(1 for g in gold.values() if g == 2)
    n_g1 = sum(1 for g in gold.values() if g == 1)
    n_g0 = len(articles) - len(gold)
    print(f"\n  Corpus : {len(articles)} articles "
          f"(grade-2={n_g2}, grade-1={n_g1}, background={n_g0})")

    if len(gold) == 0:
        print(f"  SKIP — gold_relevance is empty (0 relevant articles).")
        print(f"  Cause: survey DOIs are likely arXiv preprints with no")
        print(f"         referenced_works in OpenAlex.")
        print(f"  Fix:   python data/oracle/build_oracle.py \\")
        print(f"           --domain {domain_id} --citation-fallback")
        return None

    # ── V1.5 Optuna run ───────────────────────────────────────────────────
    print(f"  [V1.5] Running Optuna ({n_trials} trials, keyword/bigram) ...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    result_v1 = optimize_weights(
        articles, gold, topic,
        n_trials=n_trials, force_rerun=force_rerun, use_semantic=False,
    )
    print(f"done in {time.perf_counter()-t0:.0f}s  "
          f"NDCG@15: {result_v1.baseline_ndcg_at_15:.4f} -> "
          f"{result_v1.best_ndcg_at_15:.4f} ({result_v1.improvement_pct:+.1f}%)")

    # ── V2 Optuna run ─────────────────────────────────────────────────────
    print(f"  [V2 ] Running Optuna ({n_trials} trials, semantic embed.) ...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    result_v2 = optimize_weights(
        articles, gold, topic,
        n_trials=n_trials, force_rerun=force_rerun, use_semantic=True,
    )
    print(f"done in {time.perf_counter()-t0:.0f}s  "
          f"NDCG@15: {result_v2.baseline_ndcg_at_15:.4f} -> "
          f"{result_v2.best_ndcg_at_15:.4f} ({result_v2.improvement_pct:+.1f}%)")

    # ── Compute full metrics for all 4 combinations ───────────────────────
    print(f"  Computing full metrics ...", end=" ", flush=True)
    m_v1_def = _eval_with_weights(articles, gold, topic, DEFAULT_WEIGHTS, use_semantic=False)
    m_v1_lrn = _eval_with_weights(articles, gold, topic, result_v1.best_weights, use_semantic=False)
    m_v2_def = _eval_with_weights(articles, gold, topic, DEFAULT_WEIGHTS, use_semantic=True)
    m_v2_lrn = _eval_with_weights(articles, gold, topic, result_v2.best_weights, use_semantic=True)
    print("done")

    return DomainComparison(
        domain_id, topic, result_v1, result_v2,
        m_v1_def, m_v1_lrn, m_v2_def, m_v2_lrn,
    )


# ── Per-domain detailed display ───────────────────────────────────────────────

def _print_domain_detail(cmp: DomainComparison) -> None:
    """Print full metrics + weights for one domain (4-column layout)."""
    W = 70
    print(f"\n{'='*W}")
    print(f"  Domain : {cmp.domain_id}")
    print(f"  Topic  : '{cmp.topic}'")
    print(f"{'='*W}")

    # ── Metrics table ─────────────────────────────────────────────────────
    print(f"\n  METRICS  (Default = DEFAULT_WEIGHTS  |  Learned = Optuna best)")
    print(f"  {'-'*65}")
    print(f"  {'Metric':<12} {'Def-V1.5':>9} {'Lrn-V1.5':>9} "
          f"{'Def-V2':>9} {'Lrn-V2':>9}  {'d(Opt)':>8}")
    print(f"  {'-'*65}")

    for k in _METRIC_KEYS:
        dv1 = cmp.metrics_v1_default.get(k, 0.0)
        lv1 = cmp.metrics_v1_learned.get(k, 0.0)
        dv2 = cmp.metrics_v2_default.get(k, 0.0)
        lv2 = cmp.metrics_v2_learned.get(k, 0.0)
        d_opt = lv2 - lv1
        sign = "+" if d_opt >= 0 else ""
        print(f"  {k:<12} {dv1:>9.4f} {lv1:>9.4f} "
              f"{dv2:>9.4f} {lv2:>9.4f}  {sign}{d_opt:>7.4f}")

    print(f"  {'-'*65}")

    # ── Weights table ─────────────────────────────────────────────────────
    print(f"\n  WEIGHTS  (Learned V1.5 = Optuna with keyword  |  Learned V2 = Optuna with embeddings)")
    print(f"  {'-'*65}")
    print(f"  {'Feature':<12} {'Default':>9} {'Lrn-V1.5':>9} "
          f"{'Lrn-V2':>9}  {'d(V2-V1.5)':>10}")
    print(f"  {'-'*65}")

    for k in _WEIGHT_KEYS:
        dw  = DEFAULT_WEIGHTS.get(k, 0.0)
        w1  = cmp.result_v1.best_weights.get(k, 0.0)
        w2  = cmp.result_v2.best_weights.get(k, 0.0)
        d   = w2 - w1
        sign = "+" if d >= 0 else ""
        # Highlight relevance row
        marker = " <--" if k == "relevance" else ""
        print(f"  {k:<12} {dw:>9.4f} {w1:>9.4f} "
              f"{w2:>9.4f}  {sign}{d:>9.4f}{marker}")

    print(f"  {'-'*65}")
    print(f"  Optuna improvement : "
          f"V1.5 = {cmp.result_v1.improvement_pct:+.1f}%  |  "
          f"V2 = {cmp.result_v2.improvement_pct:+.1f}%")


# ── Cross-domain summary table (NDCG@15 only) ────────────────────────────────

def _print_summary_table(comparisons: list[DomainComparison]) -> None:
    """Print NDCG@15 summary across all domains."""
    if len(comparisons) <= 1:
        return  # single domain already has full detail above

    W = 78
    print(f"\n{'='*W}")
    print(f"  CROSS-DOMAIN SUMMARY  —  NDCG@15")
    print(f"{'='*W}")
    print(f"  {'Domain':<26} {'Def-V1':>8} {'Lrn-V1':>8} "
          f"{'Def-V2':>8} {'Lrn-V2':>8}  {'dBase':>7} {'dOpt':>7}")
    print(f"  {'-'*(W-2)}")

    sums = [0.0] * 4
    for cmp in comparisons:
        bv1 = cmp.metrics_v1_default.get("ndcg@15", 0.0)
        lv1 = cmp.metrics_v1_learned.get("ndcg@15", 0.0)
        bv2 = cmp.metrics_v2_default.get("ndcg@15", 0.0)
        lv2 = cmp.metrics_v2_learned.get("ndcg@15", 0.0)
        db = bv2 - bv1; do = lv2 - lv1
        sb = "+" if db >= 0 else ""; so = "+" if do >= 0 else ""
        print(f"  {cmp.domain_id:<26} {bv1:>8.4f} {lv1:>8.4f} "
              f"{bv2:>8.4f} {lv2:>8.4f}  {sb}{db:>6.4f} {so}{do:>6.4f}")
        for i, v in enumerate([bv1, lv1, bv2, lv2]):
            sums[i] += v

    n = len(comparisons)
    if n > 1:
        means = [s / n for s in sums]
        db = means[2] - means[0]; do = means[3] - means[1]
        sb = "+" if db >= 0 else ""; so = "+" if do >= 0 else ""
        print(f"  {'-'*(W-2)}")
        print(f"  {'Mean':<26} {means[0]:>8.4f} {means[1]:>8.4f} "
              f"{means[2]:>8.4f} {means[3]:>8.4f}  {sb}{db:>6.4f} {so}{do:>6.4f}")
    print(f"{'='*W}")


# ── Winner summary ────────────────────────────────────────────────────────────

def _print_winner(comparisons: list[DomainComparison]) -> None:
    print(f"\n  POST-OPTUNA WINNER (NDCG@15):")
    v2_wins = v1_wins = ties = 0
    for cmp in comparisons:
        lv1 = cmp.metrics_v1_learned.get("ndcg@15", 0.0)
        lv2 = cmp.metrics_v2_learned.get("ndcg@15", 0.0)
        if lv2 > lv1 + 0.001:
            winner = "V2 wins  "; v2_wins += 1
        elif lv1 > lv2 + 0.001:
            winner = "V1.5 wins"; v1_wins += 1
        else:
            winner = "tie      "; ties += 1
        print(f"    {cmp.domain_id:<30}  {winner}  "
              f"(V1.5={lv1:.4f}, V2={lv2:.4f}, d={lv2-lv1:+.4f})")
    print(f"\n  V2 wins: {v2_wins}/{len(comparisons)}  |  "
          f"V1.5 wins: {v1_wins}/{len(comparisons)}  |  "
          f"Ties: {ties}/{len(comparisons)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Double-Optuna: full metrics comparison V1.5 vs V2 relevance"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all",       action="store_true",
                      help="Run all oracle domains")
    mode.add_argument("--domain",    metavar="DOMAIN_ID",
                      help="Single domain (oracle must be pre-built)")
    mode.add_argument("--synthetic", action="store_true",
                      help="Use synthetic corpus (no API call needed)")
    parser.add_argument("--n-trials",    type=int, default=OPTUNA_N_TRIALS,
                        help=f"Optuna trials per run (default: {OPTUNA_N_TRIALS})")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Delete cached Optuna studies and restart")
    args = parser.parse_args()

    if not _SEMANTIC_AVAILABLE:
        print("\n[WARNING] sentence-transformers not installed — V2 falls back to V1.5.")
        print("  Install: pip install sentence-transformers\n")

    print(f"\n{'='*60}")
    print(f"  V1.5 vs V2  --  Double-Optuna Full Comparison")
    print(f"  n_trials={args.n_trials}  |  force_rerun={args.force_rerun}")
    print(f"  sentence-transformers={'available' if _SEMANTIC_AVAILABLE else 'NOT INSTALLED'}")
    print(f"{'='*60}")

    comparisons: list[DomainComparison] = []

    if args.synthetic:
        articles, gold, topic = _make_synthetic_corpus()
        print(f"\n[synthetic] topic='{topic}'")
        comparisons.append(_run_comparison(
            "synthetic_fake_news", articles, gold, topic,
            args.n_trials, args.force_rerun,
        ))

    elif args.domain:
        print(f"\n[1/1] Loading oracle: {args.domain}")
        try:
            articles, gold, topic = _load_oracle(args.domain)
        except FileNotFoundError as exc:
            print(f"\n  ERROR: {exc}")
            print(f"  Build it: python data/oracle/build_oracle.py --domain {args.domain}")
            sys.exit(1)
        print(f"  Topic: '{topic}'")
        cmp = _run_comparison(
            args.domain, articles, gold, topic,
            args.n_trials, args.force_rerun,
        )
        if cmp is None:
            sys.exit(1)
        comparisons.append(cmp)

    else:  # --all
        domain_ids = _all_domain_ids()
        for idx, domain_id in enumerate(domain_ids, 1):
            print(f"\n[{idx}/{len(domain_ids)}] Domain: {domain_id}")
            try:
                articles, gold, topic = _load_oracle(domain_id)
            except FileNotFoundError:
                print(f"  SKIP — oracle not found. "
                      f"Build: python data/oracle/build_oracle.py --domain {domain_id}")
                continue
            print(f"  Topic: '{topic}'")
            cmp = _run_comparison(
                domain_id, articles, gold, topic,
                args.n_trials, args.force_rerun,
            )
            if cmp is not None:
                comparisons.append(cmp)

    if not comparisons:
        print("\n  No domains processed. Exiting.")
        sys.exit(1)

    # ── Print results ──────────────────────────────────────────────────────
    for cmp in comparisons:
        _print_domain_detail(cmp)

    _print_summary_table(comparisons)   # only shown if multiple domains
    _print_winner(comparisons)

    print(f"\n  Columns: Def = DEFAULT_WEIGHTS | Lrn = Optuna best weights")
    print(f"  d(Opt) = Lrn-V2 minus Lrn-V1.5  |  d(V2-V1.5) on weights")
    print(f"\n  Thesis note: use these results in Chapter 4 (AutoML).")


if __name__ == "__main__":
    main()
