"""Demo Phase 3 — AutoML scorer with Optuna.

What this demo does
-------------------
1. Loads (or builds) the oracle corpus for a domain
2. Runs the Optuna optimisation (or loads cached weights if already done)
3. Prints a comparison table: DEFAULT_WEIGHTS vs LEARNED_WEIGHTS
4. Shows the NDCG@15 improvement (hypothesis H1)

Usage (from project root, venv activated)
-----------------------------------------
    # Quick demo using synthetic data (no API call needed)
    python demo_phase3.py --synthetic

    # Real oracle — requires building it first:
    #   python data/oracle/build_oracle.py --domain fake_news_detection
    python demo_phase3.py --domain fake_news_detection

    # Force Optuna to re-run even if weights already saved
    python demo_phase3.py --domain fake_news_detection --force-rerun

    # Adjust number of Optuna trials
    python demo_phase3.py --synthetic --n-trials 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WEIGHTS, OPTUNA_N_TRIALS
from src.schemas import Article, Author, SourceType
from src.scoring.automl_scorer import OptimizationResult, load_weights_for_topic, optimize_weights
from src.scoring.metrics import evaluate_all
from src.scoring.quality_scorer import score_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Synthetic oracle ─────────────────────────────────────────────────────────

def _make_synthetic_corpus(
    topic: str = "fake news detection",
    n_relevant: int = 15,
    n_background: int = 35,
) -> tuple[list[Article], dict[str, int]]:
    """Build a small synthetic corpus for demonstration without API calls.

    Relevant articles: topic words in title/abstract, higher citations.
    Background articles: unrelated topic.
    """
    articles: list[Article] = []
    gold: dict[str, int] = {}

    # Highly relevant (grade 2) — 5 papers
    for i in range(1, 6):
        a = Article(
            id=f"rel2_{i}",
            title=f"Fake News Detection Using Transformer-Based Models: Study {i}",
            abstract="We propose a deep learning approach for detecting fake news on social media "
                     "using BERT and attention mechanisms. Our method achieves state-of-the-art results.",
            year=2023,
            source=SourceType.OPENALEX,
            url=f"https://example.com/rel2_{i}",
            citation_count=200 + i * 50,
            concepts=["fake news", "detection", "natural language processing", "transformers"],
            journal_name="IEEE Transactions on Neural Networks",
            quartile="Q1",
            authors=[
                Author(name=f"Author A{i}", h_index=25 + i, citation_count=5000)
            ],
        )
        articles.append(a)
        gold[a.id] = 2

    # Relevant (grade 1) — 10 papers
    for i in range(1, 11):
        a = Article(
            id=f"rel1_{i}",
            title=f"Misinformation and Fake News on Social Media: Survey {i}",
            abstract="This paper surveys methods for detecting misinformation and fake news "
                     "in online social networks.",
            year=2022,
            source=SourceType.OPENALEX,
            url=f"https://example.com/rel1_{i}",
            citation_count=80 + i * 10,
            concepts=["fake news", "misinformation", "social media"],
            journal_name="ACM Computing Surveys",
            quartile="Q2",
            authors=[
                Author(name=f"Author B{i}", h_index=12 + i, citation_count=2000)
            ],
        )
        articles.append(a)
        gold[a.id] = 1

    # Background (grade 0) — unrelated
    unrelated_topics = [
        ("Image Segmentation with U-Net Architecture", "computer vision", "Medical Image Analysis"),
        ("Reinforcement Learning for Robotics Control", "reinforcement learning", "Robotics and Autonomous Systems"),
        ("Graph Neural Networks for Molecular Property", "graph neural networks", "Nature Machine Intelligence"),
        ("Speech Recognition with End-to-End Models", "speech recognition", "IEEE Signal Processing"),
        ("Federated Learning for Privacy Preservation", "federated learning", "Proceedings of NeurIPS"),
        ("Object Detection in Autonomous Driving", "computer vision", "CVPR"),
        ("Neural Architecture Search Techniques", "neural architecture search", "ICLR"),
    ]
    for i in range(n_background):
        t_idx = i % len(unrelated_topics)
        title_base, concept, journal = unrelated_topics[t_idx]
        a = Article(
            id=f"bg_{i}",
            title=f"{title_base} — Part {i}",
            abstract=f"This paper addresses {concept} using deep neural networks.",
            year=2022,
            source=SourceType.OPENALEX,
            url=f"https://example.com/bg_{i}",
            citation_count=30 + i * 5,
            concepts=[concept, "deep learning"],
            journal_name=journal,
            quartile="Q1" if i % 3 == 0 else "Q2",
            authors=[Author(name=f"Author C{i}", h_index=8 + i % 10, citation_count=500)],
        )
        articles.append(a)
        # grade 0 — not in gold_relevance (missing keys → treated as 0 by metrics)

    return articles, gold


# ── Ranking helper ────────────────────────────────────────────────────────────

def _ranked_ids(
    articles: list[Article],
    topic: str,
    weights: dict,
    use_semantic: bool = True,
) -> list[str]:
    scores = score_articles(articles, topic, weights, use_semantic=use_semantic)
    paired = sorted(zip(articles, scores), key=lambda p: p[1].final_score, reverse=True)
    return [a.id for a, _ in paired]


# ── Pretty printing ───────────────────────────────────────────────────────────

def _print_metrics_table(
    label_a: str,
    metrics_a: dict[str, float],
    label_b: str,
    metrics_b: dict[str, float],
) -> None:
    """Print a side-by-side comparison table (ASCII-only, Windows-safe)."""
    keys = list(metrics_a.keys())
    col_w = 14
    sep = "-" * 54

    header = f"{'Metric':<14} {label_a:>{col_w}} {label_b:>{col_w}}  {'Delta':>8}"
    print("\n" + sep)
    print(header)
    print(sep)
    for k in keys:
        a = metrics_a[k]
        b = metrics_b[k]
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"{k:<14} {a:>{col_w}.4f} {b:>{col_w}.4f}  {sign}{delta:>7.4f}")
    print(sep)


def _print_weights_table(
    label_a: str,
    weights_a: dict[str, float],
    label_b: str,
    weights_b: dict[str, float],
) -> None:
    """Print a side-by-side weight comparison (ASCII-only, Windows-safe)."""
    keys = ["venue", "authors", "impact", "relevance"]
    sep = "-" * 48
    print(f"\n{'Weight':<14} {label_a:>16} {label_b:>16}")
    print(sep)
    for k in keys:
        print(f"{k:<14} {weights_a.get(k, 0):>16.4f} {weights_b.get(k, 0):>16.4f}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 AutoML demo — Optuna weight optimisation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true",
                      help="Use synthetic corpus (no API calls)")
    mode.add_argument("--domain",
                      help="Domain ID from domains_config.json "
                           "(oracle must be pre-built)")
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS,
                        help=f"Number of Optuna trials (default: {OPTUNA_N_TRIALS})")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Delete existing study and start fresh")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Use V1.5 keyword/bigram relevance instead of V2 "
                             "semantic embeddings (sentence-transformers). "
                             "Useful for V1.5 vs V2 comparison experiments.")
    args = parser.parse_args()
    use_semantic = not args.no_semantic

    # ── Load corpus ────────────────────────────────────────────────────────
    if args.synthetic:
        topic = "fake news detection"
        print(f"\n{'='*60}")
        print(f"  Phase 3 AutoML Demo - Synthetic Corpus")
        print(f"  Topic: '{topic}'")
        print(f"  Relevance: {'V1.5 keyword/bigram' if args.no_semantic else 'V2 semantic embeddings'}")
        print(f"{'='*60}")
        articles, gold = _make_synthetic_corpus(topic)
        print(f"\n  Corpus: {len(articles)} articles "
              f"({sum(1 for g in gold.values() if g == 2)} grade-2, "
              f"{sum(1 for g in gold.values() if g == 1)} grade-1, "
              f"{len(articles) - len(gold)} grade-0)")
    else:
        from data.oracle.build_oracle import load_oracle
        domain_id = args.domain
        try:
            articles, gold = load_oracle(domain_id)
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            sys.exit(1)

        # Load topic from config
        config_path = Path("data/oracle/domains_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        domain_cfg = next(d for d in config["domains"] if d["id"] == domain_id)
        topic = domain_cfg["topic"]

        print(f"\n{'='*60}")
        print(f"  Phase 3 AutoML Demo - Domain: {domain_id}")
        print(f"  Topic: '{topic}'")
        print(f"  Relevance: {'V1.5 keyword/bigram' if args.no_semantic else 'V2 semantic embeddings'}")
        print(f"{'='*60}")
        print(f"\n  Corpus: {len(articles)} articles "
              f"({sum(1 for g in gold.values() if g == 2)} grade-2, "
              f"{sum(1 for g in gold.values() if g == 1)} grade-1, "
              f"{len(articles) - len(gold)} grade-0 / background)")

    # ── Baseline (DEFAULT_WEIGHTS) ─────────────────────────────────────────
    print(f"\n[1/3] Computing baseline (DEFAULT_WEIGHTS)...")
    ranked_baseline = _ranked_ids(articles, topic, DEFAULT_WEIGHTS, use_semantic=use_semantic)
    metrics_baseline = evaluate_all(ranked_baseline, gold)
    print(f"      NDCG@15 = {metrics_baseline['ndcg@15']:.4f}")

    # ── Optuna optimisation ────────────────────────────────────────────────
    print(f"\n[2/3] Running Optuna ({args.n_trials} trials)...")
    result: OptimizationResult = optimize_weights(
        articles,
        gold,
        topic,
        n_trials=args.n_trials,
        force_rerun=args.force_rerun,
        use_semantic=use_semantic,
    )
    print(f"      Completed {result.n_trials_completed} trials "
          f"in {result.duration_seconds:.1f}s")
    print(f"      NDCG@15 = {result.best_ndcg_at_15:.4f} "
          f"({result.improvement_pct:+.1f}% vs baseline)")

    # ── Comparison ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Full metrics comparison")
    ranked_learned = _ranked_ids(articles, topic, result.best_weights, use_semantic=use_semantic)
    metrics_learned = evaluate_all(ranked_learned, gold)

    _print_metrics_table(
        "DEFAULT", metrics_baseline,
        "LEARNED", metrics_learned,
    )

    _print_weights_table(
        "DEFAULT", DEFAULT_WEIGHTS,
        "LEARNED", result.best_weights,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AutoML Summary")
    print(f"{'='*60}")
    print(f"  Hypothesis H1: Optuna >= +5% NDCG@15")
    if result.improvement_pct >= 5.0:
        print(f"  [CONFIRMED]     improvement = {result.improvement_pct:+.1f}%")
    elif result.improvement_pct > 0:
        print(f"  [PARTIAL]       improvement = {result.improvement_pct:+.1f}% (< 5% threshold)")
    else:
        print(f"  [NOT CONFIRMED] improvement = {result.improvement_pct:+.1f}%")
    print(f"  Weights saved: {'Yes' if result.weights_saved else 'No (below threshold)'}")
    print(f"{'='*60}")
    import sys; sys.stdout.flush()


if __name__ == "__main__":
    main()
