"""Information-retrieval evaluation metrics for the AutoML scorer.

All functions operate on:
    ranked_ids    — article IDs in ranked order (best first, as returned by
                    the pipeline after quality scoring)
    gold_relevance — ground-truth dict  {article_id: relevance_grade}
                    where relevance_grade ∈ {0, 1, 2}
                      0 = not relevant
                      1 = relevant        (cited in 1 survey)
                      2 = highly relevant (cited in ≥2 surveys / key paper)

Primary metric (used as Optuna objective):
    NDCG@15 — Normalised Discounted Cumulative Gain at rank 15.
    Rewards retrieving relevant articles AND placing them at the top.

Secondary metrics (for reporting in the thesis):
    Precision@k, Recall@k, MAP.

Articles NOT present in gold_relevance are treated as relevance=0 (unkown /
not relevant). This is the standard "incomplete judgements" assumption.
"""

from __future__ import annotations

import math


# ── Internal helpers ──────────────────────────────────────────────────────────

def _gain(relevance: int) -> float:
    """Standard NDCG gain formula:  gain = 2^rel − 1.

    rel=0 → 0.0   (not relevant — contributes nothing)
    rel=1 → 1.0   (relevant)
    rel=2 → 3.0   (highly relevant — 3× more valuable than rel=1)
    """
    return 2.0 ** relevance - 1.0


def _dcg(ranked_ids: list[str], gold_relevance: dict[str, int], k: int) -> float:
    """Raw Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for rank, aid in enumerate(ranked_ids[:k], start=1):
        rel = gold_relevance.get(aid, 0)
        dcg += _gain(rel) / math.log2(rank + 1)   # discount by log2(rank+1)
    return dcg


def _idcg(gold_relevance: dict[str, int], k: int) -> float:
    """Ideal DCG@k — DCG of the perfect ranking (best articles first)."""
    sorted_gains = sorted((_gain(r) for r in gold_relevance.values()), reverse=True)
    idcg = 0.0
    for rank, gain in enumerate(sorted_gains[:k], start=1):
        idcg += gain / math.log2(rank + 1)
    return idcg


# ── Public metrics ────────────────────────────────────────────────────────────

def ndcg_at_k(
    ranked_ids: list[str],
    gold_relevance: dict[str, int],
    k: int = 15,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    THE primary metric for Optuna optimisation (hypothesis H1).
    Range: [0, 1].  1.0 = perfect ranking.  0.0 = all top-k are irrelevant.

    Why NDCG and not Precision?
        Precision@k only counts hits, ignoring rank position.
        NDCG penalises a relevant article at rank 10 more than at rank 2,
        which matches how a researcher actually uses a ranked list.

    Args:
        ranked_ids:     Article IDs sorted by the scorer (best first).
        gold_relevance: Ground-truth relevance grades.
        k:              Rank cutoff (default 15 — typical "first page").

    Returns:
        NDCG@k ∈ [0, 1].  Returns 0.0 when no relevant articles exist.
    """
    idcg = _idcg(gold_relevance, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(ranked_ids, gold_relevance, k) / idcg


def precision_at_k(
    ranked_ids: list[str],
    gold_relevance: dict[str, int],
    k: int = 10,
    min_relevance: int = 1,
) -> float:
    """Precision@k: fraction of top-k results that are relevant.

    Args:
        ranked_ids:     Article IDs sorted by the scorer (best first).
        gold_relevance: Ground-truth relevance grades.
        k:              Rank cutoff.
        min_relevance:  Minimum grade to count as relevant (default 1).

    Returns:
        P@k ∈ [0, 1].
    """
    if not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for aid in top_k if gold_relevance.get(aid, 0) >= min_relevance)
    return hits / len(top_k)


def recall_at_k(
    ranked_ids: list[str],
    gold_relevance: dict[str, int],
    k: int = 30,
    min_relevance: int = 1,
) -> float:
    """Recall@k: fraction of all relevant articles found in top-k.

    Useful to check the pipeline does not miss important papers entirely.

    Args:
        ranked_ids:     Article IDs sorted by the scorer (best first).
        gold_relevance: Ground-truth relevance grades.
        k:              Rank cutoff.
        min_relevance:  Minimum grade to count as relevant.

    Returns:
        R@k ∈ [0, 1].  Returns 0.0 when no relevant articles exist.
    """
    total = sum(1 for r in gold_relevance.values() if r >= min_relevance)
    if total == 0:
        return 0.0
    found = sum(
        1 for aid in ranked_ids[:k]
        if gold_relevance.get(aid, 0) >= min_relevance
    )
    return found / total


def mean_average_precision(
    ranked_ids: list[str],
    gold_relevance: dict[str, int],
    min_relevance: int = 1,
) -> float:
    """Mean Average Precision (MAP) over the full ranked list.

    Computes a precision value at each rank where a relevant article appears,
    then averages them.  Sensitive to both early precision and recall.

    Args:
        ranked_ids:     Article IDs sorted by the scorer (best first).
        gold_relevance: Ground-truth relevance grades.
        min_relevance:  Minimum grade to count as relevant.

    Returns:
        MAP ∈ [0, 1].  Returns 0.0 when no relevant articles exist.
    """
    total = sum(1 for r in gold_relevance.values() if r >= min_relevance)
    if total == 0:
        return 0.0

    cumulative = 0.0
    n_seen = 0
    for rank, aid in enumerate(ranked_ids, start=1):
        if gold_relevance.get(aid, 0) >= min_relevance:
            n_seen += 1
            cumulative += n_seen / rank   # precision at this rank

    return cumulative / total


def evaluate_all(
    ranked_ids: list[str],
    gold_relevance: dict[str, int],
) -> dict[str, float]:
    """Compute all metrics at once.  Convenience wrapper for reporting.

    Returns a dict with keys:
        ndcg@15, ndcg@10, p@5, p@10, recall@30, map
    """
    return {
        "ndcg@15":   ndcg_at_k(ranked_ids, gold_relevance, k=15),
        "ndcg@10":   ndcg_at_k(ranked_ids, gold_relevance, k=10),
        "p@5":       precision_at_k(ranked_ids, gold_relevance, k=5),
        "p@10":      precision_at_k(ranked_ids, gold_relevance, k=10),
        "recall@30": recall_at_k(ranked_ids, gold_relevance, k=30),
        "map":       mean_average_precision(ranked_ids, gold_relevance),
    }
