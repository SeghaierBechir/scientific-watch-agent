"""Unit tests for src/scoring/metrics.py.

All tests are pure in-memory — no network calls, no file I/O.
Coverage: ndcg_at_k, precision_at_k, recall_at_k, mean_average_precision,
evaluate_all, and all their edge cases.
"""

from __future__ import annotations

import math
import pytest

from src.scoring.metrics import (
    evaluate_all,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_gold() -> dict[str, int]:
    """5 articles: a1/a2 highly relevant, a3 relevant, a4/a5 not relevant."""
    return {"a1": 2, "a2": 2, "a3": 1, "a4": 0, "a5": 0}


@pytest.fixture
def perfect_ranking(simple_gold) -> list[str]:
    """Articles ordered by descending relevance (ideal ranking)."""
    return ["a1", "a2", "a3", "a4", "a5"]


@pytest.fixture
def worst_ranking(simple_gold) -> list[str]:
    """All non-relevant before relevant (worst possible ranking)."""
    return ["a4", "a5", "a3", "a2", "a1"]


# ── NDCG@k ───────────────────────────────────────────────────────────────────

class TestNdcgAtK:

    def test_perfect_ranking_returns_1(self, perfect_ranking, simple_gold):
        score = ndcg_at_k(perfect_ranking, simple_gold, k=5)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_empty_ranked_ids_returns_0(self, simple_gold):
        assert ndcg_at_k([], simple_gold, k=15) == 0.0

    def test_no_relevant_articles_returns_0(self):
        gold = {"a1": 0, "a2": 0}
        assert ndcg_at_k(["a1", "a2"], gold, k=5) == 0.0

    def test_worst_ranking_is_below_1(self, worst_ranking, simple_gold):
        score = ndcg_at_k(worst_ranking, simple_gold, k=5)
        assert 0.0 <= score < 1.0

    def test_perfect_beats_worst(self, perfect_ranking, worst_ranking, simple_gold):
        perfect = ndcg_at_k(perfect_ranking, simple_gold, k=5)
        worst = ndcg_at_k(worst_ranking, simple_gold, k=5)
        assert perfect > worst

    def test_unknown_articles_treated_as_relevance_0(self, simple_gold):
        # "x_unknown" has no entry in gold → treated as 0
        ranked = ["x_unknown", "a1", "a2"]
        score_with_unknown = ndcg_at_k(ranked, simple_gold, k=3)
        # Pushing a1 down to rank 2 reduces NDCG vs perfect
        score_perfect = ndcg_at_k(["a1", "a2", "x_unknown"], simple_gold, k=3)
        assert score_with_unknown < score_perfect

    def test_grade_2_more_valuable_than_grade_1(self):
        gold = {"high": 2, "low": 1}
        ranked_high_first = ["high", "low"]
        ranked_low_first = ["low", "high"]
        assert ndcg_at_k(ranked_high_first, gold, k=2) > \
               ndcg_at_k(ranked_low_first, gold, k=2)

    def test_k_cutoff_respected(self, simple_gold):
        # Relevant article beyond k=1 is ignored
        ranked = ["a4", "a1", "a2"]   # a4 (grade 0) at rank 1
        score_k1 = ndcg_at_k(ranked, simple_gold, k=1)
        assert score_k1 == 0.0   # only rank-1 counts; it's grade-0

    def test_result_range(self, simple_gold):
        ranked = ["a3", "a1", "a4", "a2", "a5"]
        score = ndcg_at_k(ranked, simple_gold, k=5)
        assert 0.0 <= score <= 1.0

    def test_manual_dcg_calculation(self):
        """Verify formula against hand-computed values.

        gold = {a1: 2, a2: 1}
        ranked = [a1, a2]
        gain(2) = 3, gain(1) = 1
        DCG = 3/log2(2) + 1/log2(3) = 3/1 + 1/1.585 ≈ 3.631
        IDCG = same (it's the ideal ordering)
        NDCG = 1.0
        """
        gold = {"a1": 2, "a2": 1}
        score = ndcg_at_k(["a1", "a2"], gold, k=2)
        assert score == pytest.approx(1.0)

    def test_default_k_is_15(self, simple_gold):
        """Calling without k uses k=15 by default."""
        ranked = ["a1", "a2", "a3"]
        score_default = ndcg_at_k(ranked, simple_gold)
        score_k15 = ndcg_at_k(ranked, simple_gold, k=15)
        assert score_default == score_k15


# ── Precision@k ──────────────────────────────────────────────────────────────

class TestPrecisionAtK:

    def test_all_relevant_returns_1(self):
        gold = {"a1": 1, "a2": 2}
        assert precision_at_k(["a1", "a2"], gold, k=2) == pytest.approx(1.0)

    def test_none_relevant_returns_0(self):
        gold = {"a1": 0, "a2": 0}
        assert precision_at_k(["a1", "a2"], gold, k=2) == 0.0

    def test_half_relevant(self):
        gold = {"a1": 1, "a2": 0}
        assert precision_at_k(["a1", "a2"], gold, k=2) == pytest.approx(0.5)

    def test_empty_ranked_ids_returns_0(self):
        assert precision_at_k([], {"a1": 1}, k=5) == 0.0

    def test_k_larger_than_list_uses_actual_length(self):
        gold = {"a1": 1}
        # k=10 but only 1 article in list
        score = precision_at_k(["a1"], gold, k=10)
        assert score == pytest.approx(1.0)

    def test_min_relevance_grade_2_only(self):
        gold = {"a1": 1, "a2": 2}
        # grade 1 doesn't count when min_relevance=2
        p = precision_at_k(["a1", "a2"], gold, k=2, min_relevance=2)
        assert p == pytest.approx(0.5)   # only a2 counts

    def test_unknown_article_counts_as_not_relevant(self):
        gold = {"a1": 1}
        p = precision_at_k(["x_unknown", "a1"], gold, k=2)
        assert p == pytest.approx(0.5)


# ── Recall@k ─────────────────────────────────────────────────────────────────

class TestRecallAtK:

    def test_all_found_returns_1(self):
        gold = {"a1": 1, "a2": 2}
        assert recall_at_k(["a1", "a2", "a3"], gold, k=2) == pytest.approx(1.0)

    def test_none_found_returns_0(self):
        gold = {"a1": 1, "a2": 2}
        assert recall_at_k(["a3", "a4"], gold, k=2) == 0.0

    def test_no_relevant_articles_returns_0(self):
        gold = {"a1": 0, "a2": 0}
        assert recall_at_k(["a1", "a2"], gold, k=2) == 0.0

    def test_partial_recall(self):
        gold = {"a1": 1, "a2": 1, "a3": 1}
        # Only 2 of 3 relevant articles found in top-2
        r = recall_at_k(["a1", "a2"], gold, k=2)
        assert r == pytest.approx(2 / 3)

    def test_k_cutoff_respected(self):
        gold = {"a1": 1, "a2": 1}
        # a2 is at rank 3, beyond k=2
        r = recall_at_k(["a1", "x", "a2"], gold, k=2)
        assert r == pytest.approx(0.5)  # only a1 found within k=2


# ── MAP ───────────────────────────────────────────────────────────────────────

class TestMeanAveragePrecision:

    def test_perfect_ranking(self, perfect_ranking, simple_gold):
        # Relevant articles: a1 (grade2), a2 (grade2), a3 (grade1)
        # At rank 1: a1 hit → P=1/1=1.0
        # At rank 2: a2 hit → P=2/2=1.0
        # At rank 3: a3 hit → P=3/3=1.0
        # MAP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        result = mean_average_precision(perfect_ranking, simple_gold)
        assert result == pytest.approx(1.0)

    def test_no_relevant_articles_returns_0(self):
        gold = {"a1": 0, "a2": 0}
        assert mean_average_precision(["a1", "a2"], gold) == 0.0

    def test_empty_ranked_ids_returns_0(self):
        assert mean_average_precision([], {"a1": 1}) == 0.0

    def test_all_relevant_late_has_low_map(self):
        # Only 1 relevant article, placed last
        gold = {"a1": 1}
        ranked = ["x1", "x2", "x3", "x4", "a1"]
        map_score = mean_average_precision(ranked, gold)
        # AP = 1/5 = 0.2
        assert map_score == pytest.approx(1 / 5)

    def test_relevant_first_has_high_map(self):
        gold = {"a1": 1}
        ranked = ["a1", "x1", "x2"]
        assert mean_average_precision(ranked, gold) == pytest.approx(1.0)

    def test_map_range(self, simple_gold):
        ranked = ["a4", "a1", "a5", "a2", "a3"]
        result = mean_average_precision(ranked, simple_gold)
        assert 0.0 <= result <= 1.0

    def test_min_relevance_respected(self):
        gold = {"a1": 1, "a2": 2}
        # With min_relevance=2, only a2 counts as relevant
        map_r2 = mean_average_precision(["a1", "a2"], gold, min_relevance=2)
        # a2 at rank 2 → AP = 1/2
        assert map_r2 == pytest.approx(0.5)


# ── evaluate_all ─────────────────────────────────────────────────────────────

class TestEvaluateAll:

    def test_returns_all_expected_keys(self, perfect_ranking, simple_gold):
        result = evaluate_all(perfect_ranking, simple_gold)
        assert set(result.keys()) == {"ndcg@15", "ndcg@10", "p@5", "p@10", "recall@30", "map"}

    def test_all_values_in_range(self, perfect_ranking, simple_gold):
        result = evaluate_all(perfect_ranking, simple_gold)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_perfect_ranking_gives_high_scores(self, perfect_ranking, simple_gold):
        result = evaluate_all(perfect_ranking, simple_gold)
        assert result["ndcg@15"] == pytest.approx(1.0)
        assert result["ndcg@10"] == pytest.approx(1.0)
        assert result["map"] == pytest.approx(1.0)

    def test_consistent_with_individual_functions(self, simple_gold):
        ranked = ["a3", "a4", "a1", "a2", "a5"]
        result = evaluate_all(ranked, simple_gold)
        assert result["ndcg@15"] == pytest.approx(ndcg_at_k(ranked, simple_gold, k=15))
        assert result["ndcg@10"] == pytest.approx(ndcg_at_k(ranked, simple_gold, k=10))
        assert result["p@5"]     == pytest.approx(precision_at_k(ranked, simple_gold, k=5))
        assert result["p@10"]    == pytest.approx(precision_at_k(ranked, simple_gold, k=10))
        assert result["recall@30"] == pytest.approx(recall_at_k(ranked, simple_gold, k=30))
        assert result["map"]     == pytest.approx(mean_average_precision(ranked, simple_gold))
