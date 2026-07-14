"""Tests for the quality scorer orchestrator."""

from __future__ import annotations

import pytest

from src.schemas import Article, Author, QualityScore, SourceType
from src.scoring.quality_scorer import (
    _validate_weights,
    filter_top_n,
    score_article,
    score_articles,
)


def _make_article(idx: int = 1, **overrides) -> Article:
    """Build an Article with sensible defaults."""
    defaults = {
        "id": f"test_{idx}",
        "title": f"Test article {idx}",
        "year": 2023,
        "source": SourceType.OPENALEX,
        "url": "https://example.com",
    }
    defaults.update(overrides)
    return Article(**defaults)


# ============================================================
# Weight validation
# ============================================================


class TestValidateWeights:
    def test_valid_weights_summing_to_one(self):
        result = _validate_weights({
            "venue": 0.25, "authors": 0.25, "impact": 0.25, "relevance": 0.25
        })
        assert sum(result.values()) == pytest.approx(1.0)

    def test_normalizes_weights_6keys(self):
        # All 1s across 6 keys should normalize to 1/6 each
        result = _validate_weights({
            "venue": 1.0, "authors": 1.0, "impact": 1.0,
            "velocity": 1.0, "recency": 1.0, "relevance": 1.0,
        })
        for v in result.values():
            assert v == pytest.approx(1 / 6, abs=1e-6)

    def test_normalizes_weights_4keys_backward_compat(self):
        # Old 4-key input: velocity/recency default to 0, required keys normalized
        result = _validate_weights({
            "venue": 1.0, "authors": 1.0, "impact": 1.0, "relevance": 1.0
        })
        # Required keys: each should be 0.25 (total = 4.0, optionals = 0)
        for k in ("venue", "authors", "impact", "relevance"):
            assert result[k] == pytest.approx(0.25)
        assert result["velocity"] == pytest.approx(0.0)
        assert result["recency"] == pytest.approx(0.0)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing"):
            _validate_weights({"venue": 1.0, "authors": 1.0, "impact": 1.0})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_weights({
                "venue": -0.1, "authors": 0.5, "impact": 0.3, "relevance": 0.3
            })

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="zero"):
            _validate_weights({
                "venue": 0, "authors": 0, "impact": 0, "relevance": 0
            })


# ============================================================
# score_article
# ============================================================


class TestScoreArticle:
    def test_returns_validated_quality_score(self):
        article = _make_article(
            title="Fake news detection methods",
            authors=[Author(name="A", h_index=30)],
            citation_count=100,
            quartile="Q1",
            journal_name="Top Journal",
        )
        score = score_article(article, "fake news detection", current_year=2024)

        assert isinstance(score, QualityScore)
        assert score.article_id == "test_1"
        assert 0 <= score.final_score <= 1
        assert 0 <= score.venue_score <= 1
        assert 0 <= score.authors_score <= 1
        assert 0 <= score.impact_score <= 1
        assert 0 <= score.relevance_score <= 1

    def test_high_quality_article_scores_well(self):
        article = _make_article(
            title="Fake news detection: a deep learning approach",
            authors=[Author(name="A", h_index=80)],
            citation_count=500,
            year=2020,
            quartile="Q1",
            journal_name="Nature",
            concepts=["Fake news", "Deep learning"],
        )
        score = score_article(article, "fake news detection", current_year=2024)
        # All 4 features should be strong
        assert score.final_score > 0.6

    def test_weights_used_recorded(self):
        custom = {"venue": 0.4, "authors": 0.2, "impact": 0.2, "relevance": 0.2}
        article = _make_article(quartile="Q1", journal_name="J")
        score = score_article(article, "topic", weights=custom, current_year=2024)
        # weights_used should be normalized
        assert sum(score.weights_used.values()) == pytest.approx(1.0)
        # And keep the relative proportions
        assert score.weights_used["venue"] == 0.4

    def test_relevance_extremes_dominate(self):
        """Two identical articles, only relevance differs - relevance-heavy
        weights should produce different scores."""
        relevant = _make_article(idx=1, title="fake news detection")
        irrelevant = _make_article(idx=2, title="quantum chemistry")
        weights = {
            "venue": 0.05, "authors": 0.05, "impact": 0.05, "relevance": 0.85
        }
        s1 = score_article(relevant, "fake news detection",
                           weights=weights, current_year=2024)
        s2 = score_article(irrelevant, "fake news detection",
                           weights=weights, current_year=2024)
        assert s1.final_score > s2.final_score


# ============================================================
# score_articles + filter_top_n
# ============================================================


class TestScoreAndFilter:
    def test_score_articles_preserves_order(self):
        articles = [_make_article(idx=i) for i in range(5)]
        scores = score_articles(articles, "topic", current_year=2024)
        for art, sc in zip(articles, scores):
            assert sc.article_id == art.id

    def test_filter_top_n_returns_n_items(self):
        articles = [
            _make_article(idx=i, citation_count=i * 100, year=2020)
            for i in range(1, 11)
        ]
        scores = score_articles(articles, "topic", current_year=2024)
        top, top_scores = filter_top_n(articles, scores, n=3)
        assert len(top) == 3
        assert len(top_scores) == 3

    def test_filter_top_n_sorted_descending(self):
        articles = [
            _make_article(idx=i, citation_count=i * 100, year=2020)
            for i in range(1, 11)
        ]
        scores = score_articles(articles, "topic", current_year=2024)
        _, top_scores = filter_top_n(articles, scores, n=10)
        # Should be sorted descending
        for i in range(len(top_scores) - 1):
            assert top_scores[i].final_score >= top_scores[i + 1].final_score

    def test_filter_top_n_more_than_available(self):
        articles = [_make_article(idx=i) for i in range(3)]
        scores = score_articles(articles, "topic", current_year=2024)
        top, _ = filter_top_n(articles, scores, n=100)
        assert len(top) == 3

    def test_filter_empty_input(self):
        top, top_scores = filter_top_n([], [], n=10)
        assert top == []
        assert top_scores == []

    def test_filter_mismatched_lengths_raises(self):
        articles = [_make_article(idx=i) for i in range(3)]
        scores = []
        with pytest.raises(ValueError, match="same length"):
            filter_top_n(articles, scores, n=10)
