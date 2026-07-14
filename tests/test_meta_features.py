"""Tests for Phase 8a — DomainMetaFeatures extraction.

All tests use small in-memory Article objects — no network calls, no disk I/O.
"""

from __future__ import annotations

import pytest

from src.metalearning.meta_features import (
    FEATURE_NAMES,
    DomainMetaFeatures,
    _gini,
    _median,
    _std,
    extract_meta_features,
)
from src.schemas import Article, Author, SourceType


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_article(
    id: str,
    year: int = 2020,
    citation_count: int = 50,
    h_index: int | None = 15,
    quartile: str = "Q2",
    concepts: list[str] | None = None,
    authors: list[Author] | None = None,
) -> Article:
    if authors is None:
        authors = [Author(name=f"Author_{id}", h_index=h_index, citation_count=500)]
    return Article(
        id=id,
        title=f"Paper {id}",
        abstract=f"Abstract of paper {id}.",
        year=year,
        source=SourceType.OPENALEX,
        url=f"https://example.com/{id}",
        citation_count=citation_count,
        concepts=concepts or ["machine learning", "deep learning"],
        journal_name="Test Journal",
        quartile=quartile,
        authors=authors,
    )


def _small_corpus() -> tuple[list[Article], dict[str, int], str]:
    """6 articles: 2 grade-2, 2 grade-1, 2 background."""
    articles = [
        _make_article("a1", year=2022, citation_count=200, h_index=30, quartile="Q1",
                      concepts=["fake news", "detection"]),
        _make_article("a2", year=2023, citation_count=150, h_index=25, quartile="Q1",
                      concepts=["fake news", "social media"]),
        _make_article("b1", year=2020, citation_count=80, h_index=10, quartile="Q2",
                      concepts=["misinformation", "fake news"]),
        _make_article("b2", year=2021, citation_count=60, h_index=8, quartile="Q2",
                      concepts=["rumour", "social network"]),
        _make_article("bg1", year=2019, citation_count=20, h_index=5, quartile="Q3",
                      concepts=["image segmentation", "medical"]),
        _make_article("bg2", year=2018, citation_count=10, h_index=4, quartile="Q3",
                      concepts=["reinforcement learning", "robotics"]),
    ]
    gold = {"a1": 2, "a2": 2, "b1": 1, "b2": 1}
    return articles, gold, "fake news detection"


# ── Helper function tests ─────────────────────────────────────────────────────


class TestGini:
    def test_equal_values_returns_zero(self):
        assert _gini([10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=1e-6)

    def test_single_nonzero_higher_than_equal(self):
        # Concentrated distribution is more unequal than a flat one
        concentrated = _gini([0.0, 0.0, 100.0])
        flat = _gini([33.0, 33.0, 34.0])
        assert concentrated > flat
        assert concentrated > 0.5  # for n=3, max gini = (n-1)/n = 0.667

    def test_empty_returns_zero(self):
        assert _gini([]) == 0.0

    def test_all_zeros_returns_zero(self):
        assert _gini([0.0, 0.0, 0.0]) == 0.0

    def test_result_in_range(self):
        result = _gini([1.0, 5.0, 10.0, 50.0, 200.0])
        assert 0.0 <= result <= 1.0


class TestMedian:
    def test_odd_length(self):
        assert _median([1.0, 3.0, 5.0]) == 3.0

    def test_even_length(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_single_element(self):
        assert _median([7.0]) == 7.0

    def test_empty_returns_zero(self):
        assert _median([]) == 0.0


class TestStd:
    def test_constant_series_returns_zero(self):
        assert _std([5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] -> population std = 2.0
        assert _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0, abs=1e-6)

    def test_single_element_returns_zero(self):
        assert _std([42.0]) == 0.0


# ── extract_meta_features tests ───────────────────────────────────────────────


class TestExtractMetaFeatures:
    def test_returns_correct_type(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        assert isinstance(feats, DomainMetaFeatures)

    def test_domain_metadata_stored(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("my_domain", topic, articles, gold)
        assert feats.domain_id == "my_domain"
        assert feats.topic == topic
        assert feats.corpus_size == 6

    def test_temporal_features_in_range(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        assert 2000 <= feats.median_year <= 2030
        assert 0.0 <= feats.pct_recent <= 1.0
        assert feats.year_std >= 0.0

    def test_citation_features_in_range(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        assert 0.0 <= feats.citation_gini <= 1.0
        assert feats.citation_median >= 0.0
        assert 0.0 <= feats.pct_high_cited <= 1.0

    def test_author_features_in_range(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        assert 0.0 <= feats.unique_author_ratio <= 1.0
        assert feats.mean_h_index >= 0.0
        assert 0.0 <= feats.pct_high_hindex <= 1.0

    def test_pct_q1_correct(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        # 2 of 6 articles are Q1
        assert feats.pct_q1 == pytest.approx(2 / 6, abs=1e-6)

    def test_grade_ratios_correct(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        # 2 grade-2, 2 grade-1, 2 background out of 6
        assert feats.grade2_ratio == pytest.approx(2 / 6, abs=1e-6)
        assert feats.gold_ratio == pytest.approx(4 / 6, abs=1e-6)

    def test_topic_concept_overlap(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        # topic words: "fake", "news", "detection" (len>2: all three)
        # a1: ["fake news", "detection"] -> contains "fake", "news", "detection" -> yes
        # a2: ["fake news", "social media"] -> yes
        # b1: ["misinformation", "fake news"] -> yes
        # b2: ["rumour", "social network"] -> no
        # bg1: ["image segmentation", "medical"] -> no
        # bg2: ["reinforcement learning", "robotics"] -> no
        # 3/6 = 0.5
        assert feats.topic_concept_overlap == pytest.approx(0.5, abs=1e-6)

    def test_as_vector_length(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        vec = feats.as_vector()
        assert len(vec) == len(FEATURE_NAMES)

    def test_as_vector_order_matches_feature_names(self):
        articles, gold, topic = _small_corpus()
        feats = extract_meta_features("test", topic, articles, gold)
        vec = feats.as_vector()
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == pytest.approx(getattr(feats, name), abs=1e-9)

    def test_raises_on_empty_corpus(self):
        with pytest.raises(ValueError, match="Empty corpus"):
            extract_meta_features("test", "fake news", [], {})


class TestEdgeCases:
    def test_median_year_computed_correctly(self):
        """Median year is the middle value of sorted years."""
        articles = [
            _make_article("x1", year=2018),
            _make_article("x2", year=2020),
            _make_article("x3", year=2022),
        ]
        gold = {"x1": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.median_year == pytest.approx(2020.0)

    def test_articles_with_none_h_index(self):
        """None h-index is skipped from mean but authors still count."""
        articles = [
            _make_article("y1", h_index=None),
            _make_article("y2", h_index=20),
        ]
        gold = {"y1": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.mean_h_index == pytest.approx(20.0)

    def test_all_same_citations_gini_zero(self):
        articles = [_make_article(f"z{i}", citation_count=100) for i in range(5)]
        gold = {"z0": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.citation_gini == pytest.approx(0.0, abs=1e-6)

    def test_no_q1_articles(self):
        articles = [_make_article(f"q{i}", quartile="Q3") for i in range(4)]
        gold = {"q0": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.pct_q1 == 0.0

    def test_all_q1_articles(self):
        articles = [_make_article(f"q{i}", quartile="Q1") for i in range(4)]
        gold = {"q0": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.pct_q1 == 1.0

    def test_no_high_cited(self):
        articles = [_make_article(f"c{i}", citation_count=10) for i in range(5)]
        gold = {"c0": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.pct_high_cited == 0.0

    def test_unique_author_ratio_all_same(self):
        """When all articles have the same first-author name, ratio = 1/n."""
        articles = [
            Article(
                id=f"s{i}", title=f"P{i}", abstract="...",
                year=2020, source=SourceType.OPENALEX,
                url=f"https://example.com/s{i}",
                citation_count=50,
                authors=[Author(name="Same Author", h_index=10, citation_count=500)],
            )
            for i in range(4)
        ]
        gold = {"s0": 1}
        feats = extract_meta_features("test", "topic", articles, gold)
        assert feats.unique_author_ratio == pytest.approx(1 / 4)
