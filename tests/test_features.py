"""Tests for the 6 feature scoring modules (Level 1-2-3).

We use parametrized tests to cover edge cases concisely.
"""

from __future__ import annotations

import math

import pytest

from src.features.authors_score import authors_score
from src.features.impact_score import impact_score
from src.features.recency_score import recency_score
from src.features.relevance_score import relevance_score
from src.features.velocity_score import velocity_score
from src.features.venue_score import venue_score
from src.schemas import Article, Author, SourceType


# ============================================================
# Helpers
# ============================================================


def _make_article(**overrides) -> Article:
    """Build an Article with sensible defaults, override what we test."""
    defaults = {
        "id": "test_1",
        "title": "Default test title",
        "year": 2023,
        "source": SourceType.OPENALEX,
        "url": "https://example.com",
    }
    defaults.update(overrides)
    return Article(**defaults)


# ============================================================
# Venue score tests
# ============================================================


class TestVenueScore:
    @pytest.mark.parametrize("quartile,expected", [
        ("Q1", 1.00),
        ("Q2", 0.70),
        ("Q3", 0.40),
        ("Q4", 0.20),
    ])
    def test_quartile_mapping(self, quartile, expected):
        article = _make_article(journal_name="Some Journal", quartile=quartile)
        assert venue_score(article) == expected

    def test_preprint_gets_neutral_score(self):
        article = _make_article(is_preprint=True, journal_name="arXiv")
        assert venue_score(article) == 0.50

    def test_preprint_overrides_quartile(self):
        # Even if quartile is set, preprints get neutral
        article = _make_article(
            is_preprint=True, journal_name="arXiv", quartile="Q1"
        )
        assert venue_score(article) == 0.50

    def test_no_journal_low_baseline(self):
        article = _make_article(journal_name=None)
        assert venue_score(article) == 0.30

    def test_journal_no_quartile_middle_baseline(self):
        article = _make_article(journal_name="Unknown Journal", quartile=None)
        assert venue_score(article) == 0.50


# ============================================================
# Authors score tests
# ============================================================


class TestAuthorsScore:
    def test_no_authors_baseline(self):
        article = _make_article(authors=[])
        assert authors_score(article) == 0.30

    def test_authors_no_h_index(self):
        authors = [Author(name="A"), Author(name="B")]
        article = _make_article(authors=authors)
        assert authors_score(article) == 0.30

    def test_h_index_zero(self):
        article = _make_article(authors=[Author(name="A", h_index=0)])
        assert authors_score(article) == 0.0

    def test_takes_max_h_index(self):
        # max(5, 50, 20) = 50, log(51)/log(101) ~= 0.852
        authors = [
            Author(name="A", h_index=5),
            Author(name="B", h_index=50),
            Author(name="C", h_index=20),
        ]
        article = _make_article(authors=authors)
        score = authors_score(article)
        assert 0.84 < score < 0.86

    def test_h_index_at_saturation(self):
        article = _make_article(authors=[Author(name="A", h_index=100)])
        assert authors_score(article) == 1.0

    def test_h_index_above_saturation_capped(self):
        article = _make_article(authors=[Author(name="A", h_index=200)])
        assert authors_score(article) == 1.0

    def test_partial_h_index_data(self):
        """Some authors have h-index, others don't - use what we have."""
        authors = [
            Author(name="A", h_index=None),
            Author(name="B", h_index=30),
        ]
        article = _make_article(authors=authors)
        # Should use h=30
        assert 0.7 < authors_score(article) < 0.8


# ============================================================
# Impact score tests
# ============================================================


class TestImpactScore:
    def test_recent_paper_no_citations_baseline(self):
        # 2024 paper checked in 2024 with 0 citations -> baseline 0.5
        article = _make_article(year=2024, citation_count=0)
        assert impact_score(article, current_year=2024) == 0.50

    def test_recent_paper_with_citations(self):
        # 2024 paper checked in 2024, 10 cites -> 10/1 = 10/year
        article = _make_article(year=2024, citation_count=10)
        # log(11)/log(51) ~= 0.610
        score = impact_score(article, current_year=2024)
        assert 0.59 < score < 0.62

    def test_old_paper_few_citations(self):
        # 2010 paper checked in 2024, 50 cites -> 50/15 ~= 3.3/year
        article = _make_article(year=2010, citation_count=50)
        # log(4.3)/log(51) ~= 0.371
        score = impact_score(article, current_year=2024)
        assert 0.35 < score < 0.40

    def test_high_impact_saturates(self):
        # 1000 cites in 5 years = 200/year >> saturation of 50
        article = _make_article(year=2019, citation_count=1000)
        assert impact_score(article, current_year=2024) == 1.0

    def test_zero_citations_old_paper(self):
        article = _make_article(year=2015, citation_count=0)
        assert impact_score(article, current_year=2024) == 0.0

    def test_normalization_compares_correctly(self):
        """The whole point of cites/year: a recent paper should beat
        an older one with same total citations."""
        old = _make_article(year=2010, citation_count=100)  # 7/year
        recent = _make_article(year=2022, citation_count=100)  # 33/year
        assert impact_score(recent, 2024) > impact_score(old, 2024)


# ============================================================
# Relevance score tests
# ============================================================


class TestRelevanceScore:
    def test_perfect_match_in_title(self):
        article = _make_article(
            title="Fake news detection using deep learning",
            abstract="A summary",
        )
        # All 3 topic tokens (fake, news, detection) hit the title
        score = relevance_score(article, "fake news detection")
        assert score > 0.4  # Title weighted 3/6 = max 0.5 from title alone

    def test_no_match_returns_floor(self):
        article = _make_article(
            title="Quantum computing for chemistry",
            abstract="Hartree-Fock methods",
        )
        score = relevance_score(article, "fake news detection")
        assert score == 0.05  # _SCORE_FLOOR

    def test_partial_match_in_abstract(self):
        article = _make_article(
            title="A new model",  # no topic words
            abstract="We address fake news in social media.",
        )
        score = relevance_score(article, "fake news detection")
        # 2 of 3 topic tokens match in abstract
        assert 0.05 < score < 0.4

    def test_concepts_boost_relevance(self):
        article = _make_article(
            title="Generic title",
            abstract="Generic abstract",
            concepts=["Fake news", "Detection methods"],
        )
        score = relevance_score(article, "fake news detection")
        # Concepts have weight 2 -> partial match still gives signal
        assert score > 0.05

    def test_empty_topic_returns_floor(self):
        article = _make_article(title="Anything")
        # Empty topic -> only stop words
        assert relevance_score(article, "the and or") == 0.05

    def test_case_insensitive(self):
        article = _make_article(title="FAKE NEWS DETECTION methods")
        score = relevance_score(article, "fake news detection")
        assert score > 0.4

    def test_punctuation_handled(self):
        article = _make_article(title="Fake-news detection: a survey.")
        score = relevance_score(article, "fake news detection")
        assert score > 0.4

    # ── Bigram phrase matching (V1.5) ────────────────────────────────────

    def test_bigram_match_boosts_above_unigram_only(self):
        """'attention mechanism' as a phrase scores higher than tokens scattered
        in different sentences without appearing adjacent."""
        phrase_article = _make_article(
            title="Recurrent attention mechanism for dynamic position encoding",
            abstract="We use attention mechanism to improve positional encoding.",
        )
        scatter_article = _make_article(
            title="A recurrent model for position-aware sequence encoding",
            abstract="We use attention in a dynamic way with separate mechanism.",
        )
        topic = "recurrent attention mechanism dynamic position"
        phrase_score = relevance_score(phrase_article, topic)
        scatter_score = relevance_score(scatter_article, topic)
        assert phrase_score > scatter_score, (
            "Phrase match ('attention mechanism' adjacent) should outscore "
            "scattered unigram matches"
        )

    def test_bigram_match_in_title_gives_significant_boost(self):
        """When the topic bigram appears verbatim in the title, score is high."""
        article = _make_article(
            title="Attention mechanism for neural machine translation",
            abstract="We propose a novel attention mechanism.",
            concepts=["attention mechanism", "neural machine translation"],
        )
        score = relevance_score(article, "attention mechanism neural translation")
        # Topic: 4 tokens → 3 bigrams: "attention mechanism", "mechanism neural",
        # "neural translation". All 3 bigrams present in title → big bonus.
        assert score > 0.6, f"Expected > 0.6 for phrase-match title, got {score:.3f}"

    def test_single_token_topic_no_bigram_regression(self):
        """Single-token topic has no bigrams → scoring unchanged from V1."""
        article = _make_article(
            title="Transformer architecture for NLP",
            abstract="We study transformers.",
        )
        score = relevance_score(article, "transformer")
        # 1 token, 0 bigrams → unigram only; normalization = 6.0 as in V1
        # "transformer" in title → title_score=3/6=0.5; floor=0.05 → score=0.5
        assert 0.45 < score < 0.6

    def test_off_topic_single_unigram_match_stays_low(self):
        """An article matching only 1 of 5 topic tokens should score < MIN_RELEVANCE."""
        from src.config import MIN_RELEVANCE_SCORE
        article = _make_article(
            title="Bidirectional LSTM for named entity recognition",
            abstract="We use a BiLSTM model trained on CoNLL-2003.",
            concepts=["LSTM", "sequence labeling"],
        )
        topic = "recurrent attention mechanism dynamic position"
        score = relevance_score(article, topic)
        # Only "recurrent" might faintly appear in "LSTM" concepts chain, but
        # "attention", "mechanism", "dynamic", "position" are absent → low score
        assert score < MIN_RELEVANCE_SCORE, (
            f"Off-topic LSTM article scored {score:.3f} >= MIN_RELEVANCE_SCORE "
            f"({MIN_RELEVANCE_SCORE}); should be excluded by quality_critic"
        )

    # ── Bug-regression: underscore in topic ──────────────────────────────

    def test_underscore_topic_same_as_space_topic(self):
        """'federated_learning' must score the same as 'federated learning'.

        Regression test for the CLI bug where passing a slug-style topic
        (with underscores) caused tokenization to produce a single token
        ('federated_learning') that never matched any article text, resulting
        in all articles being excluded by the MIN_RELEVANCE_SCORE gate.
        """
        article = _make_article(
            title="Federated Learning for Privacy-Preserving Medical Data",
            abstract="We propose federated learning to train models without sharing data.",
            concepts=["federated learning", "privacy"],
        )
        score_space = relevance_score(article, "federated learning")
        score_slug = relevance_score(article, "federated_learning")
        assert score_space == pytest.approx(score_slug), (
            f"Underscore topic score {score_slug:.4f} != space topic score "
            f"{score_space:.4f}; _tokenize must split on underscores"
        )

    def test_underscore_topic_above_min_relevance(self):
        """A relevant article must pass the gate even with underscore topic."""
        from src.config import MIN_RELEVANCE_SCORE
        article = _make_article(
            title="Communication-Efficient Federated Learning",
            abstract="Federated learning allows training without centralizing data.",
            concepts=["federated learning", "distributed machine learning"],
        )
        score = relevance_score(article, "federated_learning")
        assert score >= MIN_RELEVANCE_SCORE, (
            f"Relevant federated learning article scored {score:.4f} below gate "
            f"({MIN_RELEVANCE_SCORE}) with underscore topic — underscore fix missing"
        )


# ============================================================
# Velocity score tests (Level 3)
# ============================================================


class TestVelocityScore:

    def test_output_in_range(self):
        article = _make_article(citation_count=50, year=2022)
        s = velocity_score(article, current_year=2024)
        assert 0.0 <= s <= 1.0

    def test_zero_citations_old_paper(self):
        article = _make_article(citation_count=0, year=2020)
        assert velocity_score(article, current_year=2024) == 0.0

    def test_too_new_no_citations_returns_baseline(self):
        """Paper published this year with 0 citations → neutral 0.50."""
        article = _make_article(citation_count=0, year=2026)
        assert velocity_score(article, current_year=2026) == pytest.approx(0.50)

    def test_saturation_capped_at_one(self):
        """Very high cites/year must not exceed 1.0."""
        article = _make_article(citation_count=1000, year=2023)
        assert velocity_score(article, current_year=2024) == pytest.approx(1.0)

    def test_known_value(self):
        """age=2, cites=10 → cpy=5, score=5/20=0.25."""
        article = _make_article(citation_count=10, year=2022)
        assert velocity_score(article, current_year=2024) == pytest.approx(0.25)

    def test_higher_velocity_scores_higher(self):
        slow  = _make_article(citation_count=5,  year=2022)
        fast  = _make_article(citation_count=30, year=2022)
        assert velocity_score(fast, current_year=2024) > velocity_score(slow, current_year=2024)

    @pytest.mark.parametrize("citations,year,expected", [
        (0,   2020,  0.00),   # 0 cpy → 0
        (20,  2022,  0.50),   # 10 cpy / 20 sat = 0.50
        (40,  2022,  1.00),   # 20 cpy / 20 sat = 1.00 (saturated)
    ])
    def test_parametrized_values(self, citations, year, expected):
        article = _make_article(citation_count=citations, year=year)
        assert velocity_score(article, current_year=2024) == pytest.approx(expected, abs=0.01)


# ============================================================
# Recency score tests (Level 3)
# ============================================================


class TestRecencyScore:

    def test_output_in_range(self):
        article = _make_article(year=2022)
        s = recency_score(article, current_year=2024)
        assert 0.0 < s <= 1.0

    def test_current_year_returns_one(self):
        """Paper published this year → age=0 → score=1.0."""
        article = _make_article(year=2024)
        assert recency_score(article, current_year=2024) == pytest.approx(1.0)

    def test_half_life_returns_half(self):
        """Paper at half-life age (3 years) → score=0.5."""
        article = _make_article(year=2021)
        assert recency_score(article, current_year=2024) == pytest.approx(0.5, abs=1e-6)

    def test_double_half_life_returns_quarter(self):
        """Paper at 2×half_life (6 years) → score=0.25."""
        article = _make_article(year=2018)
        assert recency_score(article, current_year=2024) == pytest.approx(0.25, abs=1e-6)

    def test_older_paper_scores_lower(self):
        recent = _make_article(year=2023)
        older  = _make_article(year=2019)
        assert recency_score(recent, current_year=2024) > recency_score(older, current_year=2024)

    def test_score_never_zero(self):
        """Very old papers (20+ years) should still be > 0."""
        very_old = _make_article(year=2000)
        assert recency_score(very_old, current_year=2024) > 0.0

    def test_monotonically_decreasing(self):
        """Older papers always score lower than newer ones."""
        years = [2024, 2022, 2020, 2018, 2015]
        scores = [recency_score(_make_article(year=y), current_year=2024) for y in years]
        assert scores == sorted(scores, reverse=True)
