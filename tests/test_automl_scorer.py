"""Tests for src/scoring/automl_scorer.py.

Strategy
--------
- All tests are pure in-memory (no real Optuna SQLite persistence during tests).
  We use tmp_path (pytest fixture) to redirect WEIGHTS_DIR writes.
- Optuna trials are capped at n_trials=5 so tests run in < 1 s.
- No network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from src.schemas import Article, Author, SourceType


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_article(
    idx: int,
    topic_hit: bool = False,
) -> Article:
    """Build an Article.  topic_hit=True → title contains 'fake news detection'."""
    title = (
        f"Fake News Detection via Transformer Approach {idx}"
        if topic_hit
        else f"Unrelated Computer Vision Paper {idx}"
    )
    abstract = (
        "We propose a method for detecting fake news on social media."
        if topic_hit
        else "We study image segmentation."
    )
    return Article(
        id=f"art_{idx}",
        title=title,
        abstract=abstract,
        year=2023,
        source=SourceType.OPENALEX,
        url=f"https://example.com/{idx}",
        citation_count=50 * idx,
        concepts=["fake news", "detection"] if topic_hit else ["computer vision"],
    )


def _make_corpus() -> tuple[list[Article], dict[str, int]]:
    """
    10 articles, 3 relevant (grade 2), 3 semi-relevant (grade 1), 4 not relevant.
    """
    articles = [_make_article(i, topic_hit=(i <= 6)) for i in range(1, 11)]
    gold: dict[str, int] = {}
    for i in range(1, 11):
        if i <= 3:
            gold[f"art_{i}"] = 2   # highly relevant
        elif i <= 6:
            gold[f"art_{i}"] = 1   # relevant
        else:
            gold[f"art_{i}"] = 0   # not relevant
    return articles, gold


# ── _safe_topic_name ─────────────────────────────────────────────────────────


class TestSafeTopicName:

    def test_lowercase_spaces_to_underscores(self):
        from src.scoring.automl_scorer import _safe_topic_name
        assert _safe_topic_name("Fake News Detection") == "fake_news_detection"

    def test_special_chars_removed(self):
        from src.scoring.automl_scorer import _safe_topic_name
        result = _safe_topic_name("Graph Neural Networks (GNN)")
        assert result == "graph_neural_networks_gnn"

    def test_empty_string_returns_fallback(self):
        from src.scoring.automl_scorer import _safe_topic_name
        assert _safe_topic_name("") == "unknown_topic"

    def test_already_safe(self):
        from src.scoring.automl_scorer import _safe_topic_name
        assert _safe_topic_name("nlp") == "nlp"


# ── save / load weights ───────────────────────────────────────────────────────


class TestWeightsPersistence:

    def test_save_then_load_roundtrip_6keys(self, tmp_path):
        """save/load roundtrip with full 6-key Level-3 weights."""
        from src.scoring import automl_scorer

        weights = {
            "venue": 0.15, "authors": 0.15, "impact": 0.20,
            "velocity": 0.15, "recency": 0.15, "relevance": 0.20,
        }

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            saved_path = automl_scorer.save_weights_for_topic("test topic", weights)
            assert saved_path.exists()
            loaded = automl_scorer.load_weights_for_topic("test topic")

        assert loaded is not None
        for k in weights:
            assert loaded[k] == pytest.approx(weights[k])

    def test_save_then_load_roundtrip_4keys_backward_compat(self, tmp_path):
        """Old 4-key weight files must still load successfully (backward compat)."""
        from src.scoring import automl_scorer

        weights = {"venue": 0.2, "authors": 0.15, "impact": 0.35, "relevance": 0.30}

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            saved_path = automl_scorer.save_weights_for_topic("old topic", weights)
            assert saved_path.exists()
            loaded = automl_scorer.load_weights_for_topic("old topic")

        assert loaded is not None
        for k in weights:
            assert loaded[k] == pytest.approx(weights[k])

    def test_load_returns_none_if_no_file(self, tmp_path):
        from src.scoring import automl_scorer

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = automl_scorer.load_weights_for_topic("nonexistent topic")

        assert result is None

    def test_load_returns_none_on_malformed_json(self, tmp_path):
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import _safe_topic_name

        bad_file = tmp_path / f"{_safe_topic_name('bad topic')}_weights.json"
        bad_file.write_text("not-valid-json", encoding="utf-8")

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = automl_scorer.load_weights_for_topic("bad topic")

        assert result is None

    def test_load_returns_none_on_missing_keys(self, tmp_path):
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import _safe_topic_name

        bad_file = tmp_path / f"{_safe_topic_name('incomplete')}_weights.json"
        bad_file.write_text(
            json.dumps({"topic": "incomplete", "weights": {"venue": 0.5}}),
            encoding="utf-8",
        )

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = automl_scorer.load_weights_for_topic("incomplete")

        assert result is None

    def test_save_overwrites_existing(self, tmp_path):
        from src.scoring import automl_scorer

        w1 = {"venue": 0.25, "authors": 0.25, "impact": 0.25, "relevance": 0.25}
        w2 = {"venue": 0.40, "authors": 0.10, "impact": 0.30, "relevance": 0.20}

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            automl_scorer.save_weights_for_topic("my topic", w1)
            automl_scorer.save_weights_for_topic("my topic", w2)  # overwrite
            loaded = automl_scorer.load_weights_for_topic("my topic")

        assert loaded is not None
        assert loaded["venue"] == pytest.approx(0.40)


# ── optimize_weights ──────────────────────────────────────────────────────────


class TestOptimizeWeights:

    @pytest.fixture
    def corpus(self):
        return _make_corpus()

    def test_returns_optimization_result(self, tmp_path, corpus):
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = optimize_weights(
                articles, gold, "fake news detection",
                n_trials=5, timeout=0,  # fast test: 5 trials, no timeout
            )

        assert result.topic == "fake news detection"
        assert 0.0 <= result.best_ndcg_at_15 <= 1.0
        assert 0.0 <= result.baseline_ndcg_at_15 <= 1.0
        assert result.n_trials_completed == 5
        assert result.duration_seconds >= 0

    def test_best_weights_normalized(self, tmp_path, corpus):
        """best_weights in OptimizationResult must sum to ~1.0."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = optimize_weights(articles, gold, "fake news detection",
                                      n_trials=5, timeout=0)

        total = sum(result.best_weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_best_weights_has_all_keys(self, tmp_path, corpus):
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = optimize_weights(articles, gold, "fake news detection",
                                      n_trials=5, timeout=0)

        # Level 3: 6 features
        assert set(result.best_weights.keys()) == {
            "venue", "authors", "impact", "velocity", "recency", "relevance"
        }

    def test_weights_saved_when_improvement_large_enough(self, tmp_path, corpus):
        """When NDCG improvement ≥ threshold, weights file is written."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import OptimizationResult, optimize_weights

        articles, gold = corpus

        # Fake a scenario where best_ndcg >> baseline → definitely saves
        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path), \
             patch.object(automl_scorer, "OPTUNA_MIN_IMPROVEMENT", 0.0):
            # min improvement = 0% → always save
            result = optimize_weights(articles, gold, "fake news detection",
                                      n_trials=5, timeout=0)

        # With min_improvement=0, weights_saved must be True
        assert result.weights_saved is True
        weights_file = tmp_path / "fake_news_detection_weights.json"
        assert weights_file.exists()

    def test_weights_not_saved_when_improvement_below_threshold(self, tmp_path, corpus):
        """When improvement < threshold, no weights file is written."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        # Set threshold to 200% → Optuna will never beat that → weights not saved
        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path), \
             patch.object(automl_scorer, "OPTUNA_MIN_IMPROVEMENT", 2.0):
            result = optimize_weights(articles, gold, "fake news detection",
                                      n_trials=5, timeout=0)

        assert result.weights_saved is False
        weights_file = tmp_path / "fake_news_detection_weights.json"
        assert not weights_file.exists()

    def test_raises_on_empty_articles(self, tmp_path):
        from src.scoring.automl_scorer import optimize_weights

        with patch("src.scoring.automl_scorer.WEIGHTS_DIR", tmp_path):
            with pytest.raises(ValueError, match="candidate_articles"):
                optimize_weights([], {"a1": 1}, "test topic", n_trials=2, timeout=0)

    def test_raises_on_empty_gold(self, tmp_path, corpus):
        from src.scoring.automl_scorer import optimize_weights

        articles, _ = corpus

        with patch("src.scoring.automl_scorer.WEIGHTS_DIR", tmp_path):
            with pytest.raises(ValueError, match="gold_relevance"):
                optimize_weights(articles, {}, "test topic", n_trials=2, timeout=0)

    def test_force_rerun_starts_fresh(self, tmp_path, corpus):
        """force_rerun=True must produce a fresh study (no trial accumulation)."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            # First run: 3 trials
            r1 = optimize_weights(articles, gold, "fake news",
                                  n_trials=3, timeout=0, force_rerun=False)
            # Second run: force_rerun=True, 5 trials — should have exactly 5 (not 8)
            r2 = optimize_weights(articles, gold, "fake news",
                                  n_trials=5, timeout=0, force_rerun=True)

        assert r2.n_trials_completed == 5

    def test_resume_accumulates_trials(self, tmp_path, corpus):
        """Without force_rerun, second run resumes and total trials accumulate."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            r1 = optimize_weights(articles, gold, "nlp topic",
                                  n_trials=3, timeout=0, force_rerun=False)
            r2 = optimize_weights(articles, gold, "nlp topic",
                                  n_trials=3, timeout=0, force_rerun=False)

        # After two runs of 3, total should be ≥ 6 (Optuna may skip exact count)
        assert r2.n_trials_completed >= 6


# ── OptimizationResult schema ─────────────────────────────────────────────────


class TestOptimizationResultSchema:

    def test_pydantic_model_validates(self):
        from src.scoring.automl_scorer import OptimizationResult

        # Level 3: 6-key weights
        result = OptimizationResult(
            topic="fake news detection",
            best_weights={
                "venue": 0.15, "authors": 0.15, "impact": 0.20,
                "velocity": 0.15, "recency": 0.15, "relevance": 0.20,
            },
            best_ndcg_at_15=0.85,
            baseline_ndcg_at_15=0.72,
            improvement_pct=18.06,
            n_trials_completed=150,
            duration_seconds=1.23,
            weights_saved=True,
        )
        assert result.topic == "fake news detection"
        assert result.weights_saved is True

    def test_ndcg_must_be_in_range(self):
        from pydantic import ValidationError
        from src.scoring.automl_scorer import OptimizationResult

        with pytest.raises(ValidationError):
            OptimizationResult(
                topic="t",
                best_weights={
                    "venue": 0.15, "authors": 0.15, "impact": 0.20,
                    "velocity": 0.15, "recency": 0.15, "relevance": 0.20,
                },
                best_ndcg_at_15=1.5,   # > 1 → should fail
                baseline_ndcg_at_15=0.5,
                improvement_pct=0.0,
                n_trials_completed=10,
                duration_seconds=1.0,
                weights_saved=False,
            )

    def test_relevance_method_default_is_v2(self):
        """relevance_method field defaults to 'v2' for backward compatibility."""
        from src.scoring.automl_scorer import OptimizationResult

        result = OptimizationResult(
            topic="t",
            best_weights={"venue": 0.15, "authors": 0.15, "impact": 0.20,
                          "velocity": 0.15, "recency": 0.15, "relevance": 0.20},
            best_ndcg_at_15=0.5, baseline_ndcg_at_15=0.4,
            improvement_pct=25.0, n_trials_completed=5,
            duration_seconds=1.0, weights_saved=False,
        )
        assert result.relevance_method == "v2"


# ── use_semantic parameter ────────────────────────────────────────────────────


class TestUseSemantic:
    """Tests for the use_semantic parameter added for V1.5 vs V2 comparison."""

    @pytest.fixture
    def corpus(self):
        return _make_corpus()

    def test_study_name_kw_suffix_when_v1(self):
        """_study_name with use_semantic=False ends with '_kw'."""
        from src.scoring.automl_scorer import _study_name

        name = _study_name("fake news detection", use_semantic=False)
        assert name.endswith("_kw"), f"Expected '_kw' suffix, got: {name!r}"

    def test_study_name_em_suffix_when_v2(self):
        """_study_name with use_semantic=True ends with '_em'."""
        from src.scoring.automl_scorer import _study_name

        name = _study_name("fake news detection", use_semantic=True)
        assert name.endswith("_em"), f"Expected '_em' suffix, got: {name!r}"

    def test_study_names_differ_between_v1_and_v2(self):
        """V1 and V2 runs must have different study names to avoid trial mixing."""
        from src.scoring.automl_scorer import _study_name

        topic = "fake news detection"
        assert _study_name(topic, use_semantic=False) != _study_name(topic, use_semantic=True)

    def test_optimize_weights_v1_returns_relevance_method_v1(self, tmp_path, corpus):
        """optimize_weights(use_semantic=False) → result.relevance_method == 'v1'."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = optimize_weights(
                articles, gold, "fake news detection",
                n_trials=3, timeout=0, use_semantic=False,
            )

        assert result.relevance_method == "v1"

    def test_optimize_weights_v2_returns_relevance_method_v2(self, tmp_path, corpus):
        """optimize_weights(use_semantic=True) → result.relevance_method == 'v2'."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            result = optimize_weights(
                articles, gold, "fake news detection",
                n_trials=3, timeout=0, use_semantic=True,
            )

        assert result.relevance_method == "v2"

    def test_v1_and_v2_runs_are_independent(self, tmp_path, corpus):
        """Running V1 then V2 (or vice versa) never shares Optuna trial history."""
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus
        topic = "independent topic test"

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path):
            r_v1 = optimize_weights(
                articles, gold, topic, n_trials=3, timeout=0, use_semantic=False
            )
            r_v2 = optimize_weights(
                articles, gold, topic, n_trials=3, timeout=0, use_semantic=True
            )

        # Each run must have exactly 3 trials (no cross-contamination)
        assert r_v1.n_trials_completed == 3
        assert r_v2.n_trials_completed == 3

    def test_use_semantic_false_calls_score_articles_with_v1(self, tmp_path, corpus):
        """When use_semantic=False, score_articles is called with use_semantic=False."""
        from unittest.mock import call
        from src.scoring import automl_scorer
        from src.scoring.automl_scorer import optimize_weights

        articles, gold = corpus
        call_args: list[bool] = []

        original_score = automl_scorer.score_articles

        def capturing_score(arts, topic, weights, use_semantic=True, **kw):
            call_args.append(use_semantic)
            return original_score(arts, topic, weights, use_semantic=False)

        with patch.object(automl_scorer, "WEIGHTS_DIR", tmp_path), \
             patch.object(automl_scorer, "score_articles", side_effect=capturing_score):
            optimize_weights(
                articles, gold, "test topic",
                n_trials=2, timeout=0, use_semantic=False,
            )

        # Every score_articles call must have received use_semantic=False
        assert all(sem is False for sem in call_args), (
            f"Expected all calls with use_semantic=False, got: {call_args}"
        )
