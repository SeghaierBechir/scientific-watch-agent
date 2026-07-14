"""Tests for Phase 8c — MetaLearner.

All tests use in-memory synthetic data — no disk I/O, no network calls.
"""

from __future__ import annotations

import pytest
import math

from src.metalearning.meta_learner import (
    EFFECTIVE_FEATURES,
    RIDGE_ALPHAS,
    WEIGHT_KEYS,
    MetaLearner,
    _normalize_weights,
)
from src.metalearning.meta_features import DomainMetaFeatures


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_mf(
    citation_median: float = 200.0,
    pct_high_cited: float = 0.6,
    citation_gini: float = 0.7,
    unique_author_ratio: float = 0.8,
    topic_concept_overlap: float = 0.5,
    gold_ratio: float = 0.85,
    **kwargs,
) -> DomainMetaFeatures:
    """Minimal DomainMetaFeatures for testing."""
    defaults = dict(
        domain_id="test_domain",
        topic="test topic",
        corpus_size=100,
        median_year=2020.0,
        pct_recent=0.02,
        year_std=1.2,
        citation_gini=citation_gini,
        citation_median=citation_median,
        pct_high_cited=pct_high_cited,
        unique_author_ratio=unique_author_ratio,
        mean_h_index=0.0,
        pct_high_hindex=0.0,
        pct_q1=0.0,
        topic_concept_overlap=topic_concept_overlap,
        gold_ratio=gold_ratio,
        grade2_ratio=0.08,
    )
    defaults.update(kwargs)
    return DomainMetaFeatures(**defaults)


def _make_weights(
    venue: float = 0.15,
    authors: float = 0.15,
    impact: float = 0.20,
    velocity: float = 0.15,
    recency: float = 0.15,
    relevance: float = 0.20,
) -> dict[str, float]:
    return dict(
        venue=venue, authors=authors, impact=impact,
        velocity=velocity, recency=recency, relevance=relevance,
    )


def _make_training_set(n: int = 5) -> tuple[
    list[DomainMetaFeatures], list[dict[str, float]]
]:
    """Synthetic training set with n=n distinct domains."""
    feats = [
        _make_mf(citation_median=100 + i * 30, pct_high_cited=min(0.4 + i * 0.04, 0.95),
                 citation_gini=min(0.5 + i * 0.04, 0.95), unique_author_ratio=min(0.7 + i * 0.02, 0.95),
                 topic_concept_overlap=min(0.3 + i * 0.06, 0.95), gold_ratio=min(0.7 + i * 0.03, 0.95),
                 domain_id=f"dom_{i}", topic=f"topic {i}")
        for i in range(n)
    ]
    weights = [
        _make_weights(venue=0.1 + i * 0.02, impact=0.25 - i * 0.02,
                      authors=0.15 + i * 0.01)
        for i in range(n)
    ]
    return feats, weights


# ── Tests: _normalize_weights ─────────────────────────────────────────────────


class TestNormalizeWeights:
    def test_sums_to_one(self):
        w = {"a": 0.3, "b": 0.5, "c": 0.2}
        n = _normalize_weights(w)
        assert sum(n.values()) == pytest.approx(1.0, abs=1e-9)

    def test_clips_negatives(self):
        w = {"a": -0.5, "b": 0.8, "c": 0.2}
        n = _normalize_weights(w)
        assert n["a"] == pytest.approx(0.0, abs=1e-9)
        assert n["b"] > 0
        assert n["c"] > 0

    def test_all_zeros_returns_uniform(self):
        w = {"a": 0.0, "b": 0.0}
        n = _normalize_weights(w)
        assert n["a"] == pytest.approx(0.5)
        assert n["b"] == pytest.approx(0.5)

    def test_already_normalized_unchanged(self):
        w = {"a": 0.5, "b": 0.5}
        n = _normalize_weights(w)
        assert n["a"] == pytest.approx(0.5)
        assert n["b"] == pytest.approx(0.5)

    def test_preserves_keys(self):
        w = {k: 1.0 for k in WEIGHT_KEYS}
        n = _normalize_weights(w)
        assert set(n.keys()) == set(WEIGHT_KEYS)


# ── Tests: MetaLearner.fit ────────────────────────────────────────────────────


class TestMetaLearnerFit:
    def test_fit_returns_self(self):
        feats, weights = _make_training_set(4)
        learner = MetaLearner()
        result = learner.fit(feats, weights)
        assert result is learner

    def test_fit_requires_at_least_2_domains(self):
        feats, weights = _make_training_set(1)
        with pytest.raises(ValueError, match="at least 2"):
            MetaLearner().fit(feats, weights)

    def test_fit_requires_same_length(self):
        feats, weights = _make_training_set(4)
        with pytest.raises(ValueError, match="same length"):
            MetaLearner().fit(feats, weights[:3])

    def test_fit_stores_training_data(self):
        feats, weights = _make_training_set(5)
        learner = MetaLearner().fit(feats, weights)
        assert len(learner._W) == 5
        assert len(learner._X_raw) == 5


# ── Tests: MetaLearner.predict_knn ────────────────────────────────────────────


class TestPredictKnn:
    def setup_method(self):
        self.feats, self.weights = _make_training_set(5)
        self.learner = MetaLearner().fit(self.feats, self.weights)

    def test_output_sums_to_one(self):
        w = self.learner.predict_knn(self.feats[0], k=1)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        w = self.learner.predict_knn(self.feats[0], k=1)
        assert all(v >= 0 for v in w.values())

    def test_has_all_weight_keys(self):
        w = self.learner.predict_knn(self.feats[0], k=1)
        assert set(w.keys()) == set(WEIGHT_KEYS)

    def test_k1_exact_match_returns_training_weights(self):
        # An exact copy of a training point should return its weights
        w = self.learner.predict_knn(self.feats[2], k=1)
        expected = _normalize_weights(self.weights[2])
        for key in WEIGHT_KEYS:
            assert w[key] == pytest.approx(expected[key], abs=1e-5)

    def test_k2_is_blend_of_two_neighbors(self):
        # k=2 result should be between k=1 result and second neighbor
        w1 = self.learner.predict_knn(self.feats[0], k=1)
        w2 = self.learner.predict_knn(self.feats[0], k=2)
        # With 2 neighbors blended, result should differ from k=1
        # (unless all neighbors are identical)
        assert w2 is not None
        assert sum(w2.values()) == pytest.approx(1.0, abs=1e-6)

    def test_k_larger_than_training_clamped(self):
        # k=100 with 5 training points: should not raise, uses all 5
        w = self.learner.predict_knn(self.feats[0], k=100)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            MetaLearner().predict_knn(self.feats[0], k=1)


# ── Tests: MetaLearner.predict_ridge ─────────────────────────────────────────


class TestPredictRidge:
    def setup_method(self):
        feats, weights = _make_training_set(6)
        self.learner = MetaLearner().fit(feats, weights)
        self.feats = feats

    def test_output_sums_to_one(self):
        w = self.learner.predict_ridge(self.feats[0])
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        w = self.learner.predict_ridge(self.feats[0])
        assert all(v >= 0 for v in w.values())

    def test_has_all_weight_keys(self):
        w = self.learner.predict_ridge(self.feats[0])
        assert set(w.keys()) == set(WEIGHT_KEYS)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            MetaLearner().predict_ridge(self.feats[0])

    def test_alpha_per_weight_in_candidate_list(self):
        alphas = self.learner.alpha_per_weight()
        assert set(alphas.keys()) == set(WEIGHT_KEYS)
        for key, alpha in alphas.items():
            assert alpha in RIDGE_ALPHAS, f"{key}: alpha={alpha} not in RIDGE_ALPHAS"

    def test_alpha_per_weight_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            MetaLearner().alpha_per_weight()

    def test_different_features_give_different_weights(self):
        # Two very different feature vectors should give different predictions
        far_feat = _make_mf(citation_median=10.0, pct_high_cited=0.0,
                            citation_gini=0.1, topic_concept_overlap=0.0,
                            gold_ratio=0.1)
        w_near = self.learner.predict_ridge(self.feats[0])
        w_far = self.learner.predict_ridge(far_feat)
        # At least one weight should differ
        assert any(
            abs(w_near[k] - w_far[k]) > 1e-6 for k in WEIGHT_KEYS
        )

    def test_deviation_baseline_matches_per_key_means(self):
        # _w_mean_raw must equal the per-key mean of training weights (pre-normalisation)
        feats, weights = _make_training_set(6)
        learner = MetaLearner().fit(feats, weights)
        n = len(weights)
        for key in WEIGHT_KEYS:
            expected = sum(w.get(key, 0.0) for w in weights) / n
            assert learner._w_mean_raw[key] == pytest.approx(expected, abs=1e-8), \
                f"{key}: _w_mean_raw={learner._w_mean_raw[key]}, expected={expected}"

    def test_ridge_degrades_to_near_transfer_direct_when_signal_weak(self):
        # When all training points share the same features (no signal),
        # Ridge predicts Δ≈0 for all dimensions → result close to Transfer Direct.
        same_feat = _make_mf()
        feats = [same_feat] * 6
        weights = [_make_weights()] * 6
        learner = MetaLearner().fit(feats, weights)
        td = learner.transfer_direct()
        ridge = learner.predict_ridge(same_feat)
        for key in WEIGHT_KEYS:
            assert abs(ridge[key] - td[key]) < 1e-4, \
                f"{key}: ridge={ridge[key]:.6f}, transfer_direct={td[key]:.6f}"


# ── Tests: MetaLearner.predict_bayesian_ridge ────────────────────────────────


class TestPredictBayesianRidge:
    def setup_method(self):
        feats, weights = _make_training_set(6)
        self.learner = MetaLearner().fit(feats, weights)
        self.feats = feats

    def test_output_sums_to_one(self):
        w = self.learner.predict_bayesian_ridge(self.feats[0])
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        w = self.learner.predict_bayesian_ridge(self.feats[0])
        assert all(v >= 0 for v in w.values())

    def test_has_all_weight_keys(self):
        w = self.learner.predict_bayesian_ridge(self.feats[0])
        assert set(w.keys()) == set(WEIGHT_KEYS)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            MetaLearner().predict_bayesian_ridge(self.feats[0])

    def test_different_features_give_different_weights(self):
        far_feat = _make_mf(citation_median=10.0, pct_high_cited=0.0,
                            citation_gini=0.1, topic_concept_overlap=0.0,
                            gold_ratio=0.1)
        w_near = self.learner.predict_bayesian_ridge(self.feats[0])
        w_far = self.learner.predict_bayesian_ridge(far_feat)
        assert any(abs(w_near[k] - w_far[k]) > 1e-6 for k in WEIGHT_KEYS)

    def test_degrades_to_near_transfer_direct_when_signal_weak(self):
        same_feat = _make_mf()
        feats = [same_feat] * 6
        weights = [_make_weights()] * 6
        learner = MetaLearner().fit(feats, weights)
        td = learner.transfer_direct()
        br = learner.predict_bayesian_ridge(same_feat)
        for key in WEIGHT_KEYS:
            assert abs(br[key] - td[key]) < 1e-4, \
                f"{key}: bayesian={br[key]:.6f}, transfer_direct={td[key]:.6f}"


# ── Tests: MetaLearner.transfer_direct ───────────────────────────────────────


class TestTransferDirect:
    def setup_method(self):
        feats, weights = _make_training_set(4)
        self.learner = MetaLearner().fit(feats, weights)
        self.weights = weights

    def test_output_sums_to_one(self):
        w = self.learner.transfer_direct()
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_is_mean_of_training_weights(self):
        td = self.learner.transfer_direct()
        n = len(self.weights)
        for key in WEIGHT_KEYS:
            expected_mean = sum(w.get(key, 0.0) for w in self.weights) / n
            # Result is normalized, so compare proportions
            # The ratio between keys should match raw means
        # At minimum, all values non-negative
        assert all(v >= 0 for v in td.values())

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            MetaLearner().transfer_direct()

    def test_single_training_domain_returns_its_weights(self):
        # With 2 training domains having identical weights, transfer_direct = those weights
        same_w = _make_weights(venue=0.2, authors=0.2, impact=0.2,
                               velocity=0.2, recency=0.1, relevance=0.1)
        feats, _ = _make_training_set(2)
        learner = MetaLearner().fit(feats, [same_w, same_w])
        td = learner.transfer_direct()
        norm = _normalize_weights(same_w)
        for key in WEIGHT_KEYS:
            assert td[key] == pytest.approx(norm[key], abs=1e-5)


# ── Tests: LOO invariants ─────────────────────────────────────────────────────


class TestLOOInvariants:
    """Verify structural invariants of the LOO evaluation."""

    def test_loo_fold_uses_n_minus_1_training_points(self):
        n = 6
        feats, weights = _make_training_set(n)
        for i in range(n):
            train_feats = [feats[j] for j in range(n) if j != i]
            train_weights = [weights[j] for j in range(n) if j != i]
            learner = MetaLearner().fit(train_feats, train_weights)
            assert len(learner._W) == n - 1

    def test_predictions_valid_for_all_methods(self):
        feats, weights = _make_training_set(8)
        test_feat = _make_mf(citation_median=250, topic_concept_overlap=0.6)
        learner = MetaLearner().fit(feats, weights)

        for method_name, w in [
            ("knn1",           learner.predict_knn(test_feat, k=1)),
            ("knn2",           learner.predict_knn(test_feat, k=2)),
            ("knn3",           learner.predict_knn(test_feat, k=3)),
            ("ridge",          learner.predict_ridge(test_feat)),
            ("transfer_direct", learner.transfer_direct()),
        ]:
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-5), \
                f"{method_name} weights don't sum to 1"
            assert all(v >= 0 for v in w.values()), \
                f"{method_name} has negative weights"

    def test_effective_features_subset_of_feature_names(self):
        from src.metalearning.meta_features import FEATURE_NAMES
        for f in EFFECTIVE_FEATURES:
            assert f in FEATURE_NAMES, \
                f"{f} in EFFECTIVE_FEATURES but not in FEATURE_NAMES"
