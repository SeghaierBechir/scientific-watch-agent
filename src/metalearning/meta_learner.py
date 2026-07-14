"""Phase 8c — Meta-learner core.

Maps domain meta-features to optimal scoring weights using two methods:

  kNN (k=1,2,3): predict weights as inverse-distance-weighted average of
                 the k most similar training domains (in feature space).

  RidgeCV:       fit one RidgeCV regression per weight dimension independently;
                 alpha is selected automatically via generalised cross-validation
                 from RIDGE_ALPHAS; predictions are clipped to ≥0 and re-normalised.

Feature set: 7 features with max|r| > 0.40 on n=19 domains after oracle repair (Phase 8b).
Normalization: MinMax per feature (required so citation_median doesn't dominate kNN).

Thesis context — Hypothesis H3
-------------------------------
    "Predicting weights via meta_features(D) -> weights beats Transfer Direct
    (mean of training weights) on >=75% of held-out domains."

Evaluated with Leave-One-Out CV in phase8c.py.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.preprocessing import MinMaxScaler

from src.metalearning.meta_features import DomainMetaFeatures

logger = logging.getLogger(__name__)

WEIGHT_KEYS: list[str] = [
    "venue", "authors", "impact", "velocity", "recency", "relevance"
]

# Alpha candidates for RidgeCV — 6 orders of magnitude (auto-selected per weight dimension)
RIDGE_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Features selected by Phase 8b re-run (|r| > 0.40, n=19 domains after oracle repair).
# Update vs previous: grade2_ratio rose to 0.67 (added), pct_recent rose to 0.45 (added).
# year_std dropped 0.41->0.38 (removed), unique_author_ratio dropped to 0.18 (removed).
EFFECTIVE_FEATURES: list[str] = [
    "grade2_ratio",         # max|r|=0.67 -> velocity(-0.67)  [new after oracle repair]
    "mean_h_index",         # max|r|=0.61 -> relevance(-0.61), authors(+0.44)
    "citation_median",      # max|r|=0.58 -> recency(+0.58), velocity(-0.45)
    "citation_gini",        # max|r|=0.56 -> venue(-0.56)
    "pct_high_hindex",      # max|r|=0.52 -> relevance(-0.52)
    "pct_recent",           # max|r|=0.45 -> relevance(-0.45)  [new after oracle repair]
    "pct_high_cited",       # max|r|=0.41 -> recency(+0.41)
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    """Clip negatives and re-normalise weights to sum to 1."""
    clipped = {k: max(0.0, v) for k, v in w.items()}
    total = sum(clipped.values())
    if total < 1e-10:
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}


# ── MetaLearner ───────────────────────────────────────────────────────────────


class MetaLearner:
    """Predicts 6 scoring weights from 6 domain meta-features.

    Usage:
        learner = MetaLearner()
        learner.fit(train_features, train_weights)
        w_knn = learner.predict_knn(test_feat, k=1)
        w_ridge = learner.predict_ridge(test_feat)
        w_td = learner.transfer_direct()   # baseline
    """

    def __init__(self, feature_names: list[str] = EFFECTIVE_FEATURES) -> None:
        self.feature_names = feature_names
        self._scaler = MinMaxScaler()
        self._ridge: dict[str, RidgeCV] = {}
        self._bayesian: dict[str, BayesianRidge] = {}
        self._gpr: dict[str, GaussianProcessRegressor] = {}
        self._X_raw: list[list[float]] = []
        self._W: list[dict[str, float]] = []
        self._domain_ids: list[str] = []
        self._w_mean_raw: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        features: list[DomainMetaFeatures],
        weights: list[dict[str, float]],
        domain_ids: Optional[list[str]] = None,
    ) -> "MetaLearner":
        """Fit on (meta_features, Optuna_weights) training pairs.

        Args:
            features:   DomainMetaFeatures for each training domain.
            weights:    Corresponding Optuna-learned weight dicts.
            domain_ids: Optional identifiers (used for logging).
        """
        if len(features) < 2:
            raise ValueError("Need at least 2 training domains.")
        if len(features) != len(weights):
            raise ValueError("`features` and `weights` must have the same length.")

        self._X_raw = [self._vec(mf) for mf in features]
        self._W = list(weights)
        self._domain_ids = domain_ids or [f"domain_{i}" for i in range(len(features))]

        X = np.array(self._X_raw, dtype=float)
        self._scaler.fit(X)
        X_s = self._scaler.transform(X)

        self._w_mean_raw = {
            key: float(np.mean([w.get(key, 0.0) for w in weights]))
            for key in WEIGHT_KEYS
        }

        for key in WEIGHT_KEYS:
            y = np.array([w.get(key, 0.0) for w in weights], dtype=float)
            y_delta = y - self._w_mean_raw[key]

            ridge_m = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
            ridge_m.fit(X_s, y_delta)
            self._ridge[key] = ridge_m

            bayes_m = BayesianRidge(fit_intercept=True)
            bayes_m.fit(X_s, y_delta)
            self._bayesian[key] = bayes_m

            # GPR: ConstantKernel * RBF (smooth non-linear) + WhiteKernel (noise).
            # A fresh kernel per weight avoids shared hyperparameter state.
            # Features are MinMax-scaled to [0,1] so length_scale bounds of (0.01, 10)
            # cover the full range of possible smoothness.
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
            )
            gpr_m = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                normalize_y=False,
                random_state=42,
            )
            gpr_m.fit(X_s, y_delta)
            self._gpr[key] = gpr_m

        logger.debug("RidgeCV selected alphas: %s", self.alpha_per_weight())
        return self

    def predict_knn(self, mf: DomainMetaFeatures, k: int = 1) -> dict[str, float]:
        """Predict via k-NN (inverse-distance-weighted average).

        When k=1 or an exact match is found, returns the nearest neighbour's
        weights directly (no blending).
        """
        self._check_fitted()
        x_s = self._scale(self._vec(mf))
        X_s = self._scaler.transform(np.array(self._X_raw, dtype=float))

        dists = np.sqrt(np.sum((X_s - x_s) ** 2, axis=1))
        k_eff = min(k, len(dists))
        idx = np.argsort(dists)[:k_eff]
        d_k = dists[idx]

        if k_eff == 1 or d_k[0] < 1e-10:
            return _normalize_weights(dict(self._W[idx[0]]))

        inv = 1.0 / (d_k + 1e-10)
        inv /= inv.sum()

        predicted: dict[str, float] = {
            key: float(sum(inv[i] * self._W[idx[i]].get(key, 0.0) for i in range(k_eff)))
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(predicted)

    def predict_ridge(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Predict via RidgeCV on deviations from Transfer Direct baseline.

        Trains on Δw = w_domain − w_mean; adds w_mean back at prediction time.
        When regularisation is strong (Δ→0), prediction degrades gracefully to
        Transfer Direct rather than adding noise around the mean.
        """
        self._check_fitted()
        x_s = self._scale(self._vec(mf))
        predicted = {
            key: float(self._w_mean_raw[key] + self._ridge[key].predict([x_s])[0])
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(predicted)

    def predict_bayesian_ridge(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Predict via Bayesian Ridge on deviations from Transfer Direct baseline.

        Regularisation strength is estimated from the data via empirical Bayes
        (Type-II maximum likelihood) rather than cross-validation.  Like
        predict_ridge, trains on Δw and adds w_mean back — so it degrades
        gracefully to Transfer Direct when no signal is detected.
        """
        self._check_fitted()
        x_s = self._scale(self._vec(mf))
        predicted = {
            key: float(self._w_mean_raw[key] + self._bayesian[key].predict([x_s])[0])
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(predicted)

    def predict_ensemble(
        self,
        mf: DomainMetaFeatures,
        alpha: float = 0.5,
    ) -> dict[str, float]:
        """Blend kNN(1) and Ridge: alpha*kNN(1) + (1-alpha)*Ridge.

        alpha=1.0 → pure kNN(1)
        alpha=0.0 → pure Ridge (equivalent to predict_ridge)
        """
        w_knn = self.predict_knn(mf, k=1)
        w_ridge = self.predict_ridge(mf)
        blended = {
            key: alpha * w_knn.get(key, 0.0) + (1.0 - alpha) * w_ridge.get(key, 0.0)
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(blended)

    # ── Rank-fusion ensemble (kNN1 + kNN2 + kNN3 + BayesRidge) ──────────────

    def _member_preds(self, mf: DomainMetaFeatures) -> list[dict[str, float]]:
        """Return raw weight predictions from the 4 ensemble members.

        Members: kNN(1), kNN(2), kNN(3), BayesianRidge.
        All four are trained on the same feature space and weight targets,
        but make predictions through structurally different mechanisms —
        making them complementary base learners for rank fusion.
        """
        return [
            self.predict_knn(mf, k=1),
            self.predict_knn(mf, k=2),
            self.predict_knn(mf, k=3),
            self.predict_bayesian_ridge(mf),
        ]

    def predict_soft_vote(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Soft vote: uniform average of the 4 member weight predictions.

        Each member contributes equally.  After averaging, weights are
        re-normalised so they sum to 1.  Equivalent to a linear opinion
        pool with uniform prior on members.
        """
        preds = self._member_preds(mf)
        n = len(preds)
        averaged = {
            key: sum(p.get(key, 0.0) for p in preds) / n
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(averaged)

    def predict_hard_vote(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Hard vote: per-dimension median across the 4 member predictions.

        The median is more robust to a single outlier prediction than the
        mean (e.g. when one kNN neighbour has an anomalous weight profile).
        With 4 values the median is interpolated as the mean of the two
        middle values — equivalent to trimming one outlier on each side.
        """
        preds = self._member_preds(mf)
        median_w: dict[str, float] = {}
        for key in WEIGHT_KEYS:
            vals = sorted(p.get(key, 0.0) for p in preds)
            # 4 values → average of 2nd and 3rd (0-indexed: 1 and 2)
            median_w[key] = (vals[1] + vals[2]) / 2.0
        return _normalize_weights(median_w)

    def predict_borda(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Borda count: rank-based fusion of the 4 member predictions.

        Each member ranks the 6 weight dimensions by predicted value
        (highest weight → rank 1).  Borda score for dimension d is
        sum_members(n_dims - rank_m(d)), so rank-1 earns (n-1) points
        and rank-n earns 0.  Scores are normalised to a weight vector.

        Motivation: Borda aggregates ordinal preferences and is immune
        to scale differences between members (e.g. kNN returning sparse
        weights vs BayesRidge returning near-uniform weights).
        """
        preds = self._member_preds(mf)
        n_dims = len(WEIGHT_KEYS)
        borda: dict[str, float] = {k: 0.0 for k in WEIGHT_KEYS}
        for pred in preds:
            ranked = sorted(WEIGHT_KEYS, key=lambda k: pred[k], reverse=True)
            for rank, key in enumerate(ranked):            # rank=0 → best
                borda[key] += float(n_dims - 1 - rank)    # n-1 down to 0
        return _normalize_weights(borda)

    def predict_rrf(self, mf: DomainMetaFeatures, k: int = 1) -> dict[str, float]:
        """Reciprocal Rank Fusion (RRF) of the 4 member predictions.

        RRF score for dimension d = sum_members( 1 / (k + rank_m(d)) )
        where rank is 1-based (best = 1).  k=60 is standard for IR over
        large document collections; with only 6 dimensions we use k=1 to
        preserve differentiation between top- and bottom-ranked features.

        RRF gives diminishing returns to high ranks and is robust to
        outlier rankers that place a feature at position 1 by a large margin
        — because 1/(k+1) is bounded regardless of the raw weight gap.
        """
        preds = self._member_preds(mf)
        rrf: dict[str, float] = {key: 0.0 for key in WEIGHT_KEYS}
        for pred in preds:
            ranked = sorted(WEIGHT_KEYS, key=lambda k: pred[k], reverse=True)
            for rank_0based, key in enumerate(ranked):
                rrf[key] += 1.0 / (k + rank_0based + 1)   # 1-based rank
        return _normalize_weights(rrf)

    def predict_ensemble_bayes(
        self,
        mf: DomainMetaFeatures,
        alpha: float = 0.5,
    ) -> dict[str, float]:
        """Blend kNN(1) and BayesianRidge: alpha*kNN(1) + (1-alpha)*BayesianRidge.

        alpha=1.0 → pure kNN(1)
        alpha=0.0 → pure BayesianRidge (equivalent to predict_bayesian_ridge)

        BayesRidge uses empirical Bayes regularisation instead of CV-Ridge,
        and has shown different failure patterns in LOO — making it a more
        informative blend partner than plain Ridge.
        """
        w_knn = self.predict_knn(mf, k=1)
        w_bayes = self.predict_bayesian_ridge(mf)
        blended = {
            key: alpha * w_knn.get(key, 0.0) + (1.0 - alpha) * w_bayes.get(key, 0.0)
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(blended)

    def predict_gpr(self, mf: DomainMetaFeatures) -> dict[str, float]:
        """Predict via Gaussian Process Regression on deviations from Transfer Direct.

        Kernel: ConstantKernel * RBF + WhiteKernel.  Hyperparameters are
        optimised per weight dimension via log-marginal-likelihood maximisation
        (3 random restarts to avoid local optima).

        Like Ridge, trains on delta w = w_domain - w_mean and adds w_mean back,
        so the prediction degrades gracefully to Transfer Direct when the kernel
        finds no usable signal (high noise relative to amplitude).
        """
        self._check_fitted()
        x_s = self._scale(self._vec(mf))
        predicted = {
            key: float(self._w_mean_raw[key] + self._gpr[key].predict([x_s])[0])
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(predicted)

    def predict_gpr_with_std(
        self, mf: DomainMetaFeatures
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Predict via GPR and return predictive standard deviation per weight.

        High std on a weight dimension means the posterior is close to the prior
        (= Transfer Direct mean) — the model is uncertain for that dimension.
        Mean std across weights can drive adaptive routing:
            if mean_std > threshold: fall back to Ridge or Transfer Direct.
        """
        self._check_fitted()
        x_s = self._scale(self._vec(mf))
        weights: dict[str, float] = {}
        stds: dict[str, float] = {}
        for key in WEIGHT_KEYS:
            mu, sigma = self._gpr[key].predict([x_s], return_std=True)
            weights[key] = float(self._w_mean_raw[key] + mu[0])
            stds[key] = float(sigma[0])
        return _normalize_weights(weights), stds

    def transfer_direct(self) -> dict[str, float]:
        """Mean of all training weights — Transfer Direct baseline.

        This represents "use any domain's learned weights on a new domain
        without adaptation", averaged over all training domains.
        """
        self._check_fitted()
        mean_w = {
            key: float(np.mean([w.get(key, 0.0) for w in self._W]))
            for key in WEIGHT_KEYS
        }
        return _normalize_weights(mean_w)

    def alpha_per_weight(self) -> dict[str, float]:
        """Return the alpha selected by RidgeCV for each weight dimension.

        Useful for diagnostics: a high alpha means RidgeCV found little signal
        for that weight dimension and fell back to heavy regularisation.
        """
        self._check_fitted()
        return {key: float(self._ridge[key].alpha_) for key in WEIGHT_KEYS}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vec(self, mf: DomainMetaFeatures) -> list[float]:
        return [float(getattr(mf, f)) for f in self.feature_names]

    def _scale(self, x: list[float]) -> np.ndarray:
        return self._scaler.transform(np.array([x], dtype=float))[0]

    def _check_fitted(self) -> None:
        if not self._X_raw:
            raise RuntimeError("MetaLearner not fitted. Call fit() first.")
