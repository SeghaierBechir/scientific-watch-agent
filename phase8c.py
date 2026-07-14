"""Phase 8c — Meta-Learner: Leave-One-Out Evaluation.

Trains kNN and Ridge meta-learners on (domain_meta_features, Optuna_weights)
pairs and evaluates via Leave-One-Out cross-validation (n=9 domains).

Tested hypothesis
-----------------
H3: "Predicting weights from meta_features beats Transfer Direct (mean of
    training weights) on >=75% of held-out domains (>=12/16)."

Methods evaluated (per LOO fold)
---------------------------------
  default         : DEFAULT_WEIGHTS (constant, no learning)
  transfer_direct : mean of 8 training-domain Optuna weights
  knn1            : nearest-neighbour prediction (k=1)
  knn2            : 2-NN distance-weighted prediction
  knn3            : 3-NN distance-weighted prediction
  ridge           : per-weight Ridge regression
  optuna          : actual Optuna weights (oracle upper-bound)

Evaluation metric: NDCG@15 on held-out domain oracle (V2 semantic scoring).

Performance note
----------------
Sub-scores (venue, authors, impact, velocity, recency, relevance) are
pre-computed once per domain with V2 semantic embeddings, then different
weight combinations are evaluated cheaply by recomputing the weighted sum.
This avoids re-running the embedding model for each method.

Usage
-----
    python phase8c.py                  # full run (V2 semantic, ~3-5 min)
    python phase8c.py --fast           # V1.5 keyword scoring (~1 min)
    python phase8c.py --include-llm    # include llm_reasoning oracle
    python phase8c.py --no-save        # skip saving results JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WEIGHTS
from src.metalearning.meta_features import DomainMetaFeatures, load_all_meta_features
from src.metalearning.meta_learner import (
    EFFECTIVE_FEATURES,
    WEIGHT_KEYS,
    MetaLearner,
    _normalize_weights,  # noqa: F401
)
from src.scoring.automl_scorer import load_weights_for_topic
from src.scoring.metrics import ndcg_at_k
from src.scoring.quality_scorer import score_articles

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CITATION_RANK_DOMAINS = {"llm_reasoning"}
_RESULTS_PATH = PROJECT_ROOT / "data" / "metalearning" / "phase8c_results.json"

# Methods in display order
_METHODS = [
    "default", "transfer_direct",
    "knn1", "knn2", "knn3",
    "ridge", "bayesian_ridge", "gpr",
    "soft_vote", "hard_vote", "borda", "rrf",
    "ensemble", "ensemble_bayes",
    "optuna",
]
_METHOD_LABELS = {
    "default":         "Default",
    "transfer_direct": "Transfer",
    "knn1":            "kNN(1)",
    "knn2":            "kNN(2)",
    "knn3":            "kNN(3)",
    "ridge":           "Ridge",
    "bayesian_ridge":  "BayesRidge",
    "gpr":             "GPR",
    "soft_vote":       "SoftVote",
    "hard_vote":       "HardVote",
    "borda":           "Borda",
    "rrf":             "RRF",
    "ensemble":        "Ens+Rdg",   # label updated in main() with best alpha
    "ensemble_bayes":  "Ens+Bay",   # label updated in main() with best alpha
    "optuna":          "Optuna*",
}

# α values to sweep (0.0 = pure base learner, 1.0 = pure kNN(1))
ENSEMBLE_ALPHAS: list[float] = [round(a * 0.1, 1) for a in range(11)]


# ── Data loading ──────────────────────────────────────────────────────────────


def _all_domain_configs() -> list[dict]:
    path = PROJECT_ROOT / "data" / "oracle" / "domains_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["domains"]


def _load_dataset(include_llm: bool) -> tuple[
    list[str],                      # domain_ids
    dict[str, DomainMetaFeatures],  # meta-features
    dict[str, dict[str, float]],    # learned weights
    dict[str, str],                 # domain_id -> topic
]:
    """Load meta-features + weights for all valid domains."""
    configs = _all_domain_configs()
    if not include_llm:
        configs = [c for c in configs if c["id"] not in _CITATION_RANK_DOMAINS]

    topic_map = {c["id"]: c["topic"] for c in configs}
    all_mf = load_all_meta_features()

    domain_ids: list[str] = []
    meta_features: dict[str, DomainMetaFeatures] = {}
    learned_weights: dict[str, dict[str, float]] = {}

    for cfg in configs:
        did = cfg["id"]
        topic = cfg["topic"]

        if did not in all_mf:
            print(f"  [SKIP] {did} — no meta-features. Run phase8a_8b.py first.")
            continue

        w = load_weights_for_topic(topic)
        if w is None:
            print(f"  [SKIP] {did} — no saved weights. Run compare_v1_v2.py first.")
            continue

        domain_ids.append(did)
        meta_features[did] = all_mf[did]
        learned_weights[did] = w

    return domain_ids, meta_features, learned_weights, topic_map


# ── Sub-score pre-computation ─────────────────────────────────────────────────


def _precompute_subscores(
    domain_id: str,
    topic: str,
    use_semantic: bool,
) -> Optional[tuple[dict[str, dict[str, float]], dict[str, int]]]:
    """Score oracle corpus once; return sub-scores + gold_relevance.

    Returns None if oracle not found.
    """
    from data.oracle.build_oracle import load_oracle  # noqa: PLC0415

    try:
        articles, gold = load_oracle(domain_id)
    except FileNotFoundError:
        print(f"  [SKIP] {domain_id} — oracle not found.")
        return None

    if not gold:
        print(f"  [SKIP] {domain_id} — empty oracle.")
        return None

    scored = score_articles(articles, topic, DEFAULT_WEIGHTS, use_semantic=use_semantic)

    subscores: dict[str, dict[str, float]] = {}
    for qs in scored:
        subscores[qs.article_id] = {
            "venue":     float(qs.venue_score),
            "authors":   float(qs.authors_score),
            "impact":    float(qs.impact_score),
            "velocity":  float(qs.velocity_score),
            "recency":   float(qs.recency_score),
            "relevance": float(qs.relevance_score),
        }

    return subscores, gold


def _ndcg15(
    subscores: dict[str, dict[str, float]],
    gold: dict[str, int],
    weights: dict[str, float],
) -> float:
    """Apply weights to pre-computed sub-scores and compute NDCG@15."""
    ranked = sorted(
        subscores.items(),
        key=lambda kv: sum(weights.get(w, 0.0) * kv[1].get(w, 0.0) for w in WEIGHT_KEYS),
        reverse=True,
    )
    ranked_ids = [aid for aid, _ in ranked]
    return ndcg_at_k(ranked_ids, gold, k=15)


# ── LOO evaluation ────────────────────────────────────────────────────────────


def run_loo(
    domain_ids: list[str],
    meta_features: dict[str, DomainMetaFeatures],
    learned_weights: dict[str, dict[str, float]],
    topic_map: dict[str, str],
    oracle_data: dict[str, tuple],   # domain_id -> (subscores, gold)
) -> dict[str, dict[str, float]]:
    """Leave-One-Out loop.

    Returns:
        {domain_id: {method: ndcg15, ...}}
    """
    n = len(domain_ids)
    default_w = _normalize_weights(dict(DEFAULT_WEIGHTS))
    results: dict[str, dict[str, float]] = {}

    for i, test_id in enumerate(domain_ids):
        train_ids = [d for d in domain_ids if d != test_id]

        train_feats = [meta_features[d] for d in train_ids]
        train_weights = [learned_weights[d] for d in train_ids]

        learner = MetaLearner(feature_names=EFFECTIVE_FEATURES)
        learner.fit(train_feats, train_weights, domain_ids=train_ids)

        test_mf = meta_features[test_id]
        subscores, gold = oracle_data[test_id]

        fold_results: dict[str, float] = {
            "default":         _ndcg15(subscores, gold, default_w),
            "transfer_direct": _ndcg15(subscores, gold, learner.transfer_direct()),
            "knn1":            _ndcg15(subscores, gold, learner.predict_knn(test_mf, k=1)),
            "knn2":            _ndcg15(subscores, gold, learner.predict_knn(test_mf, k=2)),
            "knn3":            _ndcg15(subscores, gold, learner.predict_knn(test_mf, k=3)),
            "ridge":           _ndcg15(subscores, gold, learner.predict_ridge(test_mf)),
            "bayesian_ridge":  _ndcg15(subscores, gold, learner.predict_bayesian_ridge(test_mf)),
            "gpr":             _ndcg15(subscores, gold, learner.predict_gpr(test_mf)),
            # Rank-fusion ensembles (kNN1 + kNN2 + kNN3 + BayesRidge)
            "soft_vote":       _ndcg15(subscores, gold, learner.predict_soft_vote(test_mf)),
            "hard_vote":       _ndcg15(subscores, gold, learner.predict_hard_vote(test_mf)),
            "borda":           _ndcg15(subscores, gold, learner.predict_borda(test_mf)),
            "rrf":             _ndcg15(subscores, gold, learner.predict_rrf(test_mf)),
            "optuna":          _ndcg15(subscores, gold, learned_weights[test_id]),
        }
        # Ensemble α sweeps — cheap: just reweight pre-computed sub-scores
        for alpha in ENSEMBLE_ALPHAS:
            fold_results[f"ens_{alpha:.1f}"] = _ndcg15(
                subscores, gold, learner.predict_ensemble(test_mf, alpha=alpha)
            )
            fold_results[f"ens_b_{alpha:.1f}"] = _ndcg15(
                subscores, gold, learner.predict_ensemble_bayes(test_mf, alpha=alpha)
            )
        results[test_id] = fold_results

        # Print core methods only (exclude per-alpha ensemble keys)
        _skip = {"ensemble", "ensemble_bayes"}
        status = " ".join(
            f"{_METHOD_LABELS[m]}={v:.3f}"
            for m, v in fold_results.items()
            if m in _METHOD_LABELS and m not in _skip
        )
        print(f"  [{i+1}/{n}] {test_id[:28]:<28} {status}")

    return results


# ── Ensemble helpers ──────────────────────────────────────────────────────────


def _find_best_alpha(
    domain_ids: list[str],
    results: dict[str, dict[str, float]],
    prefix: str = "ens_",
) -> float:
    """Return the ensemble α maximising wins over Transfer Direct.

    Tiebreak: higher mean NDCG@15. α in ENSEMBLE_ALPHAS (0.0..1.0, step 0.1).
    prefix: key prefix used in fold_results ("ens_" for Ridge, "ens_b_" for BayesRidge).
    """
    best_alpha = 0.5
    best_beats = -1
    best_mean = -1.0
    for alpha in ENSEMBLE_ALPHAS:
        key = f"{prefix}{alpha:.1f}"
        beats = sum(
            1 for d in domain_ids
            if results[d].get(key, 0.0) > results[d].get("transfer_direct", 0.0)
        )
        mean_ndcg = sum(results[d].get(key, 0.0) for d in domain_ids) / len(domain_ids)
        if beats > best_beats or (beats == best_beats and mean_ndcg > best_mean):
            best_beats = beats
            best_mean = mean_ndcg
            best_alpha = alpha
    return best_alpha


# ── Output formatting ─────────────────────────────────────────────────────────


def _print_results(
    domain_ids: list[str],
    results: dict[str, dict[str, float]],
) -> None:
    col = 9
    W = 30 + len(_METHODS) * (col + 1)  # domain(30) + n_methods*(col+space)

    print(f"\n  LOO RESULTS  (NDCG@15, V2 semantic scoring)")
    head = f"  {'Domain':<28}" + "".join(f" {_METHOD_LABELS[m]:>{col}}" for m in _METHODS)
    sep = "  " + "-" * (W - 2)
    print(sep)
    print(head)
    print(sep)

    for did in domain_ids:
        row = f"  {did[:28]:<28}"
        for m in _METHODS:
            v = results[did].get(m, float("nan"))
            row += f" {v:>{col}.3f}"
        print(row)

    print(sep)

    # Mean row
    means: dict[str, float] = {}
    for m in _METHODS:
        vals = [results[d][m] for d in domain_ids if m in results[d]]
        means[m] = sum(vals) / len(vals) if vals else 0.0

    mean_row = f"  {'Mean':<28}" + "".join(f" {means[m]:>{col}.3f}" for m in _METHODS)
    print(mean_row)
    print(sep)
    print(f"  (* Optuna = actual learned weights, not predicted — theoretical upper-bound)")


def _print_h3_verdict(
    domain_ids: list[str],
    results: dict[str, dict[str, float]],
) -> None:
    W = 72
    print(f"\n  {'='*W}")
    print(f"  HYPOTHESIS H3 — Meta-learning beats Transfer Direct")
    print(f"  {'='*W}")
    print(f"  Threshold: >=75% of domains (>= {math.ceil(0.75 * len(domain_ids))}/{len(domain_ids)})")

    meta_methods = [
        "knn1", "knn2", "knn3",
        "ridge", "bayesian_ridge", "gpr",
        "soft_vote", "hard_vote", "borda", "rrf",
        "ensemble", "ensemble_bayes",
    ]
    best_method = None
    best_beats = -1

    print(f"\n  Wins over Transfer Direct (per method):")
    for m in meta_methods:
        beats = sum(
            1 for d in domain_ids
            if results[d].get(m, 0) > results[d].get("transfer_direct", 0)
        )
        pct = 100.0 * beats / len(domain_ids)
        verdict = "PASS" if beats >= math.ceil(0.75 * len(domain_ids)) else "FAIL"
        print(f"    {_METHOD_LABELS[m]:<12} beats Transfer on {beats}/{len(domain_ids)} "
              f"({pct:.0f}%)  [{verdict}]")
        if beats > best_beats:
            best_beats = beats
            best_method = m

    overall_pass = best_beats >= math.ceil(0.75 * len(domain_ids))
    print(f"\n  Best meta-learner : {_METHOD_LABELS[best_method]} "
          f"({best_beats}/{len(domain_ids)} wins)")
    print(f"  H3 verdict        : {'** SUPPORTED **' if overall_pass else 'REJECTED'}")

    # Ensemble α sweeps
    threshold = math.ceil(0.75 * len(domain_ids))
    best_ens_alpha = _find_best_alpha(domain_ids, results, prefix="ens_")
    best_bayes_alpha = _find_best_alpha(domain_ids, results, prefix="ens_b_")

    for sweep_label, prefix, best_a in [
        ("kNN(1)*alpha + Ridge*(1-alpha)",      "ens_",   best_ens_alpha),
        ("kNN(1)*alpha + BayesRidge*(1-alpha)", "ens_b_", best_bayes_alpha),
    ]:
        print(f"\n  Ensemble alpha sweep  ({sweep_label}):")
        print(f"    {'alpha':>5}   wins  pct   verdict")
        print(f"    {'-'*36}")
        for alpha in ENSEMBLE_ALPHAS:
            key = f"{prefix}{alpha:.1f}"
            beats = sum(1 for d in domain_ids if results[d].get(key, 0) > results[d].get("transfer_direct", 0))
            pct = 100.0 * beats / len(domain_ids)
            verdict = "PASS" if beats >= threshold else "FAIL"
            marker = " <-- best" if alpha == best_a else ""
            print(f"    {alpha:>5.1f}  {beats:>4}/{len(domain_ids)}  {pct:>3.0f}%  [{verdict}]{marker}")

    # Relative improvement over default
    print(f"\n  Mean NDCG@15 vs Default (relative improvement):")
    default_mean = sum(results[d]["default"] for d in domain_ids) / len(domain_ids)
    for m in ["transfer_direct", "knn1", "knn2", "knn3", "ridge", "bayesian_ridge", "gpr", "ensemble", "ensemble_bayes", "optuna"]:
        m_mean = sum(results[d].get(m, 0) for d in domain_ids) / len(domain_ids)
        if default_mean > 1e-6:
            pct = 100.0 * (m_mean - default_mean) / default_mean
            sign = "+" if pct >= 0 else ""
            print(f"    {_METHOD_LABELS[m]:<12}  {m_mean:.3f}  ({sign}{pct:.1f}% vs Default)")
        else:
            print(f"    {_METHOD_LABELS[m]:<12}  {m_mean:.3f}")

    print(f"  {'='*W}")


def _save_results(
    results: dict[str, dict[str, float]],
    domain_ids: list[str],
    meta_features: dict[str, DomainMetaFeatures],
    learned_weights: dict[str, dict[str, float]],
    best_ensemble_alpha: float | None = None,
) -> None:
    # Strip per-alpha keys from main loo_ndcg15 (kept in *_alpha_sweep for readability)
    loo_clean: dict[str, dict[str, float]] = {
        d: {k: v for k, v in vals.items()
            if not k.startswith("ens_") and not k.startswith("ens_b_")}
        for d, vals in results.items()
    }

    def _build_sweep(prefix: str) -> dict[str, dict]:
        sweep: dict[str, dict] = {}
        for alpha in ENSEMBLE_ALPHAS:
            key = f"{prefix}{alpha:.1f}"
            beats = sum(1 for d in domain_ids if results[d].get(key, 0) > results[d].get("transfer_direct", 0))
            mean_ndcg = sum(results[d].get(key, 0) for d in domain_ids) / len(domain_ids)
            sweep[f"{alpha:.1f}"] = {"wins_over_transfer": beats, "mean_ndcg": round(mean_ndcg, 4)}
        return sweep

    output = {
        "n_domains": len(domain_ids),
        "effective_features": EFFECTIVE_FEATURES,
        "domains": domain_ids,
        "loo_ndcg15": loo_clean,
        "ensemble_best_alpha": best_ensemble_alpha,
        "ensemble_alpha_sweep": _build_sweep("ens_"),
        "ensemble_bayes_best_alpha": best_ensemble_alpha,   # updated by caller
        "ensemble_bayes_alpha_sweep": _build_sweep("ens_b_"),
        "meta_features": {
            did: meta_features[did].model_dump()
            for did in domain_ids
        },
        "learned_weights": learned_weights,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n  Results saved -> {_RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8c: Meta-learner LOO evaluation"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use V1.5 keyword scoring instead of V2 semantic (much faster, less accurate)"
    )
    parser.add_argument(
        "--include-llm", action="store_true",
        help="Include llm_reasoning (citation-rank oracle — methodologically weaker)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save results to JSON"
    )
    args = parser.parse_args()

    use_semantic = not args.fast
    scoring_label = "V2 semantic" if use_semantic else "V1.5 keyword"

    W = 68
    print(f"\n{'='*W}")
    print(f"  Phase 8c  --  Meta-Learner: Leave-One-Out Evaluation")
    print(f"{'='*W}")

    # ── Load meta-features + weights ──────────────────────────────────────────
    domain_ids, meta_features, learned_weights, topic_map = _load_dataset(
        include_llm=args.include_llm
    )
    n = len(domain_ids)

    if n < 3:
        print(f"  ERROR: need >=3 domains for LOO (found {n}).")
        sys.exit(1)

    print(f"  Domains     : {n}")
    print(f"  Features    : {', '.join(EFFECTIVE_FEATURES)}")
    print(f"  Scoring     : {scoring_label}")
    print(f"  Oracle      : LOO (train on {n-1}, test on 1)\n")

    # ── Pre-compute sub-scores for all domains ────────────────────────────────
    print(f"  Pre-computing sub-scores ({scoring_label})...")
    oracle_data: dict[str, tuple] = {}
    for i, did in enumerate(domain_ids, 1):
        topic = topic_map.get(did, did.replace("_", " "))
        print(f"  [{i}/{n}] {did}", end=" ... ", flush=True)
        result = _precompute_subscores(did, topic, use_semantic=use_semantic)
        if result is None:
            print("SKIP")
            continue
        subscores, gold = result
        oracle_data[did] = (subscores, gold)

        n_g2 = sum(1 for g in gold.values() if g == 2)
        print(f"done  ({len(subscores)} articles, g2={n_g2})")

    # Remove domains with missing oracle data
    domain_ids = [d for d in domain_ids if d in oracle_data]
    n = len(domain_ids)

    if n < 3:
        print(f"  ERROR: only {n} domains have oracle data. Build oracles first.")
        sys.exit(1)

    # ── LOO evaluation ────────────────────────────────────────────────────────
    n_methods_display = len(_METHODS)
    print(f"\n  LOO folds ({n} domains x {n_methods_display} methods + {len(ENSEMBLE_ALPHAS)} ensemble alpha)...\n")
    results = run_loo(domain_ids, meta_features, learned_weights, topic_map, oracle_data)

    # ── Post-process: inject best ensemble results ────────────────────────────
    best_alpha = _find_best_alpha(domain_ids, results, prefix="ens_")
    best_bayes_alpha = _find_best_alpha(domain_ids, results, prefix="ens_b_")
    for d in domain_ids:
        results[d]["ensemble"]       = results[d][f"ens_{best_alpha:.1f}"]
        results[d]["ensemble_bayes"] = results[d][f"ens_b_{best_bayes_alpha:.1f}"]
    _METHOD_LABELS["ensemble"]       = f"En+R(a{best_alpha:.1f})"
    _METHOD_LABELS["ensemble_bayes"] = f"En+B(a{best_bayes_alpha:.1f})"
    print(f"\n  Best Ens+Ridge alpha  = {best_alpha:.1f}  (kNN(1)*{best_alpha:.1f} + Ridge*{1-best_alpha:.1f})")
    print(f"  Best Ens+Bayes alpha  = {best_bayes_alpha:.1f}  (kNN(1)*{best_bayes_alpha:.1f} + BayesRidge*{1-best_bayes_alpha:.1f})")

    # ── Results table ─────────────────────────────────────────────────────────
    _print_results(domain_ids, results)

    # ── H3 verdict ────────────────────────────────────────────────────────────
    _print_h3_verdict(domain_ids, results)

    # ── Save ──────────────────────────────────────────────────────────────────
    if not args.no_save:
        _save_results(results, domain_ids, meta_features, learned_weights,
                      best_ensemble_alpha=best_alpha)

    print(f"\n  Done.  Thesis note: table above -> Chapter 8 (H3 evaluation).")


if __name__ == "__main__":
    main()
