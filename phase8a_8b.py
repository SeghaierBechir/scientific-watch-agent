"""Phase 8a + 8b — Meta-feature extraction and correlation analysis.

Phase 8a  Extract 13 quantitative meta-features from each oracle corpus and
          save them to data/metalearning/.  These features describe the
          structural properties of each research domain (temporal dynamics,
          citation inequality, community size, venue concentration, semantic
          coherence).

Phase 8b  Correlate meta-features with the Optuna-learned scoring weights
          stored in data/weights/.  High |r| values reveal which corpus
          properties predict which scoring dimensions — this is the empirical
          foundation for the Phase-8c meta-learner.

Thesis context (Chapter 8 — Meta-learning)
------------------------------------------
Research question: "Which structural properties of a research domain
predict the optimal scoring weights found by Optuna?"

Ideal interpretation of correlations:
  pct_recent     ↑  →  velocity ↑   (fast-moving field → prioritise momentum)
  citation_gini  ↑  →  impact ↑    (few dominant papers → citation count matters)
  uniq_author_r. ↑  →  authors ↓   (large community → individual h-index less useful)
  topic_overlap  ↑  →  relevance ↓  (semantically focused corpus → relevance is trivial)

Usage
-----
    # Run 8a + 8b on all oracle domains
    python phase8a_8b.py

    # Skip 8b (only extract and save features)
    python phase8a_8b.py --no-correlation

    # Include llm_reasoning (citation-rank oracle, weaker signal)
    python phase8a_8b.py --include-llm

Note: weights from data/weights/ should be freshly learned on the current
oracle.  If you just rebuilt the oracles, run first:
    python compare_v1_v2.py --all --force-rerun
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metalearning.meta_features import (
    FEATURE_NAMES,
    DomainMetaFeatures,
    extract_meta_features,
    load_all_meta_features,
    save_meta_features,
)
from src.scoring.automl_scorer import load_weights_for_topic

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_WEIGHT_KEYS = ["venue", "authors", "impact", "velocity", "recency", "relevance"]

# Domains with citation-rank oracles (methodologically weaker — excluded by default)
_CITATION_RANK_DOMAINS = {"llm_reasoning"}


# ── Oracle helpers ────────────────────────────────────────────────────────────

def _all_domain_configs() -> list[dict]:
    config_path = PROJECT_ROOT / "data" / "oracle" / "domains_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))["domains"]


def _load_oracle(domain_id: str):
    from data.oracle.build_oracle import load_oracle  # noqa: PLC0415
    return load_oracle(domain_id)


# ── Pearson correlation ───────────────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r for two equal-length sequences. Returns 0.0 on degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return num / (sx * sy)


# ── Phase 8a — Feature extraction ────────────────────────────────────────────

def run_phase_8a(domain_configs: list[dict]) -> dict[str, DomainMetaFeatures]:
    W = 68
    print(f"\n{'='*W}")
    print(f"  Phase 8a  --  Domain Meta-Feature Extraction")
    print(f"{'='*W}")
    print(f"  Domains : {len(domain_configs)}")
    print(f"  Output  : data/metalearning/*_meta_features.json\n")

    results: dict[str, DomainMetaFeatures] = {}
    for i, cfg in enumerate(domain_configs, 1):
        domain_id = cfg["id"]
        topic = cfg["topic"]
        print(f"  [{i}/{len(domain_configs)}] {domain_id}", end=" ... ", flush=True)

        try:
            articles, gold = _load_oracle(domain_id)
        except FileNotFoundError:
            print("SKIP (oracle not found)")
            continue

        if not gold:
            print("SKIP (empty gold set)")
            continue

        n_g2 = sum(1 for g in gold.values() if g == 2)
        n_g1 = sum(1 for g in gold.values() if g == 1)
        features = extract_meta_features(domain_id, topic, articles, gold)
        save_meta_features(features)
        results[domain_id] = features
        print(f"done  ({len(articles)} articles, g2={n_g2}, g1={n_g1})")

    # ── Summary table ──────────────────────────────────────────────────────────
    if not results:
        print("\n  No features extracted.")
        return results

    domains = list(results.keys())
    short = [d[:12] for d in domains]   # truncate for display

    _print_feature_table(results, domains, short)
    return results


def _print_feature_table(
    results: dict[str, DomainMetaFeatures],
    domains: list[str],
    short: list[str],
) -> None:
    col = 10
    sep_w = 20 + len(domains) * (col + 1)
    print(f"\n  META-FEATURES SUMMARY")
    print(f"  {'-'*sep_w}")

    # header
    header = f"  {'Feature':<20}"
    for s in short:
        header += f" {s:>{col}}"
    print(header)
    print(f"  {'-'*sep_w}")

    # Display format per feature
    fmt = {
        "median_year":          lambda v: f"{v:.1f}",
        "pct_recent":           lambda v: f"{v:.3f}",
        "year_std":             lambda v: f"{v:.2f}",
        "citation_gini":        lambda v: f"{v:.3f}",
        "citation_median":      lambda v: f"{v:.0f}",
        "pct_high_cited":       lambda v: f"{v:.3f}",
        "unique_author_ratio":  lambda v: f"{v:.3f}",
        "mean_h_index":         lambda v: f"{v:.1f}",
        "pct_high_hindex":      lambda v: f"{v:.3f}",
        "pct_q1":               lambda v: f"{v:.3f}",
        "topic_concept_overlap":lambda v: f"{v:.3f}",
        "gold_ratio":           lambda v: f"{v:.3f}",
        "grade2_ratio":         lambda v: f"{v:.3f}",
    }

    for feat in FEATURE_NAMES:
        row = f"  {feat:<20}"
        for d in domains:
            val = getattr(results[d], feat)
            row += f" {fmt[feat](val):>{col}}"
        print(row)

    print(f"  {'-'*sep_w}")


# ── Phase 8b — Correlation analysis ──────────────────────────────────────────

def run_phase_8b(meta_features: dict[str, DomainMetaFeatures], domain_configs: list[dict]) -> None:
    W = 68
    print(f"\n{'='*W}")
    print(f"  Phase 8b  --  Correlation: Meta-Features vs Learned Weights")
    print(f"{'='*W}")
    n = len(meta_features)
    print(f"  (Pearson r, n={n} domains — indicative only; |r|>0.80 = strong signal)\n")

    # ── Load learned weights for each domain ──────────────────────────────────
    topic_map = {cfg["id"]: cfg["topic"] for cfg in domain_configs}
    weights_by_domain: dict[str, dict[str, float]] = {}

    for domain_id in meta_features:
        topic = topic_map.get(domain_id, domain_id.replace("_", " "))
        w = load_weights_for_topic(topic)
        if w is None:
            print(f"  [WARN] No saved weights for '{domain_id}' — run compare_v1_v2.py first")
        else:
            weights_by_domain[domain_id] = w

    loaded = [d for d in meta_features if d in weights_by_domain]
    n_loaded = len(loaded)
    print(f"  Weights loaded: {n_loaded}/{len(meta_features)} domains")

    if n_loaded == 0:
        print("  Nothing to correlate. Run compare_v1_v2.py --all first.")
        return

    # ── Weight profiles table ─────────────────────────────────────────────────
    print(f"\n  LEARNED WEIGHT PROFILES (V2, from data/weights/)")
    col = 10
    print(f"  {'-'*(20 + len(_WEIGHT_KEYS)*(col+1))}")
    hdr = f"  {'Domain':<20}" + "".join(f" {k:>{col}}" for k in _WEIGHT_KEYS)
    print(hdr)
    print(f"  {'-'*(20 + len(_WEIGHT_KEYS)*(col+1))}")
    for d in loaded:
        row = f"  {d[:20]:<20}"
        for k in _WEIGHT_KEYS:
            row += f" {weights_by_domain[d].get(k, 0.0):>{col}.4f}"
        print(row)
    print(f"  {'-'*(20 + len(_WEIGHT_KEYS)*(col+1))}")

    # ── Compute correlations ──────────────────────────────────────────────────
    # For each (feature, weight): collect values across all loaded domains
    feat_vals: dict[str, list[float]] = {f: [] for f in FEATURE_NAMES}
    wgt_vals: dict[str, list[float]] = {k: [] for k in _WEIGHT_KEYS}

    for d in loaded:
        feats = meta_features[d]
        wgts = weights_by_domain[d]
        for f in FEATURE_NAMES:
            feat_vals[f].append(getattr(feats, f))
        for k in _WEIGHT_KEYS:
            wgt_vals[k].append(wgts.get(k, 0.0))

    corr: dict[str, dict[str, float]] = {}
    for f in FEATURE_NAMES:
        corr[f] = {}
        for k in _WEIGHT_KEYS:
            corr[f][k] = _pearson(feat_vals[f], wgt_vals[k])

    # Sort features by max |r| across all weights (most informative first)
    sorted_features = sorted(
        FEATURE_NAMES,
        key=lambda f: max(abs(corr[f][k]) for k in _WEIGHT_KEYS),
        reverse=True,
    )

    # ── Correlation table ─────────────────────────────────────────────────────
    c = 9
    print(f"\n  PEARSON r  (sorted by max |r| across weights)")
    sep_w = 22 + len(_WEIGHT_KEYS) * (c + 1) + 8
    print(f"  {'-'*sep_w}")
    hdr = f"  {'Feature':<22}" + "".join(f" {k:>{c}}" for k in _WEIGHT_KEYS) + f"  {'max|r|':>6}"
    print(hdr)
    print(f"  {'-'*sep_w}")

    for f in sorted_features:
        row = f"  {f:<22}"
        max_r = 0.0
        for k in _WEIGHT_KEYS:
            r = corr[f][k]
            if abs(r) > max_r:
                max_r = abs(r)
            sign = "+" if r >= 0 else ""
            row += f" {sign}{r:>{c-1}.2f}"
        row += f"  {max_r:>6.2f}"
        print(row)

    print(f"  {'-'*sep_w}")

    # ── Strongest predictors (|r| > 0.40, threshold aligned with EFFECTIVE_FEATURES) ─
    _DISPLAY_THRESHOLD = 0.40
    print(f"\n  STRONGEST PREDICTORS  (|r| > {_DISPLAY_THRESHOLD:.2f})")
    found_any = False
    for f in sorted_features:
        for k in _WEIGHT_KEYS:
            r = corr[f][k]
            if abs(r) >= _DISPLAY_THRESHOLD:
                print(f"    {f:<25} -> {k:<10}  r={r:+.2f}  "
                      f"(higher {f} => {'more' if r>0 else 'less'} {k} weight)")
                found_any = True
    if not found_any:
        print(f"    None found with |r| > {_DISPLAY_THRESHOLD:.2f} on n={n_loaded} domains.")

    print(f"\n  Thesis note: update this table in Chapter 8 after running")
    print(f"  compare_v1_v2.py --all --force-rerun on the rebuilt oracles.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8a+8b: meta-feature extraction and weight correlation"
    )
    parser.add_argument("--no-correlation", action="store_true",
                        help="Only run Phase 8a (skip correlation analysis)")
    parser.add_argument("--include-llm", action="store_true",
                        help="Include llm_reasoning (citation-rank oracle — weaker signal)")
    args = parser.parse_args()

    all_configs = _all_domain_configs()
    if not args.include_llm:
        all_configs = [c for c in all_configs if c["id"] not in _CITATION_RANK_DOMAINS]

    # Phase 8a
    meta_features = run_phase_8a(all_configs)

    if not meta_features:
        print("\n  No domains processed. Build oracles first:")
        print("  python data/oracle/build_oracle.py --domain <id>")
        sys.exit(1)

    # Phase 8b
    if not args.no_correlation:
        run_phase_8b(meta_features, all_configs)

    print(f"\n  Done.  Next: run Phase 8c (meta-learner) once weights are refreshed.")


if __name__ == "__main__":
    main()
