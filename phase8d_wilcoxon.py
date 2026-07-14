"""Phase 8d — Wilcoxon signed-rank test for H3.

For each meta-learner, tests whether NDCG@15 improvements over Transfer Direct
are statistically significant across 19 LOO-CV domains.

Null hypothesis  H0 : median(delta) = 0   (no improvement)
Alt. hypothesis  H1 : median(delta) > 0   (meta-learner beats Transfer Direct)
Test             : Wilcoxon signed-rank (one-sided, paired, non-parametric)
Significance     : α = 0.05

Effect size: rank-biserial correlation
    r_rb = 1 - (2 * W_neg) / (n*(n+1)/2)
    Interpretation: |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large

Usage
-----
    python phase8d_wilcoxon.py
    python phase8d_wilcoxon.py --plot        # also save bar chart of p-values
    python phase8d_wilcoxon.py --alpha 0.10  # custom significance level
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent
RESULTS_PATH = PROJECT_ROOT / "data" / "metalearning" / "phase8c_results.json"
PLOTS_DIR    = PROJECT_ROOT / "data" / "plots"

EVAL_METHODS = [
    "knn1", "knn2", "knn3",
    "ridge", "bayesian_ridge", "gpr",
    "soft_vote", "hard_vote", "borda", "rrf",
    "ensemble", "ensemble_bayes",
]
EVAL_LABELS = {
    "knn1":          "kNN(1)",
    "knn2":          "kNN(2)",
    "knn3":          "kNN(3)",
    "ridge":         "Ridge",
    "bayesian_ridge":"BayesRidge",
    "gpr":           "GPR",
    "soft_vote":     "SoftVote",
    "hard_vote":     "HardVote",
    "borda":         "Borda",
    "rrf":           "RRF",
    "ensemble":      "En+R",
    "ensemble_bayes":"En+B",
}


def _rank_biserial(deltas: np.ndarray) -> float:
    """Rank-biserial correlation as effect size for the Wilcoxon test.

    r_rb = (W+ - W-) / (W+ + W-)
    where W+ = sum of positive ranks, W- = sum of negative ranks.
    """
    nonzero = deltas[deltas != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus  = ranks[nonzero > 0].sum()
    w_minus = ranks[nonzero < 0].sum()
    denom = w_plus + w_minus
    return float((w_plus - w_minus) / denom) if denom > 0 else 0.0


def run_wilcoxon(alpha: float = 0.05) -> list[dict]:
    data    = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    loo     = data["loo_ndcg15"]
    domains = data["domains"]
    n       = len(domains)

    transfer_vals = np.array([loo[d]["transfer_direct"] for d in domains])

    results = []
    for method in EVAL_METHODS:
        method_vals = np.array([loo[d].get(method, 0.0) for d in domains])
        deltas = method_vals - transfer_vals

        n_wins   = int((deltas > 1e-9).sum())
        n_losses = int((deltas < -1e-9).sum())
        n_ties   = n - n_wins - n_losses
        mean_delta = float(deltas.mean())

        # Wilcoxon signed-rank (one-sided: alternative='greater')
        # Requires at least 1 non-zero difference
        nonzero = deltas[np.abs(deltas) > 1e-9]
        if len(nonzero) < 5:
            stat, pval = np.nan, np.nan
        else:
            stat, pval = stats.wilcoxon(deltas, alternative="greater", zero_method="zsplit")

        r_rb     = _rank_biserial(deltas)
        sig      = not np.isnan(pval) and pval < alpha

        results.append({
            "method":     method,
            "label":      EVAL_LABELS[method],
            "n_wins":     n_wins,
            "n_losses":   n_losses,
            "n_ties":     n_ties,
            "mean_delta": mean_delta,
            "W_stat":     stat,
            "p_value":    pval,
            "r_rb":       r_rb,
            "significant": sig,
        })

    # Sort by p-value ascending (most significant first)
    results.sort(key=lambda x: (x["p_value"] if not np.isnan(x["p_value"]) else 999))
    return results, n


def _effect_label(r: float) -> str:
    if abs(r) >= 0.5:
        return "large"
    if abs(r) >= 0.3:
        return "medium"
    return "small"


def print_results(results: list[dict], n: int, alpha: float) -> None:
    print()
    print("=" * 90)
    print(f"  WILCOXON SIGNED-RANK TEST — H3: meta-learning > Transfer Direct")
    print(f"  n = {n} domains (LOO-CV)  |  one-sided  |  α = {alpha}")
    print("=" * 90)
    print(f"  {'Method':<12}  {'Wins':>5}  {'Loss':>5}  {'Ties':>4}  "
          f"{'ΔMean':>7}  {'W+':>7}  {'p-value':>9}  {'r_rb':>6}  {'Effect':>7}  {'Sig?':>5}")
    print(f"  {'-'*87}")

    for r in results:
        pval_str = f"{r['p_value']:.4f}" if not np.isnan(r['p_value']) else "   N/A"
        stat_str = f"{r['W_stat']:.1f}"  if not np.isnan(r['W_stat'])  else "  N/A"
        sig_str  = "✓ YES" if r["significant"] else "  no"
        eff      = _effect_label(r["r_rb"])
        print(
            f"  {r['label']:<12}  {r['n_wins']:>5}  {r['n_losses']:>5}  {r['n_ties']:>4}  "
            f"{r['mean_delta']:>+7.4f}  {stat_str:>7}  {pval_str:>9}  "
            f"{r['r_rb']:>+6.3f}  {eff:>7}  {sig_str:>5}"
        )

    print(f"  {'-'*87}")

    # Summary: best method for H3
    sig_methods = [r for r in results if r["significant"]]
    best = results[0]
    print()
    print(f"  Best p-value  : {best['label']} (p={best['p_value']:.4f}, "
          f"r_rb={best['r_rb']:+.3f}, {_effect_label(best['r_rb'])} effect)")
    if sig_methods:
        print(f"  Significant at α={alpha}: {', '.join(r['label'] for r in sig_methods)}")
    else:
        print(f"  No method reaches significance at α={alpha} with n={n}")
    print()
    print("  Thesis note: report W+, p-value, and r_rb for Borda in Section 8.4 (H3).")
    print("=" * 90)


def save_plot(results: list[dict], n: int, alpha: float) -> None:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.family": "sans-serif"})
    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")
        return

    labels   = [r["label"]   for r in results]
    pvals    = [r["p_value"] for r in results]
    r_rbs    = [r["r_rb"]    for r in results]
    sig_mask = [r["significant"] for r in results]

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"Wilcoxon signed-rank test — H3: meta-learner > Transfer Direct\n"
        f"n={n} domains (LOO-CV), one-sided, α={alpha}",
        fontsize=11, fontweight="bold",
    )

    # ── Top: p-values ─────────────────────────────────────────────────────────
    bar_colors = ["#2ecc71" if s else "#e74c3c" for s in sig_mask]
    bars = ax1.bar(x, pvals, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(alpha, color="#e67e22", linewidth=1.5, linestyle="--",
                label=f"α = {alpha}")
    ax1.set_ylabel("p-value (one-sided)", fontsize=9)
    ax1.set_ylim(0, max(max(p for p in pvals if not np.isnan(p)) * 1.15, alpha * 2))
    ax1.legend(fontsize=8.5)
    ax1.set_yticks([0, 0.05, 0.10, 0.20, 0.30, 0.50])

    for bar, pv, s in zip(bars, pvals, sig_mask):
        if not np.isnan(pv):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{pv:.3f}", ha="center", va="bottom", fontsize=7.5,
                     color="#1a5e35" if s else "#7b0000")

    # ── Bottom: rank-biserial effect size ─────────────────────────────────────
    rr_colors = ["#27ae60" if v > 0 else "#c0392b" for v in r_rbs]
    ax2.bar(x, r_rbs, color=rr_colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0,   color="#555", linewidth=0.8)
    ax2.axhline( 0.3, color="#aaa", linewidth=1, linestyle=":")
    ax2.axhline( 0.5, color="#888", linewidth=1, linestyle="--")
    ax2.axhline(-0.3, color="#aaa", linewidth=1, linestyle=":")
    ax2.set_ylabel("Rank-biserial r (effect size)", fontsize=9)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)

    for i, (rv, s) in enumerate(zip(r_rbs, sig_mask)):
        offset = 0.03 if rv >= 0 else -0.06
        ax2.text(i, rv + offset, f"{rv:+.2f}", ha="center", va="bottom",
                 fontsize=7, color="#1a5e35" if rv > 0 else "#7b0000")

    ax2.text(12.6,  0.5, "large", va="center", fontsize=7, color="#888")
    ax2.text(12.6,  0.3, "medium", va="center", fontsize=7, color="#aaa")

    # Legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#2ecc71", label=f"p < {alpha}  (significant)"),
        Patch(facecolor="#e74c3c", label=f"p ≥ {alpha}  (not significant)"),
    ]
    ax1.legend(handles=legend_els + [ax1.lines[0]], fontsize=8, loc="upper right")

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "phase8d_wilcoxon.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {save_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wilcoxon test for H3")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--plot", action="store_true",
                        help="Also save a bar chart of p-values and effect sizes")
    args = parser.parse_args()

    results, n = run_wilcoxon(alpha=args.alpha)
    print_results(results, n, alpha=args.alpha)

    if args.plot:
        save_plot(results, n, alpha=args.alpha)


if __name__ == "__main__":
    main()
