"""Promote top grade-1 articles to grade-2 in an existing oracle.

Problem being solved
--------------------
When oracle surveys don't overlap enough in their bibliographies, the standard
build_oracle.py logic produces 0 grade-2 articles (none cited by >=2 surveys).
NDCG@15 = 0 with no grade-2 -> Optuna has nothing to optimize against.

Solution
--------
Among the existing grade-1 articles, promote the "most survey-like" ones to
grade-2 using a composite promotion score:

    promotion_score = survey_bonus + citation_score

    survey_bonus = 3.0  if title contains survey/review/overview/systematic/
                           comprehensive  (case-insensitive)
               = 0.0  otherwise

    citation_score = article.citation_count / max_citation_count_in_corpus
                     (normalized to [0, 1] within the grade-1 pool)

Why surveys?  Review/survey papers are inherently "highly relevant" to a domain
--- they synthesize it.  Promoting them as grade-2 gives Optuna a meaningful
signal: articles that are both well-cited AND topically representative.

Usage
-----
    # Dry-run: see candidates without saving
    python promote_oracle_grade2.py --domain medical_image_segmentation

    # Promote top 5 (default) and save
    python promote_oracle_grade2.py --domain medical_image_segmentation --apply

    # Promote a custom number
    python promote_oracle_grade2.py --domain medical_image_segmentation --n 8 --apply

    # See all domains available
    python promote_oracle_grade2.py --list-domains
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

ORACLE_DIR = PROJECT_ROOT / "data" / "oracle"
CONFIG_FILE = ORACLE_DIR / "domains_config.json"

# ── Survey keyword bonus ───────────────────────────────────────────────────────
_SURVEY_KEYWORDS = frozenset([
    "survey", "review", "overview", "systematic", "comprehensive",
    "literature", "taxonomy", "tutorial", "benchmark", "roadmap",
])
_SURVEY_BONUS = 3.0   # equivalent to 3x the max-citation score
_DEFAULT_PROMOTE_N = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_survey_like(title: str) -> bool:
    """Return True if the title contains at least one survey keyword."""
    lower = title.lower()
    return any(kw in lower for kw in _SURVEY_KEYWORDS)


def _promotion_score(article: dict, max_citations: float) -> float:
    """Composite score used to rank grade-1 articles for promotion.

    score = survey_bonus + citation_score
    survey_bonus  = 3.0 if title is survey-like, else 0.0
    citation_score = citation_count / max_citations  (normalized [0,1])
    """
    bonus = _SURVEY_BONUS if _is_survey_like(article.get("title", "")) else 0.0
    cit = article.get("citation_count") or 0
    cit_score = (cit / max_citations) if max_citations > 0 else 0.0
    return bonus + cit_score


def _load_oracle(domain_id: str) -> tuple[list[dict], dict[str, int]]:
    """Load raw JSON oracle files (articles as dicts, not Article objects)."""
    domain_dir = ORACLE_DIR / domain_id
    art_path = domain_dir / "gold_articles.json"
    rel_path = domain_dir / "gold_relevance.json"

    if not art_path.exists() or not rel_path.exists():
        raise FileNotFoundError(
            f"Oracle not found for '{domain_id}'. "
            f"Run:  python data/oracle/build_oracle.py --domain {domain_id}"
        )

    articles: list[dict] = json.loads(art_path.read_text(encoding="utf-8"))
    gold: dict[str, int] = json.loads(rel_path.read_text(encoding="utf-8"))
    return articles, gold


def _list_domains() -> list[str]:
    """Return all domain IDs from domains_config.json."""
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return [d["id"] for d in cfg["domains"]]


# ── Core logic ─────────────────────────────────────────────────────────────────

def find_promotion_candidates(
    articles: list[dict],
    gold: dict[str, int],
    n: int = _DEFAULT_PROMOTE_N,
) -> list[dict]:
    """Return the top-N grade-1 articles ranked by promotion_score.

    Args:
        articles: raw article dicts from gold_articles.json
        gold: current {article_id: grade} dict
        n: how many candidates to return

    Returns:
        List of article dicts, sorted by promotion_score descending.
        These are the candidates to be promoted to grade-2.
    """
    # Build a fast lookup: id -> article dict
    id_to_art = {a["id"]: a for a in articles}

    # Collect current grade-1 articles (only those present in corpus)
    grade1_ids = [aid for aid, g in gold.items() if g == 1 and aid in id_to_art]
    grade1_arts = [id_to_art[aid] for aid in grade1_ids]

    if not grade1_arts:
        return []

    # Compute max citations among grade-1 pool for normalization
    max_cit = max(
        (a.get("citation_count") or 0) for a in grade1_arts
    )

    # Rank by promotion score
    ranked = sorted(
        grade1_arts,
        key=lambda a: _promotion_score(a, max_cit),
        reverse=True,
    )

    return ranked[:n]


def promote_to_grade2(
    domain_id: str,
    n: int = _DEFAULT_PROMOTE_N,
    apply: bool = False,
    select_indices: list[int] | None = None,
    force: bool = False,
) -> list[str]:
    """Find and optionally promote the top-N grade-1 articles to grade-2.

    Args:
        domain_id: oracle domain (e.g. 'medical_image_segmentation')
        n: number of articles to promote (used when select_indices is None)
        apply: if True, overwrite gold_relevance.json; if False, dry-run only
        select_indices: 1-based indices from the citation-sorted grade-1 list;
                        when provided, overrides auto top-N selection
        force: skip the "already has grade-2" guard

    Returns:
        List of article IDs that were (or would be) promoted.
    """
    articles, gold = _load_oracle(domain_id)
    domain_dir = ORACLE_DIR / domain_id

    # Current grade summary
    n_grade2_before = sum(1 for g in gold.values() if g == 2)
    n_grade1_before = sum(1 for g in gold.values() if g == 1)
    n_total_corpus  = len(articles)

    print(f"\n{'='*60}")
    print(f"Domain : {domain_id}")
    print(f"Corpus : {n_total_corpus} articles")
    print(f"Before : grade-2={n_grade2_before}  grade-1={n_grade1_before}")
    print(f"{'='*60}")

    if n_grade2_before > 0 and not force:
        print(
            f"\n[INFO] Domain already has {n_grade2_before} grade-2 articles.\n"
            f"       Promotion is optional -- use --force to add more."
        )

    # Build grade-1 list sorted by citations (reference order for --select)
    grade1_arts = [a for a in articles if gold.get(a["id"]) == 1]
    grade1_by_cit = sorted(
        grade1_arts,
        key=lambda a: a.get("citation_count") or 0,
        reverse=True,
    )

    if select_indices:
        # Manual selection: pick specific 1-based indices from citation-sorted list
        candidates = []
        invalid = []
        for idx in select_indices:
            if 1 <= idx <= len(grade1_by_cit):
                candidates.append(grade1_by_cit[idx - 1])
            else:
                invalid.append(idx)
        if invalid:
            print(f"\n[WARN] Indices out of range (1-{len(grade1_by_cit)}): {invalid}")
        if not candidates:
            print("\n[WARN] No valid articles selected.")
            return []
    else:
        candidates = find_promotion_candidates(articles, gold, n=n)
        if not candidates:
            print("\n[WARN] No grade-1 articles found in corpus -- nothing to promote.")
            return []

    # Build max_cit for display
    max_cit = max((a.get("citation_count") or 0) for a in grade1_arts) if grade1_arts else 1

    header = "Selected" if select_indices else f"Top {len(candidates)} auto-scored"
    print(f"\n{header} candidates for grade-2 promotion:\n")
    print(f"  {'#':<3}  {'Score':>6}  {'Cit':>6}  {'Survey?':<8}  Title")
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*50}")

    promoted_ids: list[str] = []
    for i, art in enumerate(candidates, 1):
        score  = _promotion_score(art, max_cit)
        cit    = art.get("citation_count") or 0
        survey = "YES" if _is_survey_like(art.get("title", "")) else "no"
        title  = (art.get("title") or "")[:60]
        print(f"  {i:<3}  {score:>6.2f}  {cit:>6}  {survey:<8}  {title}")
        promoted_ids.append(art["id"])

    if not apply:
        print(
            f"\n[DRY RUN] {len(promoted_ids)} articles would be promoted to grade-2."
            f"\n          Re-run with --apply to save changes."
        )
        return promoted_ids

    # Apply: update gold_relevance.json
    new_gold = dict(gold)
    for aid in promoted_ids:
        new_gold[aid] = 2

    n_grade2_after = sum(1 for g in new_gold.values() if g == 2)
    n_grade1_after = sum(1 for g in new_gold.values() if g == 1)

    rel_path = domain_dir / "gold_relevance.json"
    rel_path.write_text(
        json.dumps(new_gold, indent=2),
        encoding="utf-8",
    )

    print(f"\n[OK] gold_relevance.json updated:")
    print(f"     Before: grade-2={n_grade2_before}  grade-1={n_grade1_before}")
    print(f"     After : grade-2={n_grade2_after}  grade-1={n_grade1_after}")
    print(f"     File  : {rel_path}")
    print(
        f"\n     Next step: run Optuna with the updated oracle:\n"
        f"     python demo_phase3.py \"{domain_id.replace('_', ' ')}\""
    )

    return promoted_ids


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote top grade-1 articles to grade-2 in an oracle domain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python promote_oracle_grade2.py --domain medical_image_segmentation
  python promote_oracle_grade2.py --domain medical_image_segmentation --apply
  python promote_oracle_grade2.py --domain medical_image_segmentation --n 8 --apply
  python promote_oracle_grade2.py --list-domains
        """,
    )
    parser.add_argument(
        "--domain",
        help="Oracle domain ID (e.g. 'medical_image_segmentation').",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_PROMOTE_N,
        help=f"Number of grade-1 articles to promote (default: {_DEFAULT_PROMOTE_N}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the updated gold_relevance.json (default: dry-run).",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available oracle domain IDs and exit.",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        metavar="INDICES",
        help=(
            "Comma-separated 1-based indices from the citation-sorted grade-1 list "
            "(e.g. '6,7,8,9,10'). Overrides --n. "
            "Run without --apply first to see the numbered list."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the 'already has grade-2' guard (needed when adding to existing grade-2).",
    )

    args = parser.parse_args()

    if args.list_domains:
        domains = _list_domains()
        print("Available domains:")
        for d in domains:
            print(f"  {d}")
        return

    if not args.domain:
        parser.error("--domain is required (use --list-domains to see options).")

    # Parse --select indices
    select_indices: list[int] | None = None
    if args.select:
        try:
            select_indices = [int(x.strip()) for x in args.select.split(",") if x.strip()]
        except ValueError:
            parser.error("--select must be comma-separated integers, e.g. '6,7,8,9,10'")

    promote_to_grade2(
        domain_id=args.domain,
        n=args.n,
        apply=args.apply,
        select_indices=select_indices,
        force=args.force,
    )


if __name__ == "__main__":
    main()
