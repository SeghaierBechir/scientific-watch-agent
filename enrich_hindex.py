"""Enrich oracle articles with author h-index from OpenAlex.

mean_h_index and pct_high_hindex are currently 0 in all 16 oracle domains
because the works endpoint does not return h-index.  This script:

  1. Collects all unique author OpenAlex IDs across every oracle domain.
  2. Fetches h-index in batches of 100 via the authors endpoint.
  3. Saves a local cache: data/hindex_cache.json
  4. Updates every oracle gold_articles.json in-place (fills h_index field).

After running this script, re-run the full evaluation pipeline:

    python compare_v1_v2.py --all --force-rerun
    python phase8a_8b.py
    python phase8c.py

Usage
-----
    python enrich_hindex.py                # full run
    python enrich_hindex.py --dry-run      # count only, no API calls
    python enrich_hindex.py --cache-only   # build cache, skip JSON update
    python enrich_hindex.py --no-update    # build cache only
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).parent
ORACLE_DIR    = PROJECT_ROOT / "data" / "oracle"
CACHE_PATH    = PROJECT_ROOT / "data" / "hindex_cache.json"
OPENALEX_EMAIL = "seghaierbechir@gmail.com"
BATCH_SIZE    = 100
SLEEP_BETWEEN = 0.12   # ~8 req/s  (polite pool limit = 10/s)


# ── Collect unique author IDs ─────────────────────────────────────────────────

def collect_author_ids() -> dict[str, list[Path]]:
    """Return {openalex_full_url: [oracle_json_paths where this author appears]}."""
    id_to_files: dict[str, set[Path]] = {}
    for oracle_file in sorted(ORACLE_DIR.glob("*/gold_articles.json")):
        articles = json.loads(oracle_file.read_text(encoding="utf-8"))
        for article in articles:
            for author in article.get("authors") or []:
                oid = (author.get("openalex_id") or "").strip()
                if oid:
                    id_to_files.setdefault(oid, set()).add(oracle_file)
    return {k: list(v) for k, v in id_to_files.items()}


# ── OpenAlex batch fetch ──────────────────────────────────────────────────────

def _short_id(full_url: str) -> str:
    """'https://openalex.org/A5103216912' → 'A5103216912'."""
    return full_url.rstrip("/").rsplit("/", 1)[-1]


def fetch_hindex_batch(full_ids: list[str]) -> dict[str, int]:
    """Fetch h-index for up to BATCH_SIZE author IDs. Returns {full_url: h_index}."""
    short_ids = [_short_id(u) for u in full_ids]
    filter_str = "|".join(short_ids)
    params = urllib.parse.urlencode({
        "filter":   f"ids.openalex:{filter_str}",
        "per-page": len(full_ids),
        "select":   "id,summary_stats",
        "mailto":   OPENALEX_EMAIL,
    })
    url = f"https://api.openalex.org/authors?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": f"scientific-watch-agent (mailto:{OPENALEX_EMAIL})"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result: dict[str, int] = {}
    for author in data.get("results", []):
        h = (author.get("summary_stats") or {}).get("h_index")
        if h is not None and isinstance(h, int):
            result[author["id"]] = h
    return result


def build_cache(all_ids: list[str], existing_cache: dict[str, int]) -> dict[str, int]:
    """Fetch h-index for all IDs not already in cache."""
    missing = [i for i in all_ids if i not in existing_cache]
    if not missing:
        logger.info("Cache is complete — no API calls needed.")
        return existing_cache

    logger.info("Fetching h-index for %d authors (%d batches) ...", len(missing), -(-len(missing) // BATCH_SIZE))
    cache = dict(existing_cache)
    errors = 0
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = -(-len(missing) // BATCH_SIZE)
        try:
            result = fetch_hindex_batch(batch)
            cache.update(result)
            logger.info("  Batch %d/%d — fetched %d h-index values", batch_num, total_batches, len(result))
        except Exception as exc:
            errors += 1
            logger.warning("  Batch %d/%d — error: %s", batch_num, total_batches, exc)
        time.sleep(SLEEP_BETWEEN)

    if errors:
        logger.warning("%d batch errors — those authors will have h_index=null", errors)
    return cache


# ── Update oracle JSON files ──────────────────────────────────────────────────

def update_oracle_files(cache: dict[str, int]) -> dict[str, int]:
    """Fill h_index in every gold_articles.json. Returns stats per domain."""
    domain_stats: dict[str, int] = {}
    for oracle_file in sorted(ORACLE_DIR.glob("*/gold_articles.json")):
        domain_id = oracle_file.parent.name
        articles = json.loads(oracle_file.read_text(encoding="utf-8"))
        enriched = 0
        for article in articles:
            for author in article.get("authors") or []:
                oid = (author.get("openalex_id") or "").strip()
                if oid and oid in cache and author.get("h_index") is None:
                    author["h_index"] = cache[oid]
                    enriched += 1
        oracle_file.write_text(
            json.dumps(articles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        domain_stats[domain_id] = enriched
        logger.info("  %-40s  enriched %d author h-index values", domain_id, enriched)
    return domain_stats


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_summary(cache: dict[str, int]) -> None:
    if not cache:
        print("\n  Cache is empty.")
        return
    values = list(cache.values())
    print(f"\n  Cache size   : {len(cache)} authors")
    print(f"  h-index mean : {sum(values)/len(values):.1f}")
    print(f"  h-index max  : {max(values)}")
    print(f"  h-index = 0  : {sum(1 for v in values if v == 0)} ({100*sum(1 for v in values if v==0)/len(values):.0f}%)")
    print(f"  h-index >= 20: {sum(1 for v in values if v >= 20)} ({100*sum(1 for v in values if v>=20)/len(values):.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich oracle authors with h-index from OpenAlex")
    parser.add_argument("--dry-run",    action="store_true", help="Count only, no API calls or file writes")
    parser.add_argument("--cache-only", action="store_true", help="Build cache only, skip oracle JSON update")
    parser.add_argument("--no-update",  action="store_true", help="Alias for --cache-only")
    args = parser.parse_args()
    skip_update = args.cache_only or args.no_update

    print("\n  enrich_hindex.py — OpenAlex author h-index enrichment")
    print(f"  Oracle dir : {ORACLE_DIR}")
    print(f"  Cache      : {CACHE_PATH}\n")

    # 1. Collect author IDs
    logger.info("Scanning oracle files for author IDs ...")
    id_to_files = collect_author_ids()
    all_ids = list(id_to_files.keys())
    logger.info("Found %d unique author OpenAlex IDs across %d oracle domains",
                len(all_ids), len(list(ORACLE_DIR.glob("*/gold_articles.json"))))

    if args.dry_run:
        print(f"\n  DRY RUN — would fetch {len(all_ids)} authors in {-(-len(all_ids)//BATCH_SIZE)} batches")
        return

    # 2. Load existing cache
    existing_cache: dict[str, int] = {}
    if CACHE_PATH.exists():
        existing_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded existing cache: %d entries", len(existing_cache))

    # 3. Fetch missing
    cache = build_cache(all_ids, existing_cache)

    # 4. Save cache
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Cache saved → %s  (%d entries)", CACHE_PATH, len(cache))

    # 5. Update oracle files
    if not skip_update:
        logger.info("Updating oracle JSON files ...")
        domain_stats = update_oracle_files(cache)
        total_enriched = sum(domain_stats.values())
        print(f"\n  Total h-index values written: {total_enriched}")

    # 6. Summary
    print_summary(cache)

    if not skip_update:
        print("\n  Next steps:")
        print("    python compare_v1_v2.py --all --force-rerun")
        print("    python phase8a_8b.py")
        print("    python phase8c.py")


if __name__ == "__main__":
    main()
