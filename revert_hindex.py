"""Reset h_index to None in all oracle gold_articles.json files.

Reverts the changes made by enrich_hindex.py so the pipeline runs
exactly as it did before h-index enrichment.

Usage
-----
    python revert_hindex.py
    python revert_hindex.py --dry-run   # preview counts, no writes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ORACLE_DIR   = PROJECT_ROOT / "data" / "oracle"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_reset = 0
    for oracle_file in sorted(ORACLE_DIR.glob("*/gold_articles.json")):
        articles = json.loads(oracle_file.read_text(encoding="utf-8"))
        reset = 0
        for article in articles:
            for author in article.get("authors") or []:
                if author.get("h_index") is not None:
                    author["h_index"] = None
                    reset += 1
        total_reset += reset
        if args.dry_run:
            print(f"  [dry-run] {oracle_file.parent.name:40s}  would reset {reset} h_index values")
        else:
            oracle_file.write_text(
                json.dumps(articles, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  {oracle_file.parent.name:40s}  reset {reset} h_index values")

    print(f"\n  Total: {total_reset} h_index values {'would be ' if args.dry_run else ''}set to null")
    if not args.dry_run:
        print("\n  Next steps:")
        print("    python compare_v1_v2.py --all --force-rerun")
        print("    python phase8c.py")


if __name__ == "__main__":
    main()
