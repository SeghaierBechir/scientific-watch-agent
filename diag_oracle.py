"""Quick diagnostic: inspect oracle files for a domain."""
import json
from pathlib import Path
import sys

domain = sys.argv[1] if len(sys.argv) > 1 else "federated_learning"
oracle_dir = Path("data/oracle") / domain

print(f"\n=== Oracle diagnostic: {domain} ===\n")

rel_path = oracle_dir / "gold_relevance.json"
art_path = oracle_dir / "gold_articles.json"

if not rel_path.exists():
    print("gold_relevance.json  MISSING — run build_oracle.py first")
else:
    gold = json.loads(rel_path.read_text(encoding="utf-8"))
    grade2 = sum(1 for v in gold.values() if v == 2)
    grade1 = sum(1 for v in gold.values() if v == 1)
    print(f"gold_relevance.json  {len(gold)} entries  (grade-2: {grade2}, grade-1: {grade1})")
    for k, v in list(gold.items())[:3]:
        print(f"  {k!r}: {v}")

if not art_path.exists():
    print("gold_articles.json   MISSING")
else:
    arts = json.loads(art_path.read_text(encoding="utf-8"))
    print(f"\ngold_articles.json   {len(arts)} articles")
    for a in arts[:3]:
        print(f"  id={a['id']!r}  year={a.get('year')}  cit={a.get('citation_count')}")

# Cross-check: how many article IDs appear in gold_relevance?
if rel_path.exists() and art_path.exists():
    gold = json.loads(rel_path.read_text(encoding="utf-8"))
    arts = json.loads(art_path.read_text(encoding="utf-8"))
    corpus_ids = {a["id"] for a in arts}
    matched = {k: v for k, v in gold.items() if k in corpus_ids}
    print(f"\nCross-check: {len(matched)}/{len(gold)} gold IDs found in corpus")
    if len(matched) == 0 and len(gold) > 0:
        print("  !! MISMATCH: gold IDs format vs article IDs format")
        if gold:
            print(f"  gold key sample : {list(gold.keys())[0]!r}")
        if arts:
            print(f"  article id sample: {arts[0]['id']!r}")
