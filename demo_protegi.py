"""demo_protegi.py -- ProTeGi prompt optimization demo for the Summarizer.

What this script does (step by step):
    1. Loads the PubMed dataset from the local protegi_data/ directory
       (train.csv + validation.csv, columns: "article", "abstract")
    2. Samples N examples from each split
    3. Computes BASELINE metrics (ROUGE + LLM Judge) with the current
       SYSTEM_PROMPT from src/agents/summarizer.py
    4. Runs ProTeGi for --n-iter iterations (default 3):
           for each iteration:
               a. Find worst-scoring train examples  (gradient signal)
               b. LLM generates a textual gradient   (diagnosis of failures)
               c. LLM proposes --beam-size improved prompts (candidates)
               d. Best candidate is selected on the validation set
    5. Computes FINAL metrics with the optimized prompt
    6. Prints a side-by-side comparison table
    7. Saves the optimized prompt to data/optimized_prompts/

Dataset structure (protegi_data/):
    train.csv      -- 119,924 rows
    validation.csv --   6,633 rows
    test.csv       --   6,658 rows
    Columns: "article" (full paper body, up to 28 KB), "abstract" (gold summary)

    Mapping to Summarizer:
        article_text = article[:2000]   -- truncated body as Summarizer input
        reference    = abstract          -- gold text for ROUGE comparison

Usage:
    python demo_protegi.py
    python demo_protegi.py --n-train 10 --n-val 5 --n-iter 2 --no-judge
    python demo_protegi.py --dry-run        # metrics only, no optimization
    python demo_protegi.py --beam-size 2 --n-worst 3
    python demo_protegi.py --data-dir path/to/other/csvs

Cost estimate (gpt-4o-mini):
    Full run (20 train, 10 val, 3 iterations, beam=4, judge ON):
      - Summarize:  (20 train + 10 val) * (1 baseline + 3 iters * (1 train + 4 cands))
                  = ~30 * 16 = ~480 summarize calls
      - Judge:       same 480 calls
      - Gradient:    3 calls
      - Candidates:  3 calls
      Total: ~966 calls * avg 300 tokens = ~290K tokens
      Input  cost: 290K * 0.15/1M = ~$0.04
      Output cost: 290K * 0.60/1M = ~$0.17
      Total: ~$0.21

    Judge OFF (--no-judge): ~$0.04
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Increase CSV field size limit (PubMed articles can be 28 KB per cell) ────
csv.field_size_limit(10_000_000)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_protegi")

# Default location of the dataset files
DEFAULT_DATA_DIR = PROJECT_ROOT / "protegi_data"


# ============================================================
# CLI arguments
# ============================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ProTeGi prompt optimization demo -- Summarizer agent"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing train.csv + validation.csv (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--n-train", type=int, default=20,
        help="Number of training examples sampled from train.csv (default: 20)",
    )
    p.add_argument(
        "--n-val", type=int, default=10,
        help="Number of validation examples sampled from validation.csv (default: 10)",
    )
    p.add_argument(
        "--n-iter", type=int, default=3,
        help="Number of ProTeGi iterations (default: 3)",
    )
    p.add_argument(
        "--beam-size", type=int, default=4,
        help="Candidate prompts generated per iteration (default: 4)",
    )
    p.add_argument(
        "--n-worst", type=int, default=5,
        help="Worst examples used as gradient signal per iteration (default: 5)",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="Disable LLM Judge (ROUGE only, ~5x cheaper)",
    )
    p.add_argument(
        "--groq",
        nargs="?",
        const="llama-3.1-8b-instant",
        metavar="MODEL",
        help=(
            "Use Groq instead of TASK_MODELS for all LLM calls. "
            "Optionally specify a model (default: llama-3.1-8b-instant). "
            "Examples: --groq  OR  --groq llama-3.3-70b-versatile"
        ),
    )
    # ── Nebius AI Studio flag ─────────────────────────────────────────────
    p.add_argument(
        "--nebius",
        action="store_true",
        help=(
            "Use Nebius AI Studio for all LLM calls. "
            "Default combination: DeepSeek-V3 (summarize + critic) and "
            "DeepSeek-R1 (judge). Override individual roles with "
            "--nebius-sum, --nebius-grad, --nebius-judge."
        ),
    )
    p.add_argument(
        "--nebius-sum",
        type=str,
        default="deepseek-ai/DeepSeek-V3",
        metavar="MODEL",
        help="Nebius model for the Summarizer (default: deepseek-ai/DeepSeek-V3)",
    )
    p.add_argument(
        "--nebius-grad",
        type=str,
        default="deepseek-ai/DeepSeek-V3",
        metavar="MODEL",
        help="Nebius model for the Critic/Gradient (default: deepseek-ai/DeepSeek-V3)",
    )
    p.add_argument(
        "--nebius-judge",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        metavar="MODEL",
        help="Nebius model for the Judge (default: deepseek-ai/DeepSeek-R1)",
    )
    p.add_argument(
        "--no-rouge", action="store_true",
        help=(
            "Disable ROUGE metrics — use LLM Judge only as the optimization signal. "
            "Recommended when the reference text (e.g. PubMed abstract) has a different "
            "format than the generated summary (structured 6-field output). "
            "Requires --no-judge to be OFF (Judge must be active to keep a signal)."
        ),
    )
    p.add_argument(
        "--narrative", action="store_true",
        help=(
            "Use narrative summary mode: the Summarizer produces a prose paragraph "
            "(150-250 words) instead of a 6-field structured JSON. "
            "ROUGE becomes meaningful again because the output format matches "
            "PubMed gold abstracts. Combine with --no-rouge=False (default) to "
            "use both ROUGE and Judge as optimization signal."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute baseline metrics only, skip ProTeGi optimization",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for example sampling (default: 42)",
    )
    p.add_argument(
        "--article-max-chars", type=int, default=2000,
        help="Max chars of article text used as Summarizer input (default: 2000)",
    )
    p.add_argument(
        "--run-name", type=str, default=None,
        metavar="NAME",
        help=(
            "Short label for this experiment run, saved in the results log. "
            "Example: --run-name llama8b-haiku-sonnet. "
            "If omitted, a name is auto-generated from the model combination."
        ),
    )
    p.add_argument(
        "--experiments-file",
        type=Path,
        default=None,
        help=(
            "Path to the JSON Lines file that accumulates experiment results. "
            "Default: data/protegi_experiments.jsonl"
        ),
    )
    return p.parse_args()


# ============================================================
# Dataset loading
# ============================================================


def _load_csv_rows(data_dir: Path, filename: str) -> list[dict]:
    """Load all rows from a CSV file in data_dir.

    Uses a large field_size_limit because PubMed article cells can be 28+ KB.
    """
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Make sure '{filename}' is in the directory: {data_dir}"
        )
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def _prepare_examples(
    rows: list[dict],
    n: int,
    seed: int,
    article_max_chars: int,
    prefix: str,
) -> list:
    """Convert N sampled CSV rows to Example objects.

    Filters out rows with empty article/abstract BEFORE sampling so that
    the returned list always contains exactly min(n, available) valid examples.
    Empty rows in the raw CSV are logged at DEBUG level (not WARNING) since
    this is expected in large PubMed exports.

    Args:
        rows            : list of dicts from csv.DictReader
        n               : how many valid examples to return
        seed            : random seed (different for train vs val)
        article_max_chars: truncate article text to this length
        prefix          : string prefix for article_id

    Returns:
        list[Example]
    """
    from src.optimization.protegi_optimizer import Example

    # Pre-filter: keep only rows that have both fields non-empty.
    # This avoids the "Skipping empty row" noise and guarantees we get
    # exactly N examples (or all available valid ones if fewer exist).
    valid_rows = [
        row for row in rows
        if (row.get("article") or "").strip() and (row.get("abstract") or "").strip()
    ]
    n_dropped = len(rows) - len(valid_rows)
    if n_dropped:
        logger.debug(
            "%s: dropped %d/%d rows with empty article or abstract",
            prefix, n_dropped, len(rows),
        )

    rng = random.Random(seed)
    sample = rng.sample(valid_rows, min(n, len(valid_rows)))

    examples: list[Example] = []
    for i, row in enumerate(sample):
        article_text = (row.get("article") or "").strip()[:article_max_chars]
        reference = (row.get("abstract") or "").strip()
        examples.append(
            Example(
                article_text=article_text,
                reference=reference,
                article_id=f"{prefix}_{i}",
            )
        )

    return examples


# ============================================================
# Display helpers
# ============================================================


def _fmt(v: float) -> str:
    """Format a [0,1] float as a percentage string with 1 decimal."""
    return f"{v * 100:5.1f}%"


def _print_divider(char: str = "-", width: int = 62) -> None:
    print(char * width)


def _print_metrics_table(
    label: str,
    results: list,
    show_rouge: bool = True,
    show_judge: bool = True,
) -> dict[str, float]:
    """Print metric averages and return them as a dict.

    Args:
        label      : section header (e.g. "BASELINE (val set)")
        results    : list of EvalResult objects
        show_rouge : False when --no-rouge is active
        show_judge : False when --no-judge is active (hides faithfulness/coverage)
    """
    if not results:
        print(f"  No results for {label}")
        return {}

    n = len(results)
    avgs = {
        "rouge1":       sum(r.rouge1 for r in results) / n,
        "rouge2":       sum(r.rouge2 for r in results) / n,
        "rougeL":       sum(r.rougeL for r in results) / n,
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "coverage":     sum(r.coverage for r in results) / n,
        "composite":    sum(r.composite for r in results) / n,
    }

    print(f"\n  {label} ({n} examples):")

    # ROUGE block
    if show_rouge:
        print(f"    ROUGE-1       : {_fmt(avgs['rouge1'])}")
        print(f"    ROUGE-2       : {_fmt(avgs['rouge2'])}")
        print(f"    ROUGE-L       : {_fmt(avgs['rougeL'])}")
    else:
        print(f"    ROUGE         : disabled (--no-rouge)")

    # Judge block — only shown when Judge is active (otherwise values are
    # placeholder 0.5 neutrals and would mislead the reader)
    if show_judge:
        print(f"    Faithfulness  : {_fmt(avgs['faithfulness'])}")
        print(f"    Coverage      : {_fmt(avgs['coverage'])}")
    else:
        print(f"    Faithfulness  : n/a (--no-judge)")
        print(f"    Coverage      : n/a (--no-judge)")

    print(f"    COMPOSITE (*) : {_fmt(avgs['composite'])}")

    # Show the formula actually used
    if show_rouge and show_judge:
        print(f"    (*) 0.4*ROUGE-L + 0.4*faithfulness + 0.2*coverage")
    elif show_judge and not show_rouge:
        print(f"    (*) 0.6*faithfulness + 0.4*coverage  [Judge-only mode]")
    else:
        print(f"    (*) mean(ROUGE-1, ROUGE-2, ROUGE-L)  [ROUGE-only mode]")

    return avgs


# ============================================================
# Experiment logging (accumulate results across runs)
# ============================================================


def _save_experiment(
    experiments_file: Path,
    run_name: str,
    models: dict,
    params: dict,
    baseline_avgs: dict,
    final_avgs: dict,
    result,          # ProTeGiResult
) -> None:
    """Append one experiment result to a JSON Lines file.

    Each line is a self-contained JSON record — easy to append, easy to read
    back for comparison.  The file is created if it doesn't exist.

    Args:
        experiments_file : path to the .jsonl accumulator file
        run_name         : short label chosen by the user (--run-name)
        models           : {"summarize": "...", "critic": "...", "judge": "..."}
        params           : hyperparameters (n_train, n_val, n_iter, …)
        baseline_avgs    : dict of metric → float from _print_metrics_table
        final_avgs       : idem for the optimized prompt
        result           : ProTeGiResult object
    """
    record = {
        "run_name":          run_name,
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "models":            models,
        "params":            params,
        "baseline": {
            "composite":    round(baseline_avgs.get("composite",    0.0), 4),
            "rouge1":       round(baseline_avgs.get("rouge1",       0.0), 4),
            "rouge2":       round(baseline_avgs.get("rouge2",       0.0), 4),
            "rougeL":       round(baseline_avgs.get("rougeL",       0.0), 4),
            "faithfulness": round(baseline_avgs.get("faithfulness", 0.0), 4),
            "coverage":     round(baseline_avgs.get("coverage",     0.0), 4),
        },
        "final": {
            "composite":    round(final_avgs.get("composite",    0.0), 4),
            "rouge1":       round(final_avgs.get("rouge1",       0.0), 4),
            "rouge2":       round(final_avgs.get("rouge2",       0.0), 4),
            "rougeL":       round(final_avgs.get("rougeL",       0.0), 4),
            "faithfulness": round(final_avgs.get("faithfulness", 0.0), 4),
            "coverage":     round(final_avgs.get("coverage",     0.0), 4),
        },
        "improvement_pct":   round(result.improvement_pct, 2),
        "duration_seconds":  round(result.duration_seconds, 1),
        "prompt_saved_to":   result.prompt_saved_to,
    }

    experiments_file.parent.mkdir(parents=True, exist_ok=True)
    with open(experiments_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Experiment saved → %s", experiments_file)


def _print_experiments_table(experiments_file: Path) -> None:
    """Read all past experiments and print a ranked comparison table."""
    if not experiments_file.exists():
        return

    records: list[dict] = []
    with open(experiments_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return

    # Sort by final composite descending
    records.sort(key=lambda r: r["final"]["composite"], reverse=True)

    print()
    _print_divider("=")
    print("  ALL EXPERIMENTS (ranked by final composite)")
    _print_divider("=")
    print(
        f"  {'#':<3} {'Run name':<28} {'Summarize':<22} "
        f"{'Critic':<22} {'Baseline':>9} {'Final':>7} {'Impr':>7}"
    )
    _print_divider()
    for rank, rec in enumerate(records, 1):
        models  = rec.get("models", {})
        base_c  = rec["baseline"]["composite"]
        final_c = rec["final"]["composite"]
        impr    = rec.get("improvement_pct", 0.0)
        arrow   = "▲" if impr > 0.5 else ("▼" if impr < -0.5 else "–")
        print(
            f"  {rank:<3} {rec['run_name']:<28} "
            f"{models.get('summarize','?'):<22} "
            f"{models.get('critic','?'):<22} "
            f"{base_c*100:>8.1f}%"
            f"{final_c*100:>7.1f}%"
            f"  {arrow}{impr:>+.1f}%"
        )
    _print_divider("=")
    print(f"  Total runs logged: {len(records)}")
    print(f"  File: {experiments_file}")
    _print_divider("=")
    print()


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = _parse_args()

    print()
    _print_divider("=")
    print("  ProTeGi Prompt Optimization Demo -- Summarizer Agent")
    _print_divider("=")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Train    : {args.n_train} examples  |  Val: {args.n_val} examples")
    print(f"  Iterations: {args.n_iter}  |  Beam: {args.beam_size}  |  N-worst: {args.n_worst}")
    _provider_label = (
        "Groq / " + args.groq if args.groq
        else "Nebius AI Studio" if args.nebius
        else "TASK_MODELS (config.py)"
    )
    print(f"  Provider : {_provider_label}")
    print(f"  LLM Judge: {'OFF' if args.no_judge else 'ON'}")
    print(f"  Summary  : {'NARRATIVE (prose paragraph)' if args.narrative else 'STRUCTURED (6 fields)'}")
    print(f"  ROUGE    : {'OFF (Judge-only mode)' if args.no_rouge else 'ON'}")
    if args.no_rouge and args.no_judge:
        print("\n  [ERROR] --no-rouge requires the LLM Judge to be active.")
        print("  Remove --no-judge or remove --no-rouge.")
        sys.exit(1)
    print(f"  Mode     : {'DRY RUN (metrics only)' if args.dry_run else 'FULL OPTIMIZATION'}")
    _print_divider()
    print()

    # ── Step 1: Load dataset ──────────────────────────────────────────────
    print("[1/5] Loading PubMed dataset from local directory...")
    try:
        train_rows = _load_csv_rows(args.data_dir, "train.csv")
        val_rows   = _load_csv_rows(args.data_dir, "validation.csv")
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    # ── Step 2: Sample and build Examples ────────────────────────────────
    print("[2/5] Sampling examples...")
    train_examples = _prepare_examples(
        train_rows, args.n_train, args.seed,
        args.article_max_chars, prefix="train",
    )
    val_examples = _prepare_examples(
        val_rows, args.n_val, args.seed + 1,
        args.article_max_chars, prefix="val",
    )

    print(f"      Sampled: {len(train_examples)} train / {len(val_examples)} val")

    # Print one example so the user sees the data
    if train_examples:
        ex = train_examples[0]
        print(f"\n  Sample article (first 120 chars):")
        print(f"    {ex.article_text[:120]}...")
        print(f"  Gold abstract (first 120 chars):")
        print(f"    {ex.reference[:120]}...")
        print()

    # ── Step 3: Build LLM clients ─────────────────────────────────────────
    print("[3/5] Initializing LLM clients...")
    from src.config import TASK_MODELS
    from src.llm.factory import get_llm_for_task

    if args.groq:
        # --groq flag: use Groq for all three roles
        from src.llm.groq_client import GroqClient
        groq_model    = args.groq
        summarize_llm = GroqClient(model=groq_model)
        gradient_llm  = GroqClient(model=groq_model)
        judge_llm     = GroqClient(model=groq_model) if not args.no_judge else None
        _sum_model = _grad_model = _jud_model = f"groq/{groq_model}"

    elif args.nebius:
        # --nebius flag: use Nebius AI Studio (DeepSeek-V3/V3/R1 by default)
        from src.llm.nebius_client import NebiusClient
        summarize_llm = NebiusClient(model=args.nebius_sum)
        gradient_llm  = NebiusClient(model=args.nebius_grad)
        judge_llm     = NebiusClient(model=args.nebius_judge) if not args.no_judge else None
        _sum_model  = f"nebius/{args.nebius_sum}"
        _grad_model = f"nebius/{args.nebius_grad}"
        _jud_model  = f"nebius/{args.nebius_judge}"

    else:
        # Default: driven by TASK_MODELS in config.py
        summarize_llm = get_llm_for_task("summarize")
        gradient_llm  = get_llm_for_task("critic")
        judge_llm     = get_llm_for_task("judge") if not args.no_judge else None
        _sum_model  = "{}/{}".format(*TASK_MODELS["summarize"])
        _grad_model = "{}/{}".format(*TASK_MODELS["critic"])
        _jud_model  = "{}/{}".format(*TASK_MODELS["judge"])

    # Auto-generate run name if not provided
    _jud_label = _jud_model if not args.no_judge else "no-judge"
    _run_name = args.run_name or f"{_sum_model}|{_grad_model}|{_jud_label}"
    _run_name = _run_name[:60]  # cap length for display

    # Experiments log file
    _experiments_file = args.experiments_file or (
        PROJECT_ROOT / "data" / "protegi_experiments.jsonl"
    )

    print(f"      summarize : {_sum_model}")
    print(f"      gradient  : {_grad_model}")
    if not args.no_judge:
        print(f"      judge     : {_jud_model}")
    else:
        print(f"      judge     : disabled (--no-judge)")

    # ── Step 4: Baseline evaluation ───────────────────────────────────────
    print("\n[4/5] Computing BASELINE metrics (initial SYSTEM_PROMPT)...")
    from src.agents.summarizer import SYSTEM_PROMPT, SYSTEM_PROMPT_NARRATIVE
    from src.optimization.protegi_optimizer import ProTeGiOptimizer

    INITIAL_PROMPT = SYSTEM_PROMPT_NARRATIVE if args.narrative else SYSTEM_PROMPT

    optimizer = ProTeGiOptimizer(
        summarize_llm=summarize_llm,
        gradient_llm=gradient_llm,
        judge_llm=judge_llm,
        n_iterations=args.n_iter,
        beam_size=args.beam_size,
        n_worst=args.n_worst,
        use_llm_judge=not args.no_judge,
        use_rouge=not args.no_rouge,
        narrative_mode=args.narrative,
    )

    baseline_results = optimizer._evaluate_set(INITIAL_PROMPT, val_examples)
    baseline_avgs = _print_metrics_table(
        "BASELINE (initial prompt, val set)", baseline_results,
        show_rouge=not args.no_rouge,
        show_judge=not args.no_judge,
    )

    # Show a sample generated summary
    if baseline_results:
        r0 = baseline_results[0]
        print(f"\n  Sample generated summary (first 200 chars):")
        print(f"    {r0.generated[:200]}...")

    if args.dry_run:
        print("\n  [DRY RUN] Skipping ProTeGi. Remove --dry-run to run optimization.")
        print()
        return

    # ── Step 5: ProTeGi optimization ──────────────────────────────────────
    print("\n[5/5] Running ProTeGi optimization...")
    _print_divider()

    save_dir = PROJECT_ROOT / "data" / "optimized_prompts"
    result = optimizer.optimize(
        initial_prompt=INITIAL_PROMPT,
        train_examples=train_examples,
        val_examples=val_examples,
        save_dir=save_dir,
        run_name=_run_name,
    )

    # ── Final results ─────────────────────────────────────────────────────
    print()
    _print_divider("=")
    print("  OPTIMIZATION RESULTS")
    _print_divider("=")

    # Re-evaluate the best prompt on the validation set
    final_results = optimizer._evaluate_set(result.best_prompt, val_examples)
    final_avgs = _print_metrics_table(
        "FINAL (optimized prompt, val set)", final_results,
        show_rouge=not args.no_rouge,
        show_judge=not args.no_judge,
    )

    # Side-by-side comparison
    print()
    _print_divider()
    print("  BEFORE vs AFTER:")
    _print_divider()
    rouge_metrics = {"rouge1", "rouge2", "rougeL"}
    metrics_order = ["rouge1", "rouge2", "rougeL", "faithfulness", "coverage", "composite"]
    for m in metrics_order:
        if m in rouge_metrics and args.no_rouge:
            continue  # hide ROUGE rows when disabled
        before = baseline_avgs.get(m, 0.0)
        after  = final_avgs.get(m, 0.0)
        delta  = after - before
        if delta > 0.005:
            arrow = "(+)"
        elif delta < -0.005:
            arrow = "(-)"
        else:
            arrow = "(=)"
        print(
            f"  {m:<15}: {_fmt(before)}  ->  {_fmt(after)}"
            f"  {arrow} {delta*100:+.1f}pp"
        )
    _print_divider()
    print(f"  Composite improvement : {result.improvement_pct:+.1f}%")
    print(f"  Total duration        : {result.duration_seconds:.0f}s")
    print(f"  Optimized prompt saved: {result.prompt_saved_to or 'not saved'}")
    _print_divider("=")

    # Iteration history
    print("\n  Iteration history:")
    for it in result.iterations:
        tag = "[improved]" if it.get("improved") else "[no change]"
        print(
            f"    Iter {it['iteration']}: "
            f"train_composite={it['train_composite']:.4f}  "
            f"best_candidate={it['best_candidate_composite']:.4f}  {tag}"
        )

    # Prompt diff (first 300 chars)
    print()
    _print_divider()
    print("  PROMPT DIFF (first 300 chars):")
    _print_divider()
    initial_preview = INITIAL_PROMPT[:300].replace("\n", " | ")
    final_preview   = result.best_prompt[:300].replace("\n", " | ")
    print(f"  INITIAL  : {initial_preview}...")
    print()
    print(f"  OPTIMIZED: {final_preview}...")
    _print_divider()

    print()
    print("  To use the optimized prompt in production, update SYSTEM_PROMPT in:")
    print("  src/agents/summarizer.py")
    print(f"  with the content of: {result.prompt_saved_to}")
    print()

    # ── Save experiment result and show comparison table ──────────────────
    _save_experiment(
        experiments_file=_experiments_file,
        run_name=_run_name,
        models={
            "summarize": _sum_model,
            "critic":    _grad_model,
            "judge":     _jud_label,
        },
        params={
            "n_train":    len(train_examples),
            "n_val":      len(val_examples),
            "n_iter":     args.n_iter,
            "beam_size":  args.beam_size,
            "n_worst":    args.n_worst,
            "use_rouge":  not args.no_rouge,
            "use_judge":  not args.no_judge,
            "seed":       args.seed,
        },
        baseline_avgs=baseline_avgs,
        final_avgs=final_avgs,
        result=result,
    )

    _print_experiments_table(_experiments_file)


if __name__ == "__main__":
    main()
