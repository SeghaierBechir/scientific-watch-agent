"""AutoML scorer — learns optimal quality-scoring weights via Optuna (Phase 3).

How it works
------------
Quality scoring combines 6 sub-scores into a weighted sum:

    final = w_venue·venue + w_authors·authors + w_impact·impact
          + w_velocity·velocity + w_recency·recency + w_relevance·relevance

Level 3 adds velocity (citation momentum) and recency (publication freshness)
to the original 4 features.  Optuna searches the 6-dimensional weight simplex
to maximise NDCG@15 against a gold-standard corpus.

Study versioning
----------------
Changing the feature set would corrupt a previously stored Optuna study
(parameters would not match).  We version-stamp the study name with the
feature count (``_f6``) so that adding features auto-creates a fresh study
without deleting historical ones.

Why Optuna TPE?
    - 6 continuous parameters → TPE converges in ~200 trials (<2 ms / trial)
    - Sample-efficient (smarter than random / grid search)
    - Persistent SQLite storage: studies are resumable if interrupted

Typical call sequence
---------------------
1.  Build a gold corpus for the domain (see data/oracle/build_oracle.py)
2.  ``result = optimize_weights(articles, gold_relevance, topic)``
3.  Check ``result.improvement_pct`` — if above threshold, weights are saved.
4.  Later, ``load_weights_for_topic(topic)`` retrieves them for quality_critic.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import optuna

from src.config import (
    DEFAULT_WEIGHTS,
    OPTUNA_MIN_IMPROVEMENT,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    WEIGHTS_DIR,
)
from src.schemas import Article
from src.scoring.metrics import ndcg_at_k
from src.scoring.quality_scorer import score_articles

logger = logging.getLogger(__name__)

# Suppress Optuna's own INFO logs (they are very verbose).
# We log summary lines ourselves via our logger.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Pydantic result schema ────────────────────────────────────────────────────

from pydantic import BaseModel, Field  # noqa: E402  (after optuna import)


class OptimizationResult(BaseModel):
    """Output of one Optuna optimisation run.

    Stored as JSON next to the learned weights file so experiments are
    reproducible and comparable across domains.
    """

    topic: str
    best_weights: dict[str, float] = Field(
        ..., description="Normalised weights learned by Optuna"
    )
    best_ndcg_at_15: float = Field(..., ge=0, le=1)
    baseline_ndcg_at_15: float = Field(
        ...,
        ge=0,
        le=1,
        description="NDCG@15 achieved by DEFAULT_WEIGHTS (before optimisation)",
    )
    improvement_pct: float = Field(
        ...,
        description="Relative improvement over baseline in percent. "
        "Positive = Optuna is better.",
    )
    n_trials_completed: int = Field(..., ge=0)
    duration_seconds: float = Field(..., ge=0)
    weights_saved: bool = Field(
        ...,
        description="True when improvement >= OPTUNA_MIN_IMPROVEMENT and "
        "weights were persisted to disk.",
    )
    relevance_method: str = Field(
        default="v2",
        description="Relevance scoring method used: 'v2' (semantic embeddings) "
        "or 'v1' (keyword/bigram). Set by the use_semantic parameter.",
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _safe_topic_name(topic: str) -> str:
    """Convert a free-text topic to a filesystem-safe identifier.

    Examples
    --------
    "Fake News Detection"  → "fake_news_detection"
    "Graph Neural Networks (GNN)" → "graph_neural_networks_gnn"
    """
    lower = topic.lower()
    safe = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    return safe or "unknown_topic"


def _weights_json_path(topic: str) -> Path:
    """Path to the learned-weights JSON for a topic."""
    return WEIGHTS_DIR / f"{_safe_topic_name(topic)}_weights.json"


# Version stamp: bump when the feature set changes so new runs create a fresh
# Optuna study rather than reusing an incompatible old one.
_FEATURE_VERSION = "f6"  # 6 features: venue, authors, impact, velocity, recency, relevance


def _study_name(topic: str, use_semantic: bool = True) -> str:
    """Versioned study name — includes feature count + relevance method.

    Suffix 'em' = semantic embeddings (V2), 'kw' = keyword/bigram (V1.5).
    Different names ensure V1 and V2 Optuna runs never share trial history.
    """
    rel_tag = "em" if use_semantic else "kw"
    return f"{_safe_topic_name(topic)}_{_FEATURE_VERSION}_{rel_tag}"


def _study_db_path(topic: str) -> str:
    """SQLite URL for Optuna study persistence.

    Both V1 and V2 studies for the same topic share one DB file — SQLite
    supports multiple named studies per database.
    """
    db_file = WEIGHTS_DIR / f"{_safe_topic_name(topic)}.db"
    return f"sqlite:///{db_file}"


def _ranked_ids_from_weights(
    articles: list[Article],
    topic: str,
    weights: dict[str, float],
    use_semantic: bool = True,
) -> list[str]:
    """Score and rank articles with given weights; return IDs best-first.

    Args:
        use_semantic: if True, use V2 semantic relevance (sentence-transformers).
            If False, use V1.5 keyword/bigram scoring.  Passed directly to
            score_articles() which handles the fallback if V2 is unavailable.
    """
    scores = score_articles(articles, topic, weights, use_semantic=use_semantic)
    paired = sorted(
        zip(articles, scores),
        key=lambda p: p[1].final_score,
        reverse=True,
    )
    return [art.id for art, _ in paired]


def _compute_baseline(
    articles: list[Article],
    gold_relevance: dict[str, int],
    topic: str,
    use_semantic: bool = True,
) -> float:
    """NDCG@15 achieved by DEFAULT_WEIGHTS — used as comparison baseline."""
    ranked = _ranked_ids_from_weights(articles, topic, DEFAULT_WEIGHTS, use_semantic=use_semantic)
    return ndcg_at_k(ranked, gold_relevance, k=15)


# ── Core optimisation ─────────────────────────────────────────────────────────


def optimize_weights(
    candidate_articles: list[Article],
    gold_relevance: dict[str, int],
    topic: str,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
    force_rerun: bool = False,
    use_semantic: bool = True,
) -> OptimizationResult:
    """Run Optuna to learn optimal scoring weights for a domain.

    The objective maximises NDCG@15: we want the most relevant articles
    (according to gold_relevance) to appear at the top of the ranked list.

    Parameters
    ----------
    candidate_articles:
        Pool of articles to rank.  Ideally 50-200 articles covering the domain,
        including both relevant and non-relevant ones.
    gold_relevance:
        Ground-truth dict ``{article_id: grade}`` with grade ∈ {0, 1, 2}.
        Built from survey referenced_works (see build_oracle.py).
    topic:
        Free-text topic string (e.g. "fake news detection").  Used to name
        the persisted study and weights file.
    n_trials:
        Number of Optuna trials.  150 is enough for 4 parameters (TPE converges
        well before that).  Override for quick tests.
    timeout:
        Wall-clock seconds budget.  The study stops whichever comes first:
        n_trials exhausted OR timeout reached.  Set 0 to disable timeout.
    force_rerun:
        If True, deletes any existing SQLite study and starts fresh.
        If False (default), resumes from the saved study if it exists.
    use_semantic:
        If True (default), use V2 semantic relevance (sentence-transformers).
        If False, use V1.5 keyword/bigram scoring.
        The Optuna study name includes a suffix ('_em' or '_kw') so V1 and V2
        runs never share cached trial history.

    Returns
    -------
    OptimizationResult
        Contains best_weights, NDCG values, improvement %, relevance_method,
        and whether the weights were saved to disk.

    Notes
    -----
    Weight search space: each of the 4 dimensions is searched in [0.05, 0.60].
    The raw sampled values are NOT normalised before scoring — quality_scorer's
    ``_validate_weights`` normalises them internally, so the actual effective
    range is [0.05/sum, 0.60/sum].  This lets Optuna explore a rich set of
    relative proportions without artificial boundaries.
    """
    if not candidate_articles:
        raise ValueError("candidate_articles must not be empty")
    if not gold_relevance:
        raise ValueError("gold_relevance must not be empty")

    rel_label = "v2 (semantic)" if use_semantic else "v1.5 (keyword/bigram)"
    t_start = time.perf_counter()

    # ── Baseline (DEFAULT_WEIGHTS) ────────────────────────────────────────────
    baseline_ndcg = _compute_baseline(candidate_articles, gold_relevance, topic, use_semantic=use_semantic)
    logger.info(
        f"[AutoML] Topic='{topic}' | relevance={rel_label} | "
        f"Baseline NDCG@15={baseline_ndcg:.4f} (DEFAULT_WEIGHTS)"
    )

    # ── Optuna study ──────────────────────────────────────────────────────────
    storage_url = _study_db_path(topic)

    study_name = _study_name(topic, use_semantic=use_semantic)
    if force_rerun:
        # Delete existing study so we start clean
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
            logger.info(f"[AutoML] Deleted existing study '{study_name}'")
        except Exception:
            pass  # study didn't exist — that's fine

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,   # resume if interrupted
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: sample 6 weights → rank articles → NDCG@15."""
        raw_weights = {
            "venue":     trial.suggest_float("venue",     0.05, 0.60),
            "authors":   trial.suggest_float("authors",   0.05, 0.60),
            "impact":    trial.suggest_float("impact",    0.05, 0.60),
            "velocity":  trial.suggest_float("velocity",  0.05, 0.60),
            "recency":   trial.suggest_float("recency",   0.05, 0.60),
            "relevance": trial.suggest_float("relevance", 0.05, 0.60),
        }
        ranked = _ranked_ids_from_weights(
            candidate_articles, topic, raw_weights, use_semantic=use_semantic
        )
        return ndcg_at_k(ranked, gold_relevance, k=15)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout if timeout > 0 else None,
        show_progress_bar=False,
    )

    # ── Extract best result ───────────────────────────────────────────────────
    best_trial = study.best_trial
    best_raw_weights = best_trial.params

    # Re-normalise so weights sum to 1.0 (quality_scorer also normalises, but
    # we want the persisted JSON to be human-readable and consistent)
    total = sum(best_raw_weights.values())
    best_weights: dict[str, float] = {
        k: round(v / total, 6) for k, v in best_raw_weights.items()
    }

    best_ndcg = best_trial.value  # already normalised NDCG (scorer normalises internally)

    if baseline_ndcg > 0:
        improvement_pct = (best_ndcg - baseline_ndcg) / baseline_ndcg * 100
    elif best_ndcg > 0:
        # Baseline is exactly 0 (e.g. no gold articles in top-K with default weights).
        # Any positive NDCG from Optuna is a meaningful improvement — treat as +100%.
        improvement_pct = 100.0
        logger.info(
            "[AutoML] Baseline NDCG=0: any positive result accepted (improvement set to +100%)"
        )
    else:
        improvement_pct = 0.0

    duration = time.perf_counter() - t_start

    logger.info(
        f"[AutoML] Completed {len(study.trials)} trials in {duration:.1f}s | "
        f"Best NDCG@15={best_ndcg:.4f} | "
        f"Improvement={improvement_pct:+.1f}% over baseline"
    )
    logger.info(f"[AutoML] Best weights: {best_weights}")

    # ── Persist weights if improvement is meaningful ──────────────────────────
    threshold_met = improvement_pct >= OPTUNA_MIN_IMPROVEMENT * 100  # config is ratio
    if threshold_met:
        save_weights_for_topic(topic, best_weights)
        logger.info(
            f"[AutoML] Weights saved to {_weights_json_path(topic)} "
            f"(improvement {improvement_pct:.1f}% ≥ threshold "
            f"{OPTUNA_MIN_IMPROVEMENT * 100:.1f}%)"
        )
    else:
        logger.info(
            f"[AutoML] Weights NOT saved: improvement {improvement_pct:.1f}% < "
            f"threshold {OPTUNA_MIN_IMPROVEMENT * 100:.1f}%. "
            f"DEFAULT_WEIGHTS will be used."
        )

    return OptimizationResult(
        topic=topic,
        best_weights=best_weights,
        best_ndcg_at_15=best_ndcg,
        baseline_ndcg_at_15=baseline_ndcg,
        improvement_pct=improvement_pct,
        n_trials_completed=len(study.trials),
        duration_seconds=duration,
        weights_saved=threshold_met,
        relevance_method="v2" if use_semantic else "v1",
    )


# ── Persistence helpers ───────────────────────────────────────────────────────


def save_weights_for_topic(topic: str, weights: dict[str, float]) -> Path:
    """Persist normalised weights to JSON.

    The file is named ``{safe_topic}_weights.json`` and lives in WEIGHTS_DIR.
    Overwrites any existing file for the same topic.

    Returns the path where the file was written.
    """
    path = _weights_json_path(topic)
    payload = {
        "topic": topic,
        "weights": weights,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_weights_for_topic(topic: str) -> Optional[dict[str, float]]:
    """Load learned weights for a topic from disk.

    Returns None if no learned weights exist (quality_critic will then fall
    back to DEFAULT_WEIGHTS automatically).

    Args:
        topic: free-text topic string (same one passed to optimize_weights).

    Returns:
        dict with keys 'venue', 'authors', 'impact', 'relevance', or None.
    """
    path = _weights_json_path(topic)
    if not path.exists():
        logger.debug(
            f"[AutoML] No learned weights found for topic '{topic}' "
            f"(expected at {path}). Using DEFAULT_WEIGHTS."
        )
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        weights = data["weights"]
        # Accept both 4-key (Level 1-2) and 6-key (Level 3) weight files.
        required = {"venue", "authors", "impact", "relevance"}
        if not required.issubset(weights.keys()):
            logger.warning(
                f"[AutoML] Weights file {path} is malformed (missing required keys). "
                f"Ignoring."
            )
            return None
        logger.info(f"[AutoML] Loaded learned weights for '{topic}': {weights}")
        return weights
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            f"[AutoML] Could not parse weights file {path}: {exc}. "
            f"Using DEFAULT_WEIGHTS."
        )
        return None
