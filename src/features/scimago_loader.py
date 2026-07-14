"""Scimago Journal Rankings loader — ISSN → SJR Best Quartile lookup.

Provides the data backbone for the pct_q1 meta-feature in Phase 8.

The CSV (scimagojr 2025.csv) must be placed in data/ at the project root.
It is semicolon-separated with an 'Issn' column (one or more ISSNs separated
by ', ') and a 'SJR Best Quartile' column with values Q1–Q4.

The CSV is read once on first call and cached in memory for the process
lifetime (module-level dict).  Thread safety is not required here since
phase8a_8b.py is single-threaded.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SCIMAGO_PATH = _PROJECT_ROOT / "data" / "scimagojr 2025.csv"

_VALID_QUARTILES: frozenset[str] = frozenset({"Q1", "Q2", "Q3", "Q4"})

# Module-level cache populated on first call to _load()
_ISSN_TO_QUARTILE: dict[str, str] = {}
_loaded: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise_issn(issn: str) -> str:
    """Return digits-only ISSN: '1542-4863' or ' 15424863 ' → '15424863'."""
    return issn.replace("-", "").replace(" ", "").strip()


def _load(csv_path: Path = _SCIMAGO_PATH) -> None:
    """Read the Scimago CSV and populate _ISSN_TO_QUARTILE.

    Accepts an optional override path so tests can inject a small fixture CSV.
    """
    global _loaded
    if _loaded:
        return

    if not csv_path.exists():
        logger.warning(
            "[Scimago] CSV not found at %s — pct_q1 will remain 0. "
            "Place scimagojr 2025.csv in data/ to enable Q1 lookup.",
            csv_path,
        )
        _loaded = True
        return

    count = 0
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            quartile = row.get("SJR Best Quartile", "").strip()
            if quartile not in _VALID_QUARTILES:
                continue
            issn_field = row.get("Issn", "").strip().strip('"')
            if not issn_field:
                continue
            for raw in issn_field.split(","):
                norm = _normalise_issn(raw)
                if norm:
                    _ISSN_TO_QUARTILE[norm] = quartile
                    count += 1

    logger.info(
        "[Scimago] Loaded %d ISSN→quartile mappings from %s",
        count,
        csv_path.name,
    )
    _loaded = True


# ── Public API ────────────────────────────────────────────────────────────────


def get_quartile(issn: str) -> Optional[str]:
    """Return the SJR Best Quartile (Q1–Q4) for *issn*, or None if unknown.

    Accepts ISSNs with or without hyphens ('1542-4863' and '15424863' are
    treated identically).
    """
    _load()
    return _ISSN_TO_QUARTILE.get(_normalise_issn(issn))


def loaded_count() -> int:
    """Return the number of ISSN→quartile mappings currently in cache."""
    _load()
    return len(_ISSN_TO_QUARTILE)


def reset_cache() -> None:
    """Clear the in-memory cache — intended for testing only."""
    global _loaded
    _ISSN_TO_QUARTILE.clear()
    _loaded = False
