"""Tests for src/features/scimago_loader.py."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.features.scimago_loader import (
    _normalise_issn,
    get_quartile,
    loaded_count,
    reset_cache,
    _load,
    _ISSN_TO_QUARTILE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset the module-level cache before and after every test."""
    reset_cache()
    yield
    reset_cache()


def _write_fixture_csv(path: Path) -> None:
    """Write a minimal Scimago-format CSV to *path*."""
    rows = [
        # header
        ["Rank", "Sourceid", "Title", "Type", "Issn", "Publisher",
         "Open Access", "Open Access Diamond", "SJR", "SJR Best Quartile",
         "H index"],
        # Q1 — single ISSN, no hyphens
        ["1", "111", "Top Journal", "journal", "15424863", "Pub A",
         "No", "No", "10.0", "Q1", "100"],
        # Q2 — two ISSNs
        ["2", "222", "Good Journal", "journal", "10001000, 20002000", "Pub B",
         "No", "No", "5.0", "Q2", "50"],
        # Q3 — ISSN with hyphens in input (edge case: Scimago doesn't use them,
        #       but the normaliser must handle them from OpenAlex)
        ["3", "333", "Ok Journal", "journal", "3000-3000", "Pub C",
         "No", "No", "2.0", "Q3", "20"],
        # row with no quartile (should be ignored)
        ["4", "444", "Unknown Journal", "journal", "40004000", "Pub D",
         "No", "No", "", "", "10"],
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerows(rows)


# ── Tests: _normalise_issn ────────────────────────────────────────────────────


class TestNormaliseIssn:
    def test_removes_hyphens(self):
        assert _normalise_issn("1542-4863") == "15424863"

    def test_removes_spaces(self):
        assert _normalise_issn(" 15424863 ") == "15424863"

    def test_already_clean(self):
        assert _normalise_issn("15424863") == "15424863"

    def test_empty_string(self):
        assert _normalise_issn("") == ""


# ── Tests: get_quartile with fixture CSV ─────────────────────────────────────


class TestGetQuartileFixture:
    @pytest.fixture(autouse=True)
    def _inject_csv(self, tmp_path):
        """Write fixture CSV and redirect _load to it."""
        csv_path = tmp_path / "scimago_test.csv"
        _write_fixture_csv(csv_path)
        _load(csv_path)   # populate cache from fixture

    def test_q1_exact_issn(self):
        assert get_quartile("15424863") == "Q1"

    def test_q1_issn_with_hyphen(self):
        assert get_quartile("1542-4863") == "Q1"

    def test_q2_first_issn(self):
        assert get_quartile("10001000") == "Q2"

    def test_q2_second_issn(self):
        assert get_quartile("20002000") == "Q2"

    def test_q3_hyphenated_in_csv(self):
        assert get_quartile("30003000") == "Q3"

    def test_unknown_issn_returns_none(self):
        assert get_quartile("99999999") is None

    def test_row_without_quartile_not_loaded(self):
        assert get_quartile("40004000") is None

    def test_loaded_count_positive(self):
        assert loaded_count() > 0


# ── Tests: missing CSV ────────────────────────────────────────────────────────


class TestMissingCsv:
    def test_returns_none_gracefully(self, tmp_path):
        missing = tmp_path / "nonexistent.csv"
        _load(missing)
        assert get_quartile("15424863") is None

    def test_loaded_count_zero_when_csv_missing(self, tmp_path):
        _load(tmp_path / "nonexistent.csv")
        assert loaded_count() == 0
