"""Tests for the OpenAlex client.

We use mocks so these tests are fast, deterministic, and don't hit the real API.
The fixture `sample_openalex_response` mimics a real OpenAlex /works response.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.schemas import Article, SourceType
from src.sources.openalex import OpenAlexClient, OpenAlexError


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_openalex_response():
    """A minimal but realistic OpenAlex /works response (1 article)."""
    return {
        "meta": {"count": 1, "page": 1},
        "results": [
            {
                "id": "https://openalex.org/W123456789",
                "doi": "https://doi.org/10.1234/test.5678",
                "title": "Detecting Fake News Using Deep Learning",
                "publication_year": 2023,
                "cited_by_count": 42,
                "abstract_inverted_index": {
                    "We": [0],
                    "propose": [1],
                    "a": [2],
                    "novel": [3],
                    "method.": [4],
                },
                "authorships": [
                    {
                        "author": {
                            "id": "https://openalex.org/A111",
                            "display_name": "Jane Smith",
                            "orcid": "0000-0001-2345-6789",
                        },
                        "institutions": [
                            {"display_name": "MIT"}
                        ],
                    }
                ],
                "primary_location": {
                    "landing_page_url": "https://example.com/paper",
                    "source": {
                        "display_name": "Journal of Misinformation Studies",
                        "issn_l": "1234-5678",
                        "type": "journal",
                    },
                },
                "open_access": {"is_oa": True},
                "concepts": [
                    {"display_name": "Fake news"},
                    {"display_name": "Deep learning"},
                ],
            }
        ],
    }


# ============================================================
# Test: abstract reconstruction
# ============================================================


class TestAbstractReconstruction:
    """The trickiest part of OpenAlex mapping: rebuilding abstracts from
    their inverted-index format."""

    def test_simple_abstract(self):
        inverted = {"Hello": [0], "world": [1]}
        result = OpenAlexClient._reconstruct_abstract(inverted)
        assert result == "Hello world"

    def test_repeated_word(self):
        inverted = {"the": [0, 2], "cat": [1], "sat": [3]}
        result = OpenAlexClient._reconstruct_abstract(inverted)
        assert result == "the cat the sat"

    def test_none_returns_none(self):
        assert OpenAlexClient._reconstruct_abstract(None) is None

    def test_empty_returns_none(self):
        assert OpenAlexClient._reconstruct_abstract({}) is None


# ============================================================
# Test: mapping OpenAlex JSON -> Article
# ============================================================


class TestArticleMapping:
    def test_basic_mapping(self, sample_openalex_response):
        client = OpenAlexClient(email="test@example.com")
        raw = sample_openalex_response["results"][0]
        article = client._map_to_article(raw)

        assert isinstance(article, Article)
        assert article.id == "openalex_W123456789"
        assert article.doi == "10.1234/test.5678"
        assert article.title == "Detecting Fake News Using Deep Learning"
        assert article.year == 2023
        assert article.citation_count == 42
        assert article.source == SourceType.OPENALEX
        assert article.is_preprint is False
        assert article.open_access is True

    def test_authors_extracted(self, sample_openalex_response):
        client = OpenAlexClient(email="test@example.com")
        article = client._map_to_article(sample_openalex_response["results"][0])

        assert len(article.authors) == 1
        assert article.authors[0].name == "Jane Smith"
        assert article.authors[0].affiliation == "MIT"

    def test_journal_metadata(self, sample_openalex_response):
        client = OpenAlexClient(email="test@example.com")
        article = client._map_to_article(sample_openalex_response["results"][0])

        assert article.journal_name == "Journal of Misinformation Studies"
        assert article.journal_issn == "1234-5678"

    def test_abstract_reconstructed(self, sample_openalex_response):
        client = OpenAlexClient(email="test@example.com")
        article = client._map_to_article(sample_openalex_response["results"][0])

        assert article.abstract == "We propose a novel method."

    def test_concepts_truncated_to_5(self):
        client = OpenAlexClient(email="test@example.com")
        raw = {
            "id": "https://openalex.org/W1",
            "title": "T",
            "publication_year": 2020,
            "concepts": [{"display_name": f"C{i}"} for i in range(10)],
        }
        article = client._map_to_article(raw)
        assert len(article.concepts) == 5

    def test_missing_optional_fields_dont_crash(self):
        """Real-world OpenAlex responses sometimes omit fields entirely."""
        client = OpenAlexClient(email="test@example.com")
        minimal_raw = {
            "id": "https://openalex.org/W1",
            "title": "T",
            "publication_year": 2020,
        }
        article = client._map_to_article(minimal_raw)
        assert article.id == "openalex_W1"
        assert article.abstract is None
        assert article.authors == []
        assert article.citation_count == 0


# ============================================================
# Test: search() with mocked HTTP
# ============================================================


class TestSearch:
    @patch.object(OpenAlexClient, "_request")
    def test_search_returns_articles(self, mock_request, sample_openalex_response):
        mock_request.return_value = sample_openalex_response

        client = OpenAlexClient(email="test@example.com")
        articles = client.search("fake news", n_results=10)

        assert len(articles) == 1
        assert articles[0].title == "Detecting Fake News Using Deep Learning"
        mock_request.assert_called_once()

    @patch.object(OpenAlexClient, "_request")
    def test_search_filters_by_year(self, mock_request, sample_openalex_response):
        mock_request.return_value = sample_openalex_response

        client = OpenAlexClient(email="test@example.com")
        client.search("fake news", n_results=10, from_year=2020, to_year=2024)

        # Check that year filters were included in the request
        call_args = mock_request.call_args
        params = call_args[0][1]
        assert "from_publication_date:2020-01-01" in params["filter"]
        assert "to_publication_date:2024-12-31" in params["filter"]

    @patch.object(OpenAlexClient, "_request")
    def test_search_handles_empty_results(self, mock_request):
        mock_request.return_value = {"meta": {"count": 0}, "results": []}

        client = OpenAlexClient(email="test@example.com")
        articles = client.search("query that finds nothing")

        assert articles == []
