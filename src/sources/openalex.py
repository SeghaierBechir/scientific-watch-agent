"""OpenAlex API wrapper.

OpenAlex (https://openalex.org) is a free, open index of ~250M scholarly works.
It exposes a REST API with rich metadata: citations, h-index of authors,
journal rankings, concepts/topics. No API key required, but providing an email
in the User-Agent (the "polite pool") gives better rate limits.

Docs: https://docs.openalex.org/

This wrapper:
    - Searches works by topic (full-text search on title/abstract)
    - Maps the raw OpenAlex JSON to our internal `Article` schema
    - Handles retries with exponential backoff
    - Is tolerant of missing fields (V1 lenient mode)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    OPENALEX_API_KEY,
    OPENALEX_BASE_URL,
    OPENALEX_DEFAULT_PER_PAGE,
    OPENALEX_EMAIL,
    OPENALEX_MAX_RETRIES,
    OPENALEX_TIMEOUT,
)
from src.schemas import Article, Author, SourceType

logger = logging.getLogger(__name__)


# ============================================================
# Custom exceptions
# ============================================================


class OpenAlexError(Exception):
    """Base exception for OpenAlex client errors."""


class OpenAlexRateLimitError(OpenAlexError):
    """Raised when we hit OpenAlex rate limits (HTTP 429)."""


# ============================================================
# Main client class
# ============================================================


class OpenAlexClient:
    """Thin client around the OpenAlex /works endpoint.

    Example:
        >>> client = OpenAlexClient()
        >>> articles = client.search("fake news detection", n_results=20)
        >>> print(f"Got {len(articles)} articles")
    """

    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = OPENALEX_TIMEOUT,
    ):
        self.email = email or OPENALEX_EMAIL
        self.api_key = api_key or OPENALEX_API_KEY
        self.timeout = timeout

        # Build a session for connection reuse
        self.session = requests.Session()
        # Polite pool: include email in User-Agent for better rate limits
        ua = "scientific-watch-agent/0.1"
        if self.email:
            ua += f" (mailto:{self.email})"
        self.session.headers.update({"User-Agent": ua})

    # -----------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 25,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        require_abstract: bool = True,
    ) -> list[Article]:
        """Search works on OpenAlex by full-text query.

        Args:
            query: search terms (e.g. "fake news detection").
            n_results: how many articles to return (max 200 in this V1).
            from_year: optional lower bound on publication year.
            to_year: optional upper bound on publication year.
            require_abstract: if True, filter out articles without an abstract
                (these are useless for downstream summarization).

        Returns:
            List of validated Article objects. May be shorter than n_results
            if filtering removes some.
        """
        # Build the filter string. OpenAlex syntax: filter=key:val,key2:val2
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if require_abstract:
            filters.append("has_abstract:true")

        params: dict[str, Any] = {
            "search": query,
            "per_page": min(n_results, 100),  # API max is 100 per page
            "sort": "relevance_score:desc",
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{OPENALEX_BASE_URL}/works"
        logger.info(f"OpenAlex search: query='{query}', n={n_results}")

        # Pagination if user wants more than 100
        all_results: list[dict] = []
        page = 1
        while len(all_results) < n_results:
            params["page"] = page
            data = self._request(url, params)
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            if len(results) < params["per_page"]:
                break  # last page
            page += 1

        all_results = all_results[:n_results]
        logger.info(f"Retrieved {len(all_results)} raw results from OpenAlex")

        # Map to our schema (lenient: skip articles that fail validation)
        articles = []
        for raw in all_results:
            try:
                article = self._map_to_article(raw)
                if require_abstract and not article.abstract:
                    continue  # belt-and-suspenders
                articles.append(article)
            except Exception as e:
                logger.warning(f"Skipping malformed article: {e}")
                continue

        logger.info(f"Validated {len(articles)} articles after mapping")
        return articles

    # -----------------------------------------------------------
    # Internal: HTTP layer
    # -----------------------------------------------------------

    @retry(
        stop=stop_after_attempt(OPENALEX_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, OpenAlexRateLimitError)
        ),
        reraise=True,
    )
    def _request(self, url: str, params: dict) -> dict:
        """Make a GET request with retries on transient errors."""
        resp = self.session.get(url, params=params, timeout=self.timeout)

        if resp.status_code == 429:
            logger.warning("OpenAlex rate limit hit, retrying with backoff")
            raise OpenAlexRateLimitError("HTTP 429 from OpenAlex")
        if resp.status_code >= 500:
            logger.warning(f"OpenAlex server error {resp.status_code}, retrying")
            raise OpenAlexError(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            # Client error - don't retry
            raise OpenAlexError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    # -----------------------------------------------------------
    # Internal: data mapping
    # -----------------------------------------------------------

    def _map_to_article(self, raw: dict) -> Article:
        """Convert one OpenAlex work JSON into our Article schema.

        OpenAlex has many quirks:
            - abstracts are stored as "abstract_inverted_index" (token positions)
            - DOIs include the full URL prefix
            - Some fields can be missing or null
        """
        openalex_id = raw["id"].split("/")[-1]  # "https://openalex.org/W123" -> "W123"
        internal_id = f"openalex_{openalex_id}"

        # === Abstract: rebuild from inverted index ===
        abstract = self._reconstruct_abstract(raw.get("abstract_inverted_index"))

        # === Authors ===
        authors = []
        for authorship in raw.get("authorships", []):
            author_data = authorship.get("author") or {}
            institutions = authorship.get("institutions", [])
            affiliation = institutions[0]["display_name"] if institutions else None
            authors.append(
                Author(
                    name=author_data.get("display_name", "Unknown"),
                    orcid=author_data.get("orcid"),
                    openalex_id=author_data.get("id"),
                    affiliation=affiliation,
                    # h_index / citation_count would need a separate /authors call;
                    # we leave them None for now, fetched lazily later if needed
                )
            )

        # === Journal/venue ===
        primary_location = raw.get("primary_location") or {}
        source = primary_location.get("source") or {}
        journal_name = source.get("display_name")
        journal_issn = source.get("issn_l")  # ISSN-L is the canonical one
        is_preprint = source.get("type") == "repository"

        # === DOI: strip the URL prefix ===
        doi_raw = raw.get("doi")
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

        # === Concepts (topic labels) ===
        concepts = [c["display_name"] for c in raw.get("concepts", [])[:5]]

        # === URL ===
        url = primary_location.get("landing_page_url") or raw.get("id", "")

        return Article(
            id=internal_id,
            doi=doi,
            title=raw.get("title", "Untitled"),
            abstract=abstract,
            authors=authors,
            year=raw.get("publication_year") or 0,
            source=SourceType.OPENALEX,
            journal_name=journal_name,
            journal_issn=journal_issn,
            is_preprint=is_preprint,
            citation_count=raw.get("cited_by_count") or 0,
            url=url,
            open_access=(raw.get("open_access") or {}).get("is_oa", False),
            concepts=concepts,
        )

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
        """OpenAlex stores abstracts as inverted indexes: {word: [positions]}.

        Example input: {"Hello": [0], "world": [1]}
        Example output: "Hello world"
        """
        if not inverted_index:
            return None
        # Build a list of (position, word) then sort
        positions: list[tuple[int, str]] = []
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions.append((pos, word))
        positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positions)
