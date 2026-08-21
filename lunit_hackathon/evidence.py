"""Authority-aware selection and metadata extraction for MCP evidence."""

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from lunit_hackathon.schemas import EvidenceItem

_METADATA_KEYS = ("title", "url", "jurisdiction", "effective_date")
_AUTHORITY_RANKS = {
    "openapi_mfds_check_drug_permission": 5,
    "openapi_mfds_find_drugs_by_ingredient": 5,
    "openapi_mfds_get_drug_indication": 5,
    "hira_updates_search": 5,
    "openapi_hira_get_drug_price": 5,
    "openapi_hira_disease_check_code": 5,
    "kcd_search_codes": 5,
    "kcd_get_name": 5,
    "openapi_law_search": 5,
    "openapi_law_list_articles": 5,
    "openapi_law_get_article": 5,
    "index_list_documents": 4,
    "index_get_relevant_nodes": 4,
    "index_get_page_content": 4,
    "index_keyword_search": 4,
    "dailymed_get_label": 4,
    "rag_vector_query": 3,
    "hira_faq_search": 2,
    "rag_sql_query": 1,
}


def authority_rank(source_tool: str) -> int:
    """Return the authority tier for an established MCP source tool."""
    return _AUTHORITY_RANKS.get(source_tool.casefold().strip(), 0)


def extract_evidence_metadata(content: str) -> dict[str, str | None]:
    """Extract allowlisted JSON metadata without inferring it from text."""
    empty = {key: None for key in _METADATA_KEYS}
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return empty
    if not isinstance(parsed, (dict, list)):
        return empty

    values: dict[str, str | None] = {
        "title": None,
        "url": None,
        "source_link": None,
        "jurisdiction": None,
        "effective_date": None,
        "publication_date": None,
        "date": None,
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                if (
                    key in values
                    and values[key] is None
                    and isinstance(nested_value, str)
                ):
                    values[key] = nested_value
                visit(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                visit(nested_value)

    visit(parsed)
    return {
        "title": values["title"],
        "url": (
            values["url"]
            if values["url"] is not None
            else values["source_link"]
        ),
        "jurisdiction": values["jurisdiction"],
        "effective_date": (
            values["effective_date"]
            if values["effective_date"] is not None
            else (
                values["publication_date"]
                if values["publication_date"] is not None
                else values["date"]
            )
        ),
    }


def rank_deduplicate_and_bound(
    items: Sequence[EvidenceItem], maximum_chars: int = 24_000
) -> list[EvidenceItem]:
    """Keep highest-ranked exact evidence, bounded without splitting items."""
    if maximum_chars <= 0:
        return []

    ranked = sorted(
        enumerate(items),
        key=lambda indexed_item: (
            indexed_item[1].authority_rank,
            indexed_item[1].relevance_score,
            -indexed_item[0],
        ),
        reverse=True,
    )
    deduplicated: list[EvidenceItem] = []
    normalized_by_digest: dict[str, set[str]] = {}
    for _, item in ranked:
        normalized = _normalize_content(item.content)
        digest = hashlib.sha256(normalized[:1_000].encode()).hexdigest()
        known = normalized_by_digest.setdefault(digest, set())
        if normalized in known:
            continue
        known.add(normalized)
        deduplicated.append(item)

    selected: list[EvidenceItem] = []
    remaining = maximum_chars
    for item in deduplicated:
        item_length = len(item.content)
        if item_length <= remaining:
            selected.append(item)
            remaining -= item_length
        if remaining == 0:
            break
    return selected


def valid_cite_uids(items: Sequence[EvidenceItem]) -> frozenset[str]:
    """Return immutable, nonblank citation identifiers without rewriting them."""
    return frozenset(item.cite_uid for item in items if item.cite_uid.strip())


def _normalize_content(content: str) -> str:
    normalized: list[str] = []
    for index, character in enumerate(content):
        if not unicodedata.category(character).startswith("P"):
            normalized.append(character)
            continue

        previous_is_digit = index > 0 and content[index - 1].isdecimal()
        next_is_digit = index + 1 < len(content) and content[index + 1].isdecimal()
        normalized.append(character if previous_is_digit and next_is_digit else " ")
    return " ".join("".join(normalized).split()).casefold()
