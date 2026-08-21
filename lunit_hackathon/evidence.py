"""Authority-aware selection and metadata extraction for MCP evidence."""

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from lunit_hackathon.schemas import EvidenceItem

_METADATA_KEYS = ("title", "url", "jurisdiction", "effective_date")


def authority_rank(source_tool: str) -> int:
    """Return the authority tier for an established MCP source tool."""
    tool = source_tool.casefold().strip()

    # These specific sources must be classified before broader source families.
    if "hira" in tool and "faq" in tool:
        return 2
    if "faers" in tool or tool == "rag_sql_query":
        return 1
    if (
        "mfds" in tool
        or "hira" in tool
        or "kcd" in tool
        or tool.startswith("openapi_law_")
    ):
        return 5
    if tool.startswith("index_") or "dailymed" in tool:
        return 4
    if tool == "rag_vector_query" or "pubmed" in tool:
        return 3
    return 0


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
        "url": values["url"] or values["source_link"],
        "jurisdiction": values["jurisdiction"],
        "effective_date": (
            values["effective_date"]
            or values["publication_date"]
            or values["date"]
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
    without_punctuation = "".join(
        character
        for character in content
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split()).casefold()
