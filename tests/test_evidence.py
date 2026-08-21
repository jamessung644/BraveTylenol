import pytest
from pydantic import ValidationError

from lunit_hackathon.evidence import (
    authority_rank,
    extract_evidence_metadata,
    rank_deduplicate_and_bound,
    valid_cite_uids,
)
from lunit_hackathon.schemas import EvidenceItem


def evidence(
    cite_uid: str,
    source_tool: str,
    relevance: float,
    content: str,
    *,
    rank: int | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        cite_uid=cite_uid,
        source_tool=source_tool,
        relevance_score=relevance,
        authority_rank=authority_rank(source_tool) if rank is None else rank,
        content=content,
    )


def test_korean_official_source_outranks_pubmed_at_equal_relevance():
    paper = evidence("pmid:1", "rag_vector_query", 0.8, "논문")
    official = evidence("mfds:1", "openapi_mfds_get_drug_indication", 0.8, "허가사항")

    result = rank_deduplicate_and_bound([paper, official])

    assert [item.cite_uid for item in result] == ["mfds:1", "pmid:1"]


def test_duplicate_content_keeps_higher_authority_item_despite_lower_relevance():
    low = evidence("pmid:1", "rag_vector_query", 0.9, "동일한 핵심 근거")
    high = evidence("hira:1", "hira_updates_search", 0.7, "동일한 핵심 근거")

    result = rank_deduplicate_and_bound([low, high])

    assert [item.cite_uid for item in result] == ["hira:1"]


def test_bound_never_splits_an_item_or_exceeds_limit():
    first = evidence("a", "index_get_page_content", 1.0, "가" * 700)
    second = evidence("b", "rag_vector_query", 0.9, "나" * 700)

    result = rank_deduplicate_and_bound([first, second], maximum_chars=1_000)

    assert sum(len(item.content) for item in result) <= 1_000
    assert [item.cite_uid for item in result] == ["a"]
    assert result[0].content == first.content


@pytest.mark.parametrize("maximum_chars", [0, -1])
def test_nonpositive_bound_returns_no_items(maximum_chars: int):
    result = rank_deduplicate_and_bound(
        [evidence("a", "index_get_page_content", 1.0, "근거")], maximum_chars
    )

    assert result == []


def test_metadata_extraction_keeps_jurisdiction_date_and_source_link():
    metadata = extract_evidence_metadata(
        '{"title":"허가사항","jurisdiction":"KR","effective_date":"2026-01-01",'
        '"source_link":"https://example.test/item"}'
    )

    assert metadata == {
        "title": "허가사항",
        "url": "https://example.test/item",
        "jurisdiction": "KR",
        "effective_date": "2026-01-01",
    }


def test_metadata_extraction_recurses_deterministically_and_prefers_primary_fields():
    metadata = extract_evidence_metadata(
        '{"items":[{"source_link":"https://fallback.test","date":"2025-01-01"},'
        '{"title":"첫 제목","jurisdiction":"KR","publication_date":"2025-02-01"},'
        '{"title":"나중 제목","url":"https://primary.test",'
        '"effective_date":"2025-03-01"}]}'
    )

    assert metadata == {
        "title": "첫 제목",
        "url": "https://primary.test",
        "jurisdiction": "KR",
        "effective_date": "2025-03-01",
    }


@pytest.mark.parametrize("content", ["not json", "{broken", "[]"])
def test_metadata_extraction_returns_empty_shape_for_missing_or_malformed_json(content: str):
    assert extract_evidence_metadata(content) == {
        "title": None,
        "url": None,
        "jurisdiction": None,
        "effective_date": None,
    }


@pytest.mark.parametrize(
    ("source_tool", "expected_rank"),
    [
        ("openapi_mfds_get_drug_indication", 5),
        ("hira_updates_search", 5),
        ("openapi_law_get_article", 5),
        ("kcd_get_name", 5),
        ("index_get_page_content", 4),
        ("dailymed_get_label", 4),
        ("rag_vector_query", 3),
        ("hira_faq_search", 2),
        ("rag_sql_query", 1),
        ("unrecognized_tool", 0),
    ],
)
def test_authority_rank_covers_every_tier(source_tool: str, expected_rank: int):
    assert authority_rank(source_tool) == expected_rank


def test_specific_authority_tiers_beat_generic_transport_matches():
    assert authority_rank("hira_faq_search") == 2
    assert authority_rank("faers_rag_vector_query") == 1


def test_stable_ties_preserve_original_order():
    first = evidence("first", "index_get_page_content", 0.7, "첫 근거")
    second = evidence("second", "index_get_page_content", 0.7, "둘째 근거")

    assert [item.cite_uid for item in rank_deduplicate_and_bound([first, second])] == [
        "first",
        "second",
    ]


def test_similar_but_not_exact_claims_are_not_deduplicated():
    first = evidence("first", "index_get_page_content", 0.7, "복용량을 의사와 확인하세요.")
    second = evidence("second", "index_get_page_content", 0.7, "복용량을 의사와 반드시 확인하세요.")

    assert [item.cite_uid for item in rank_deduplicate_and_bound([first, second])] == [
        "first",
        "second",
    ]


def test_items_with_same_normalized_prefix_but_different_suffixes_are_not_deduplicated():
    prefix = "근거" * 600
    first = evidence("first", "index_get_page_content", 0.7, prefix + " 첫 결론")
    second = evidence("second", "index_get_page_content", 0.7, prefix + " 다른 결론")

    assert [item.cite_uid for item in rank_deduplicate_and_bound([first, second])] == [
        "first",
        "second",
    ]


def test_valid_cite_uids_filters_blank_values_and_is_immutable():
    items = [
        evidence("mfds:1", "openapi_mfds_get_drug_indication", 1.0, "a"),
        evidence("   ", "index_get_page_content", 1.0, "b"),
        evidence(" mfds:1 ", "index_get_page_content", 1.0, "c"),
    ]

    result = valid_cite_uids(items)

    assert result == frozenset({"mfds:1", " mfds:1 "})
    assert isinstance(result, frozenset)


def test_evidence_item_extends_the_existing_contract_with_validated_metadata_fields():
    item = EvidenceItem(
        cite_uid="mfds:1",
        source_tool="openapi_mfds_get_drug_indication",
        relevance_score=0.5,
        authority_rank=5,
        content="근거",
        title="허가사항",
        url="https://example.test",
        jurisdiction="KR",
        effective_date="2026-01-01",
    )

    assert item.title == "허가사항"
    assert item.authority_rank == 5
    with pytest.raises(ValidationError):
        EvidenceItem(
            cite_uid="invalid",
            source_tool="unknown",
            relevance_score=0.5,
            authority_rank=6,
            content="근거",
        )
