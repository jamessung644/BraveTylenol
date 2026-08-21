import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from lunit_hackathon.schemas import ChatMessage, MedicalDomain, RouteDecision

DOMAIN_KEYWORDS: Mapping[MedicalDomain, tuple[str, ...]] = MappingProxyType(
    {
        MedicalDomain.DRUG: (
            "아세트아미노펜",
            "약물",
            "의약품",
            "복용",
            "투약",
            "용량",
            "금기",
            "처방",
            "aspirin",
            "ibuprofen",
            "paracetamol",
            "drug",
            "medication",
        ),
        MedicalDomain.DRUG_SAFETY: (
            "부작용",
            "이상반응",
            "상호작용",
            "독성",
            "안전성",
            "adr",
            "adverse event",
        ),
        MedicalDomain.REIMBURSEMENT: (
            "건강보험",
            "비급여",
            "수가",
            "약가",
            "본인부담",
            "reimbursement",
            "insurance coverage",
        ),
        MedicalDomain.CODING: (
            "kcd",
            "icd",
            "코드",
            "질병분류",
            "상병",
            "청구",
        ),
        MedicalDomain.LAW: (
            "의료법",
            "법률",
            "시행령",
            "시행규칙",
            "조항",
            "규제",
            "regulation",
        ),
        MedicalDomain.GUIDELINE: (
            "가이드라인",
            "진료지침",
            "권고",
            "guideline",
            "clinical practice",
        ),
        MedicalDomain.RESEARCH: (
            "논문",
            "연구",
            "근거",
            "문헌",
            "메타분석",
            "pubmed",
            "research",
            "evidence",
        ),
        MedicalDomain.EMERGENCY: (
            "응급",
            "응급실",
            "흉통",
            "호흡곤란",
            "의식저하",
            "출혈",
            "자살",
            "경련",
            "심정지",
            "overdose",
            "shortness of breath",
            "difficulty breathing",
            "trouble breathing",
            "emergency",
        ),
        MedicalDomain.VULNERABLE_POPULATION: (
            "임신",
            "수유",
            "소아",
            "영아",
            "신생아",
            "노인",
            "고령",
            "장애",
            "면역저하",
            "pregnan",
            "pediatric",
        ),
    }
)

DOMAIN_TOOLS: Mapping[MedicalDomain, tuple[str, ...]] = MappingProxyType(
    {
        MedicalDomain.DRUG: (
            "openapi_mfds_check_drug_permission",
            "openapi_mfds_find_drugs_by_ingredient",
            "openapi_mfds_get_drug_indication",
            "adr_retrieve_drug_info",
        ),
        MedicalDomain.DRUG_SAFETY: ("adr_retrieve_drug_info", "rag_sql_query"),
        MedicalDomain.REIMBURSEMENT: ("hira_updates_search", "openapi_hira_get_drug_price"),
        MedicalDomain.CODING: (
            "kcd_search_codes",
            "kcd_get_name",
            "openapi_hira_disease_check_code",
        ),
        MedicalDomain.LAW: (
            "openapi_law_search",
            "openapi_law_list_articles",
            "openapi_law_get_article",
        ),
        MedicalDomain.GUIDELINE: (
            "index_list_documents",
            "index_get_relevant_nodes",
            "index_get_page_content",
            "index_keyword_search",
        ),
        MedicalDomain.RESEARCH: ("rag_vector_query",),
        MedicalDomain.GENERAL_HEALTH: ("rag_vector_query", "index_get_relevant_nodes"),
    }
)

_EVIDENCE_DOMAINS = frozenset(DOMAIN_TOOLS)
_VERIFICATION_DOMAINS = frozenset(
    {
        MedicalDomain.DRUG,
        MedicalDomain.DRUG_SAFETY,
        MedicalDomain.REIMBURSEMENT,
        MedicalDomain.CODING,
        MedicalDomain.LAW,
        MedicalDomain.EMERGENCY,
        MedicalDomain.VULNERABLE_POPULATION,
    }
)
_MEDICAL_CODING_PATTERN = re.compile(r"\b(?:medical|diagnosis|billing)\s+code(?:s)?\b")
_LEGAL_WORD_PATTERN = re.compile(r"\blegal\b")
_KOREAN_MEDICAL_BENEFIT_PATTERN = re.compile(
    r"(?:건강|의료|약|치료|보험)\s*급여|급여\s*(?:대상|기준|여부|인정)"
)


def route_messages(messages: Sequence[ChatMessage]) -> RouteDecision:
    """Deterministically select evidence domains and their allowed MCP tools."""
    query = self_contained_query(messages).casefold()
    domains = frozenset(
        domain
        for domain, keywords in DOMAIN_KEYWORDS.items()
        if _matches_domain(domain, keywords, query)
    )
    if not domains & _EVIDENCE_DOMAINS:
        domains = domains | frozenset({MedicalDomain.GENERAL_HEALTH})

    tools = _tools_for(domains)
    return RouteDecision(
        domains=domains,
        tool_names=tools,
        retrieval_required=bool(tools),
        verification_required=bool(domains & _VERIFICATION_DOMAINS),
    )


def self_contained_query(
    messages: Sequence[ChatMessage], maximum_chars: int = 4_000
) -> str:
    """Build a bounded, chronological retrieval query from recent dialogue."""
    if maximum_chars <= 0:
        return ""

    selected = [
        (message.role, message.content or "")
        for message in messages
        if message.role in {"user", "assistant"}
    ][-4:]
    if not selected:
        return ""

    latest_user_index = next(
        (index for index in range(len(selected) - 1, -1, -1) if selected[index][0] == "user"),
        None,
    )
    latest_user = selected[latest_user_index] if latest_user_index is not None else None
    entries = [f"{role}: {content}" for role, content in selected]

    while len("\n".join(entries)) > maximum_chars and len(entries) > 1:
        index = next(
            (index for index in range(len(entries)) if index != latest_user_index),
            0,
        )
        entries.pop(index)
        if latest_user_index is not None and index < latest_user_index:
            latest_user_index -= 1

    query = "\n".join(entries)
    if len(query) <= maximum_chars:
        return query

    role, content = latest_user or selected[-1]
    prefix = f"{role}: "
    if maximum_chars <= len(prefix):
        return prefix[:maximum_chars]
    return prefix + content[-(maximum_chars - len(prefix)) :]


def _matches_domain(
    domain: MedicalDomain, keywords: tuple[str, ...], query: str
) -> bool:
    if any(keyword.casefold() in query for keyword in keywords):
        return True
    if domain is MedicalDomain.CODING:
        return _MEDICAL_CODING_PATTERN.search(query) is not None
    if domain is MedicalDomain.LAW:
        return _LEGAL_WORD_PATTERN.search(query) is not None
    if domain is MedicalDomain.REIMBURSEMENT:
        return _KOREAN_MEDICAL_BENEFIT_PATTERN.search(query) is not None
    return False


def _tools_for(domains: frozenset[MedicalDomain]) -> tuple[str, ...]:
    tools: list[str] = []
    for domain in MedicalDomain:
        for tool_name in DOMAIN_TOOLS.get(domain, ()):
            if domain in domains and tool_name not in tools:
                tools.append(tool_name)
    return tuple(tools)
