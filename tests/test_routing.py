import pytest

from lunit_hackathon.routing import route_messages, self_contained_query
from lunit_hackathon.schemas import ChatMessage, MedicalDomain


@pytest.mark.parametrize(
    ("question", "domains", "tools", "verify"),
    [
        (
            "임신 중 아세트아미노펜 용량과 금기는?",
            {"drug", "vulnerable_population"},
            {"openapi_mfds_get_drug_indication", "adr_retrieve_drug_info"},
            True,
        ),
        (
            "이 항암요법은 건강보험 급여인가요?",
            {"reimbursement"},
            {"hira_updates_search"},
            True,
        ),
        (
            "KCD-9 I10 코드가 청구에 유효한가요?",
            {"coding"},
            {"kcd_get_name", "openapi_hira_disease_check_code"},
            True,
        ),
        (
            "최신 고혈압 가이드라인과 논문 근거는?",
            {"guideline", "research"},
            {"index_get_relevant_nodes", "rag_vector_query"},
            False,
        ),
    ],
)
def test_route_messages(question, domains, tools, verify):
    route = route_messages([ChatMessage(role="user", content=question)])

    assert domains <= {domain.value for domain in route.domains}
    assert tools <= set(route.tool_names)
    assert route.verification_required is verify


def test_route_messages_uses_general_health_without_specific_evidence_domain():
    route = route_messages(
        [ChatMessage(role="user", content="감기에 걸렸을 때 어떻게 쉬어야 하나요?")]
    )

    assert route.domains == frozenset({MedicalDomain.GENERAL_HEALTH})
    assert route.tool_names == ("rag_vector_query", "index_get_relevant_nodes")
    assert route.retrieval_required is True
    assert route.verification_required is False


def test_route_messages_uses_emergency_for_verification_without_emergency_tools():
    route = route_messages([ChatMessage(role="user", content="갑자기 흉통과 호흡곤란이 있어요")])

    assert MedicalDomain.EMERGENCY in route.domains
    assert "emergency" not in route.tool_names
    assert route.verification_required is True


def test_route_messages_never_falls_back_to_unrelated_tools():
    route = route_messages([ChatMessage(role="user", content="안녕하세요")])

    assert route.tool_names == ("rag_vector_query", "index_get_relevant_nodes")
    assert "openapi_law_search" not in route.tool_names
    assert "openapi_hira_disease_check_code" not in route.tool_names


def test_self_contained_query_keeps_prior_subject_before_pronoun_follow_up():
    messages = [
        ChatMessage(role="user", content="아세트아미노펜의 간독성 위험을 설명해 주세요."),
        ChatMessage(role="assistant", content="용량 초과 시 간독성 위험이 증가합니다."),
        ChatMessage(role="user", content="그 약은 임신 중에도 안전한가요?"),
    ]

    query = self_contained_query(messages)

    assert "아세트아미노펜" in query
    assert "그 약은 임신 중에도 안전한가요?" in query
    assert query.index("아세트아미노펜") < query.index("그 약은 임신 중에도 안전한가요?")


def test_self_contained_query_truncates_oldest_context_before_latest_user_message():
    messages = [
        ChatMessage(role="user", content="오래된 맥락 " + "a" * 100),
        ChatMessage(role="assistant", content="보조 설명 " + "b" * 100),
        ChatMessage(role="user", content="최신 질문은 이 약의 금기인가요?"),
    ]

    query = self_contained_query(messages, maximum_chars=80)

    assert len(query) <= 80
    assert query.endswith("최신 질문은 이 약의 금기인가요?")
    assert "a" not in query
