import inspect
import json

import pytest

import lunit_hackathon.generation as generation_module
from lunit_hackathon.errors import (
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamTimeoutError,
)
from lunit_hackathon.generation import GenerationEngine, requires_retrieval
from lunit_hackathon.schemas import (
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    ToolCall,
)


def tool_call(call_id: str, name: str, arguments: dict | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, function={"name": name, "arguments": raw})


class ScriptedL2:
    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        completion = self.completions.pop(0)
        if completion.tool_calls and completion.finish_reason is None:
            return completion.model_copy(update={"finish_reason": "tool_calls"})
        return completion


class FakeRetrieval:
    def __init__(self, result=None):
        self.queries = []
        self.result = result or RetrievalResult(
            status="sufficient",
            items=[
                EvidenceItem(
                    cite_uid="guideline:1",
                    source_tool="guideline_search",
                    relevance_score=0.95,
                    content='{"cite_uid":"guideline:1","text":"근거"}',
                )
            ],
        )

    async def retrieve(self, query):
        self.queries.append(query)
        return self.result


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("두통 최신 상태가 어떤가요?", False),
        ("감기 출처가 집인지 회사인지 모르겠어요", False),
        ("머리아픔 약 뭐먹지", False),
        ("식약처 타이레놀 허가사항", True),
        ("KCD I10 공식 명칭", True),
        ("와파린 보험 급여 심평원", True),
        ("의료법 조문 찾아줘", True),
        ("고혈압 최신 진료 지침", True),
        ("타이레놀 간독성 논문 찾아", True),
        ("고혈압 가이드라인 알려줘", True),
        ("타이레놀 간독성 논문 있어?", True),
        ("도치 입력: 출처 공식 고혈압 지침", True),
        ("심평언 와파린 급여", True),
        ("근거를 찾아, 아스피린과 와파린 상호작용", True),
        ("와파린 임신 중 먹어도 돼?", True),
        ("I10 공식 명칭이 뭐야?", True),
        ("코로나 격리 기준 지금 어떻게 돼?", True),
        ("아픽사반과 이트라코나졸 상호작용", True),
        ("14kg 소아 아세트아미노펜 용량", True),
        ("타이레놀 허가 적응증", True),
        ("현재 당뇨 진단 기준", True),
        ("아스피린 출처", True),
        ("Please provide sources for aspirin use in pregnancy", True),
        ("Can I take this medication despite the listed drug interaction?", True),
        ("What does the official label contraindicate for this medicine?", True),
        ("Read the package insert before answering.", True),
        ("What is the FDA-approved indication?", True),
        ("Summarize the FDA-approved label.", True),
        ("감염병의 예방 및 관리에 관한 법률 현행 시행 조문", True),
        ("이 보건 관련 법의 현재 시행 조문을 확인해줘", False),
        ("최신가이드라인 고혈압 알려줘", True),
        ("잠을 잘 수가 없어요", False),
        ("가이드라인이란 뭐야?", False),
        ("식약처 말고 일반 원리만 설명해줘", False),
        ('"의료법"이라는 단어를 인용했어요', False),
        ("interaction이라는 영어 단어 뜻만 알려줘", False),
        ("현행 법률 말고 일반 윤리 원리만 설명해줘", False),
        ("임신 중 병원 예약 방법", False),
        ("임신 관련 정보를 요약해줘", False),
        ("고령자 운동 정보", False),
        ("소아 교육 정책", False),
    ],
)
def test_hybrid_retrieval_router_is_high_precision_for_noisy_korean(question, expected):
    assert requires_retrieval([ChatMessage(role="user", content=question)]) is expected


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [
                ChatMessage(role="user", content="식약처 타이레놀 허가사항 알려줘"),
                ChatMessage(role="assistant", content="이전 답변"),
                ChatMessage(role="user", content="고마워"),
            ],
            False,
        ),
        (
            [
                ChatMessage(role="user", content="KCD I10 공식 명칭 알려줘"),
                ChatMessage(role="assistant", content="이전 답변"),
                ChatMessage(role="user", content="오늘 배가 아파요"),
            ],
            False,
        ),
        (
            [
                ChatMessage(role="user", content="와파린 식약처 허가를 확인해줘"),
                ChatMessage(role="assistant", content="이전 답변"),
                ChatMessage(role="user", content="그럼 그 약의 임신 금기는?"),
            ],
            True,
        ),
    ],
)
def test_router_uses_prior_turn_only_for_a_real_follow_up(messages, expected):
    assert requires_retrieval(messages) is expected


@pytest.mark.parametrize(
    ("turns", "expected_route"),
    [
        (
            [
                (
                    "user",
                    "2026-08-22 현재의 가상 사례입니다. 임신하지 않았고 당뇨병·콩팥병·"
                    "심혈관질환이 없는 54세 성인의 진료실 혈압이 두 차례 142/88, "
                    "144/90 mmHg였습니다. 대한민국 현행 지침의 진료실 진단 기준과 "
                    "합병증 없는 일반 성인의 목표혈압을 혼동하지 말고, 판·발행기관·"
                    "공식 출처를 적어 주세요. 치료 변경 지시는 하지 마세요.",
                )
            ],
            "retrieval",
        ),
        (
            [
                (
                    "user",
                    "가상 사례입니다. 와파린 복용 중인 61세 성인에게 metronidazole "
                    "처방이 추가됐습니다. 현행 공식 라벨에서 확인되는 항응고 효과·"
                    "PT/INR·출혈 위험의 방향과 권장 모니터링을 설명해 주세요. 환자에게 "
                    "임의 증감량이나 중단을 지시하지 마세요.",
                )
            ],
            "retrieval",
        ),
        (
            [
                (
                    "user",
                    "가상 사례입니다. 간질환·금기가 없고 중복 성분 복용이 없는 8세, "
                    "24 kg 소아에게 120 mg/5 mL 아세트아미노펜 현탁액을 검토합니다. "
                    "공식 라벨이 10–15 mg/kg/회 기준인지 확인하고, 맞다면 mg와 mL, "
                    "투여 간격, 1일 한도를 계산 과정으로 보여 주세요. 허가사항이 다르면 "
                    "그 차이를 우선하고 직접 복용 지시는 하지 마세요.",
                )
            ],
            "retrieval",
        ),
        (
            [
                ("user", "국민건강보험법에서 수술기록과 검사기록 보존기간을 찾고 있습니다."),
                ("assistant", "확인했습니다."),
                (
                    "user",
                    "앞의 법명은 틀렸습니다. 의료법 시행규칙 기준으로 수술기록과 검사 "
                    "내용 및 검사기록을 구분해 주세요. 2026-08-22 현행 조문과 시행일을 "
                    "공식 출처로 제시해 주세요.",
                ),
            ],
            "retrieval",
        ),
        (
            [
                (
                    "user",
                    "현행 KCD에서 J06.9와 R05의 공식 명칭과 서로 바꿔 쓸 수 없는 이유를 "
                    "일반인에게 설명해 주세요. 이 짧은 질문만으로 진단·코딩을 확정하지 "
                    "말고 KCD 판·공식 출처를 적어 주세요.",
                )
            ],
            "retrieval",
        ),
        (
            [
                ("user", "This fictional patient takes lithium for maintenance treatment."),
                ("assistant", "Understood."),
                (
                    "user",
                    "그 약을 복용하는 동안 ibuprofen이 추가되면 공식 라벨상 어떤 위험과 "
                    "모니터링이 있나요? Please answer in English and do not tell the patient "
                    "to change either drug on their own.",
                ),
            ],
            "retrieval",
        ),
        (
            [
                (
                    "user",
                    "About 20 minutes ago my face became uneven and my right arm became weak, "
                    "and my words are slurred. What should I do now? I am in Korea.",
                )
            ],
            "emergency_no_tool",
        ),
        (
            [
                (
                    "user",
                    "Two years ago I was evaluated for possible stroke, but I have no weakness, "
                    "facial droop, or speech problem now.",
                ),
                ("assistant", "Understood; those symptoms are historical and absent now."),
                (
                    "user",
                    "My actual question is why the pulse at the wrist usually matches the heart "
                    "rate. Please answer briefly in English.",
                ),
            ],
            "direct",
        ),
        (
            [
                (
                    "user",
                    "임신 38주 진료지침이나 응급상황 질문이 아닙니다. 문장 “물을 마실 "
                    "수가 없다는 뜻은 아닙니다”에서 ‘수’와 조사 ‘가’의 띄어쓰기를 설명해 "
                    "주세요.",
                )
            ],
            "direct",
        ),
        (
            [
                (
                    "user",
                    "심박수와 손목에서 세는 맥박수는 무엇을 뜻하고 보통 왜 같게 측정되는지, "
                    "정상 범위나 개인 진단 없이 쉽게 설명해 주세요.",
                )
            ],
            "direct",
        ),
    ],
)
def test_frozen_v3_synthetic_route_contract(turns, expected_route):
    messages = [ChatMessage(role=role, content=content) for role, content in turns]
    route = (
        "emergency_no_tool"
        if generation_module._is_emergency_turn(messages)
        else "retrieval"
        if requires_retrieval(messages)
        else "direct"
    )

    assert route == expected_route


async def test_generation_returns_direct_l2_text_with_only_official_application_tool():
    l2 = ScriptedL2(
        [
            L2Completion(content="internal no-evidence decision"),
            L2Completion(content="L2 원문 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "L2 원문 답변"
    assert retrieval.queries == []
    assert [tool["function"]["name"] for tool in l2.calls[0]["tools"]] == [
        "retrieve_relevant_content"
    ]
    assert l2.calls[0]["tools"][0]["function"]["strict"] is True
    assert "tools" not in l2.calls[1]
    assert l2.calls[1]["max_tokens"] == 4_096
    assert "retrieve_relevant_content" not in l2.calls[1]["messages"][0]["content"]
    assert l2.calls[1]["messages"][0]["content"] != (
        l2.calls[0]["messages"][0]["content"]
    )
    final_envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert final_envelope["final_phase_context"]["phase"] == "direct"


async def test_generation_retrieves_then_resumes_same_trajectory_without_more_tools():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "한국 고혈압 목표 혈압 진료지침"},
                    )
                ]
            ),
            L2Completion(content="근거를 반영한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="공식 진료지침 목표 혈압은?")]
    )

    assert answer == "근거를 반영한 답변 [1]"
    assert retrieval.queries == ["공식 진료지침 목표 혈압은?"]
    assert l2.calls[0]["attempt_timeout_seconds"] == 25
    assert l2.calls[1]["attempt_timeout_seconds"] == 45
    assert l2.calls[1]["max_tokens"] == 4_096
    assert l2.calls[1]["allow_blank_recovery"] is False
    assert "tools" not in l2.calls[1]
    assert "tool_choice" not in l2.calls[1]
    assert len(l2.calls[1]["messages"]) == 2
    assert l2.calls[1]["messages"][0]["role"] == "system"
    assert "retrieve_relevant_content" not in l2.calls[1]["messages"][0]["content"]
    final_envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert final_envelope["final_phase_context"]["phase"] == "post_retrieval"
    evidence = final_envelope["final_phase_context"]["evidence"]
    assert evidence["schema_version"] == "retrieval-evidence-v4"
    assert evidence["selection_contract"] == "dashboard_v1"
    assert evidence["claim_mapping_status"] == "not_available"
    assert evidence["evidence_status"] == "partial"
    assert evidence["source_normalization_status"] == "monotonic_downgrade"
    assert evidence["items"][0]["citation_id"] == 1
    assert "cite_uid" not in evidence["items"][0]
    evidence_content = json.loads(evidence["items"][0]["content"])
    assert evidence_content == {"text": "근거"}
    assert evidence["items"][0]["claim_ids"] == []
    assert evidence["items"][0]["relation"] is None


def test_model_facing_evidence_scrubs_nested_ledger_ids_but_keeps_public_code():
    content = json.dumps(
        {
            "cite_uid": "kcd9:J06.9",
            "code": "J06.9",
            "revision": "KCD-9",
            "nested": json.dumps(
                {"cite_uids": ["kcd9:J06.9"], "name": "급성 상기도감염"}
            ),
        },
        ensure_ascii=False,
    )

    cleaned = generation_module._model_facing_evidence_content(
        content,
        cite_uid="kcd9:J06.9",
    )
    payload = json.loads(cleaned)

    assert "cite_uid" not in payload
    assert payload["code"] == "J06.9"
    assert payload["revision"] == "KCD-9"
    assert json.loads(payload["nested"]) == {"name": "급성 상기도감염"}


def test_public_kcd_code_with_numeric_citation_is_not_treated_as_uid_leak():
    retrieval = RetrievalResult(
        status="sufficient",
        items=[
            EvidenceItem(
                cite_uid="kcd9:J06.9",
                source_tool="kcd_get_name",
                relevance_score=1.0,
                content='{"cite_uid":"kcd9:J06.9","code":"J06.9"}',
            )
        ],
    )

    violations = generation_module._final_violations(
        L2Completion(content="KCD-9의 공개 질병 코드는 J06.9입니다 [1]"),
        retrieval=retrieval,
    )

    assert violations == []


def test_model_facing_evidence_fails_closed_beyond_nesting_limit():
    nested: dict[str, object] = {
        "cite_uid": "opaque:deep:uid",
        "public_code": "J06.9",
    }
    for index in range(16):
        nested = {f"level_{index}": nested}

    cleaned = generation_module._model_facing_evidence_content(
        json.dumps(nested),
        cite_uid="opaque:deep:uid",
    )

    assert "cite_uid" not in cleaned
    assert "opaque:deep:uid" not in cleaned
    assert "[nested-evidence-redacted]" in cleaned


def test_model_facing_evidence_scrubs_internal_tool_protocol_but_keeps_facts():
    content = json.dumps(
        {
            "tool_call": {
                "name": "openapi_law_get_article",
                "arguments": {"article": "제21조"},
            },
            "source": "openapi_law_get_article",
            "article": "제21조",
            "text": "진료기록 보존에 관한 조문 원문",
        },
        ensure_ascii=False,
    )

    cleaned = generation_module._model_facing_evidence_content(
        content,
        cite_uid="law:medical:21",
    )
    payload = json.loads(cleaned)

    assert "tool_call" not in cleaned
    assert "openapi_law_get_article" not in cleaned
    assert payload["source"] == "[internal-source]"
    assert payload["article"] == "제21조"
    assert "진료기록 보존" in payload["text"]


def test_model_facing_evidence_scrubs_internal_identifiers_from_object_keys():
    content = json.dumps(
        {
            "openapi_law_get_article": {"article": "제21조"},
            "law:medical:21": {"text": "조문 원문"},
        },
        ensure_ascii=False,
    )

    cleaned = generation_module._model_facing_evidence_content(
        content,
        cite_uid="law:medical:21",
    )
    payload = json.loads(cleaned)

    assert "openapi_law_get_article" not in cleaned
    assert "law:medical:21" not in cleaned
    assert payload["[internal-source]"]["article"] == "제21조"
    assert payload["[internal-id]"]["text"] == "조문 원문"


def test_model_facing_retrieval_note_scrubs_protocol_and_ledger_identifiers():
    cleaned = generation_module._model_facing_retrieval_note(
        "한계: tool_call openapi_law_get_article cite_uid=law:medical:21",
        cite_uids=["law:medical:21"],
    )

    assert "tool_call" not in cleaned
    assert "openapi_law_get_article" not in cleaned
    assert "cite_uid" not in cleaned
    assert "law:medical:21" not in cleaned
    assert "한계" in cleaned


def test_identifier_only_items_are_not_exposed_as_citable_evidence():
    retrieval = RetrievalResult(
        status="partial",
        items=[
            EvidenceItem(
                cite_uid="source:one",
                source_tool="kcd_get_name",
                relevance_score=0.9,
                content=json.dumps(
                    {"cite_uid": "source:one", "related_source": "source:two"}
                ),
            ),
            EvidenceItem(
                cite_uid="source:two",
                source_tool="kcd_get_name",
                relevance_score=0.8,
                content=json.dumps(
                    {"cite_uid": "source:two", "related_source": "source:one"}
                ),
            ),
        ],
    )

    envelope = json.loads(generation_module._evidence_json(retrieval))
    serialized = json.dumps(envelope, ensure_ascii=False)

    assert "source:one" not in serialized
    assert "source:two" not in serialized
    assert envelope["items"] == []
    assert envelope["evidence_status"] == "none"


def test_citable_item_scrubs_uid_of_a_dropped_cross_referenced_shell():
    retrieval = RetrievalResult(
        status="partial",
        items=[
            EvidenceItem(
                cite_uid="empty:one",
                source_tool="kcd_get_name",
                relevance_score=0.7,
                content='{"cite_uid":"empty:one"}',
            ),
            EvidenceItem(
                cite_uid="fact:two",
                source_tool="kcd_get_name",
                relevance_score=0.9,
                content=json.dumps(
                    {"text": "공개 질병 코드 설명", "related": "empty:one"}
                ),
            ),
        ],
    )

    envelope = json.loads(generation_module._evidence_json(retrieval))
    serialized = json.dumps(envelope, ensure_ascii=False)

    assert len(envelope["items"]) == 1
    assert envelope["items"][0]["citation_id"] == 1
    assert "공개 질병 코드 설명" in serialized
    assert "empty:one" not in serialized
    assert "fact:two" not in serialized


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_generation_does_not_execute_tool_call_with_invalid_finish_reason(
    finish_reason,
):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "truncated-retrieval",
                        "retrieve_relevant_content",
                        {"query": "식약처 공식 허가 근거"},
                    )
                ],
                finish_reason=finish_reason,
            ),
            L2Completion(content="근거 요청이 유효하지 않아 제한적으로 답변합니다."),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="식약처 공식 허가 근거를 알려줘")]
    )

    assert answer == "근거 요청이 유효하지 않아 제한적으로 답변합니다."
    assert retrieval.queries == []
    assert len(l2.calls) == 2
    envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert envelope["final_phase_context"]["phase"] == "mcp_failure"


async def test_generation_accepts_official_provider_stop_for_structured_tool_call():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "provider-stop",
                        "retrieve_relevant_content",
                        {"query": "현행 KCD J06.9 공식 명칭"},
                    )
                ],
                finish_reason="stop",
            ),
            L2Completion(content="공식 근거를 반영한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="현행 KCD J06.9 공식 명칭과 출처")]
    )

    assert answer == "공식 근거를 반영한 답변 [1]"
    assert retrieval.queries == ["현행 KCD J06.9 공식 명칭과 출처"]


async def test_required_retrieval_plain_text_is_not_accepted_without_tool_execution():
    l2 = ScriptedL2(
        [
            L2Completion(content="근거를 조회하지 않고 만든 답"),
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-after-retry",
                        "retrieve_relevant_content",
                        {"query": "공식 고혈압 진료지침 출처"},
                    )
                ]
            ),
            L2Completion(content="조회한 근거에 기반한 답 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려주세요")]
    )

    assert answer == "조회한 근거에 기반한 답 [1]"
    assert retrieval.queries == ["공식 진료지침 출처를 알려주세요"]
    assert len(l2.calls) == 3
    assert l2.calls[0]["tool_choice"]["function"]["name"] == (
        "retrieve_relevant_content"
    )
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "retrieve_relevant_content"
    )
    assert l2.calls[0]["attempt_timeout_seconds"] == 25
    assert l2.calls[1]["attempt_timeout_seconds"] == 10


async def test_required_retrieval_degrades_to_l2_no_evidence_answer_after_two_ignores():
    l2 = ScriptedL2(
        [
            L2Completion(content="첫 번째 기억 기반 답"),
            L2Completion(content="두 번째 기억 기반 답"),
            L2Completion(content="공식 근거를 확인하지 못했다고 밝힌 제한적 답"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="최신 공식 진료지침 출처")]
    )

    assert answer == "공식 근거를 확인하지 못했다고 밝힌 제한적 답"
    assert retrieval.queries == []
    assert len(l2.calls) == 3
    assert l2.calls[2]["attempt_timeout_seconds"] == 45
    assert l2.calls[2]["allow_blank_recovery"] is False


async def test_generation_rejects_parallel_or_mixed_application_calls_without_execution():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call("one", "retrieve_relevant_content", {"query": "one"}),
                    tool_call("two", "unexpected", {}),
                ]
            ),
            L2Completion(content="검색하지 않았다고 밝힌 L2 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "검색하지 않았다고 밝힌 L2 답변"
    assert retrieval.queries == []
    assert all(message["role"] != "tool" for message in l2.calls[1]["messages"])
    assert "retrieve_relevant_content" not in l2.calls[1]["messages"][0]["content"]
    final_envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert final_envelope["final_phase_context"]["phase"] == "mcp_failure"
    assert "tools" not in l2.calls[1]


async def test_generation_rejects_text_mixed_with_retrieval_call_without_execution():
    l2 = ScriptedL2(
        [
            L2Completion(
                content="먼저 사용자에게 보일 텍스트",
                tool_calls=[
                    tool_call(
                        "mixed",
                        "retrieve_relevant_content",
                        {"query": "공식 지침"},
                    )
                ],
            ),
            L2Completion(content="검색을 실행하지 않은 L2 최종 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="공식 지침을 알려줘")]
    )

    assert answer == "검색을 실행하지 않은 L2 최종 답변"
    assert retrieval.queries == []
    assert all(message["role"] != "tool" for message in l2.calls[1]["messages"])
    final_envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert final_envelope["final_phase_context"]["phase"] == "mcp_failure"


async def test_generation_query_contract_forbids_extra_fields_and_control_characters():
    for arguments in (
        {"query": "valid", "hidden": "extra"},
        {"query": "bad\u0000query"},
        {"query": "x" * 2_049},
        '{"query":"first","query":"second"}',
    ):
        l2 = ScriptedL2(
            [
                L2Completion(
                    tool_calls=[
                        tool_call("bad", "retrieve_relevant_content", arguments)
                    ]
                ),
                L2Completion(content="L2 제한 답변"),
            ]
        )
        retrieval = FakeRetrieval()
        assert await GenerationEngine(l2, retrieval).answer(
            [ChatMessage(role="user", content="질문")]
        ) == "L2 제한 답변"
        assert retrieval.queries == []


@pytest.mark.parametrize(
    "model_query",
    [
        "홍길동 010-1234-5678 와파린 허가사항",
        "person@example.com의 검사 근거",
        "전체 history: user=와파린 assistant=답변",
    ],
)
async def test_safe_single_turn_does_not_forward_unsafe_model_rewrite(model_query):
    user_text = "식약처 타이레놀 허가사항을 확인해줘"
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "unsafe",
                        "retrieve_relevant_content",
                        {"query": model_query},
                    )
                ]
            ),
            L2Completion(content="사용자 원문에 근거한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content=user_text)]
    )

    assert retrieval.queries == [user_text]


@pytest.mark.parametrize(
    "user_text",
    [
        "홍길동 010-1234-5678 식약처 와파린 허가사항",
        "person@example.com의 식약처 검사 근거",
        "환자번호 12345678인 홍길동의 식약처 와파린 허가사항",
        "여권번호 M12345678 식약처 와파린 허가사항",
        "주소: 서울시 중구 세종대로 110 식약처 와파린 허가사항",
        "환자 이름: 홍길동 식약처 와파린 허가사항",
        "식약처 전체 history: user=와파린 assistant=답변",
    ],
)
async def test_unsafe_user_text_is_not_forwarded_even_after_safe_model_rewrite(user_text):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "sanitized",
                        "retrieve_relevant_content",
                        {"query": "식약처 와파린 허가사항"},
                    )
                ]
            ),
            L2Completion(content="식별정보를 제외하고 다시 질문해 주세요."),
        ]
    )
    retrieval = FakeRetrieval()

    await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content=user_text)]
    )

    assert retrieval.queries == []


async def test_single_turn_retrieval_rejects_anaphora_without_user_context():
    messages = [ChatMessage(role="user", content="그 약 임신 중 금기 최신 근거 알려줘")]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "unresolved",
                        "retrieve_relevant_content",
                        {"query": "그 약 임신 중 금기 최신 근거"},
                    )
                ]
            ),
            L2Completion(content="어떤 약인지 짧게 확인해 주세요."),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(messages)

    assert answer == "어떤 약인지 짧게 확인해 주세요."
    assert retrieval.queries == []
    final_envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    assert final_envelope["final_phase_context"]["reason_code"] == (
        "invalid_evidence_request"
    )


async def test_multi_turn_retrieval_accepts_user_grounded_self_contained_query():
    messages = [
        ChatMessage(role="user", content="와파린을 복용 중이에요"),
        ChatMessage(role="assistant", content="이전 답변"),
        ChatMessage(role="user", content="그 약 임신 중 금기 최신 근거 알려줘"),
    ]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "resolved",
                        "retrieve_relevant_content",
                        {"query": "와파린 임신 중 금기 최신 근거"},
                    )
                ]
            ),
            L2Completion(content="근거를 반영한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(messages)

    assert answer == "근거를 반영한 답변 [1]"
    assert retrieval.queries == [
        "와파린을 복용 중이에요\n임신 중 금기 최신 근거 알려줘"
    ]


async def test_multi_turn_retrieval_rejects_reference_without_concrete_prior_target():
    messages = [
        ChatMessage(role="user", content="안녕하세요, 도와주세요"),
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?"),
        ChatMessage(role="user", content="그 약 임신 중 금기는?"),
    ]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "unresolved-after-greeting",
                        "retrieve_relevant_content",
                        {"query": "임의 약물 임신 금기"},
                    )
                ]
            ),
            L2Completion(content="약 이름을 알려 달라는 제한적 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    assert await GenerationEngine(l2, retrieval).answer(messages) == (
        "약 이름을 알려 달라는 제한적 답변"
    )
    assert retrieval.queries == []


@pytest.mark.parametrize(
    "prior",
    [
        "안녕하세요",
        "안녕하세요, 도와주세요",
        "무엇을 도와드릴까요?",
        "I understand",
        "This is fictional",
        "오늘은 좀 피곤해요",
    ],
)
def test_conversational_prior_cannot_establish_a_retrieval_target(prior):
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content=prior),
            ChatMessage(role="assistant", content="이전 답변"),
            ChatMessage(role="user", content="그 약 임신 중 금기는?"),
        ]
    )

    assert query is None
    assert error == "the user's referenced target is not established"


async def test_multi_turn_retrieval_ignores_model_invented_replacement_entity():
    messages = [
        ChatMessage(role="user", content="와파린을 복용 중이에요"),
        ChatMessage(role="assistant", content="이전 답변"),
        ChatMessage(role="user", content="그 약 임신 중 금기 최신 근거 알려줘"),
    ]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "invented",
                        "retrieve_relevant_content",
                        {"query": "아스피린 임신 중 금기 최신 근거"},
                    )
                ]
            ),
            L2Completion(content="사용자 원문 근거를 반영한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    assert await GenerationEngine(l2, retrieval).answer(messages) == (
        "사용자 원문 근거를 반영한 답변 [1]"
    )
    assert len(retrieval.queries) == 1
    assert "와파린을 복용 중이에요" in retrieval.queries[0]
    assert "임신 중 금기 최신 근거 알려줘" in retrieval.queries[0]
    assert "그 약" not in retrieval.queries[0]
    assert "아스피린" not in retrieval.queries[0]


@pytest.mark.parametrize(
    "latest_user",
    [
        "임신 중 사용은?",
        "상호작용은?",
        "부작용은?",
        "허가사항은?",
        "급여는?",
        "용법 알려줘",
        "효능은?",
        "같이 먹어도 되나요?",
        "계속 먹어도 되나요?",
        "아직 먹어도 되나요?",
        "이거 먹어도 돼?",
        "그걸 먹어도 돼?",
        "What about interactions?",
        "During pregnancy?",
    ],
)
async def test_common_elliptical_followups_preserve_exact_user_context(latest_user):
    messages = [
        ChatMessage(role="user", content="와파린을 복용 중이에요"),
        ChatMessage(role="assistant", content="확인했습니다."),
        ChatMessage(role="user", content=latest_user),
    ]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "elliptical",
                        "retrieve_relevant_content",
                        {"query": "아스피린이라는 새 대상을 모델이 덧붙임"},
                    )
                ]
            ),
            L2Completion(content="원문 문맥 기반 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    assert await GenerationEngine(l2, retrieval).answer(messages) == "원문 문맥 기반 답변 [1]"
    assert len(retrieval.queries) == 1
    assert "와파린을 복용 중이에요" in retrieval.queries[0]
    assert generation_module._remove_explicit_deictics(latest_user) in retrieval.queries[0]
    assert "아스피린" not in retrieval.queries[0]


@pytest.mark.parametrize(
    "self_contained",
    [
        "와파린 상호작용은?",
        "아스피린 부작용은?",
        "타이레놀 허가사항은?",
        "임신 중 이부프로펜 사용은?",
        "와파린 급여는?",
    ],
)
def test_explicit_target_is_not_misclassified_as_elliptical(self_contained):
    assert generation_module._needs_prior_user_context(self_contained) is False


def test_explicit_target_with_redundant_deictic_is_self_contained():
    text = "이 약 와파린의 임신 금기는?"

    assert generation_module._needs_prior_user_context(text) is False
    query, error = generation_module._authoritative_retrieval_query(
        [ChatMessage(role="user", content=text)]
    )
    assert (query, error) == (text, "")


def test_multiple_explicit_appositions_are_self_contained():
    text = "이 약 와파린과 이 약 아스피린의 상호작용은?"

    assert generation_module._needs_prior_user_context(text) is False
    assert generation_module._authoritative_retrieval_query(
        [ChatMessage(role="user", content=text)]
    ) == (text, "")


def test_context_resolution_skips_nearer_nonclinical_filler_turn():
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content="와파린을 복용 중이에요"),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="좀 더 자세히 알려줘"),
            ChatMessage(role="assistant", content="어떤 부분인지 말씀해 주세요."),
            ChatMessage(role="user", content="그 약 부작용은?"),
        ]
    )

    assert error == ""
    assert query == "와파린을 복용 중이에요\n부작용은?"


def test_explicit_topic_switch_does_not_reactivate_prior_source_request():
    messages = [
        ChatMessage(role="user", content="고혈압 최신 진료지침 출처를 찾아줘"),
        ChatMessage(role="assistant", content="이전 답변입니다."),
        ChatMessage(role="user", content="그건 됐고 오늘 할 가벼운 스트레칭을 알려줘"),
    ]

    assert generation_module._needs_prior_user_context(messages[-1].content or "") is False
    assert generation_module.requires_retrieval(messages) is False


def test_context_resolution_skips_nearer_predicate_only_followup():
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content="와파린을 복용 중이에요"),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="부작용을 더 자세히 알려줘"),
            ChatMessage(role="assistant", content="어떤 부분인지 말씀해 주세요."),
            ChatMessage(role="user", content="그 약 임신 금기는?"),
        ]
    )

    assert error == ""
    assert query == "와파린을 복용 중이에요\n임신 금기는?"


@pytest.mark.parametrize("prior", ["와파린", "타이레놀", "warfarin"])
def test_bare_user_written_drug_name_can_anchor_a_short_followup(prior):
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content=prior),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="부작용은?"),
        ]
    )

    assert error == ""
    assert query == f"{prior}\n부작용은?"


def test_heldout_english_drug_antecedent_builds_context_complete_query():
    prior = "This fictional patient takes lithium for maintenance treatment."
    latest = (
        "그 약을 복용하는 동안 ibuprofen이 추가되면 공식 라벨상 어떤 위험과 "
        "모니터링이 있나요? Please answer in English and do not tell the patient "
        "to change either drug on their own."
    )

    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content=prior),
            ChatMessage(role="assistant", content="이전 답변"),
            ChatMessage(role="user", content=latest),
        ]
    )

    assert error == ""
    assert query == (
        "This fictional patient takes lithium for maintenance treatment.\n"
        "복용하는 동안 ibuprofen이 추가되면 공식 라벨상 어떤 위험과 모니터링이 "
        "있나요? Please answer in English and do not tell the patient to change either "
        "drug on their own."
    )


@pytest.mark.parametrize("prior", ["피곤해요", "좋아요", "서울"])
def test_nonclinical_bare_token_cannot_anchor_a_drug_reference(prior):
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content=prior),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="그 약 부작용은?"),
        ]
    )

    assert query is None
    assert error == "the user's referenced target is not established"


@pytest.mark.parametrize(
    "elliptical",
    [
        "임신 중 괜찮나요?",
        "복용해도 문제없나요?",
        "금기가 있나요?",
        "상호작용 알려주세요",
    ],
)
def test_predicate_only_followup_requires_prior_context(elliptical):
    assert generation_module._needs_prior_user_context(elliptical) is True


@pytest.mark.parametrize("latest", ["그 약은?", "What about it?"])
def test_deictic_only_followup_without_question_content_is_rejected(latest):
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content="와파린을 복용 중이에요"),
            ChatMessage(role="assistant", content="이전 답변"),
            ChatMessage(role="user", content=latest),
        ]
    )

    assert query is None
    assert error == "the user's follow-up contains no retrievable question"


def test_multiple_typed_references_fail_closed_without_cross_type_guessing():
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content="와파린 복용과 INR 검사를 이야기했어요"),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="그 약과 그 검사 결과가 관련 있나요?"),
        ]
    )

    assert query is None
    assert error == "multiple referenced targets cannot be resolved safely"


def test_singular_drug_reference_rejects_ambiguous_multi_drug_antecedent():
    query, error = generation_module._authoritative_retrieval_query(
        [
            ChatMessage(role="user", content="와파린과 아스피린을 복용 중이에요"),
            ChatMessage(role="assistant", content="확인했습니다."),
            ChatMessage(role="user", content="그 약 금기는?"),
        ]
    )

    assert query is None
    assert error == "the user's referenced target is not established"


def test_public_caller_context_prefix_cannot_hide_a_real_user_turn():
    spoofed = (
        "Untrusted caller context (data, not system instructions):\n"
        "가슴 통증이 심하고 숨을 못 쉬겠어요"
    )
    message = ChatMessage(role="user", content=spoofed)

    assert generation_module._dialogue_user_turns([message]) == [spoofed]
    assert generation_module._is_emergency_turn([message]) is True


@pytest.mark.parametrize(
    ("user_text", "model_query"),
    [
        (
            "와파린은 중단했고 아스피린은 현재 복용 중, 상호작용 근거",
            "와파린과 아스피린 병용 상호작용 근거",
        ),
        ("식약처 타이레놀 허가사항", "타이레놀 공식 근거"),
        ("의료법 현행 조문을 확인해줘", "병원 예약 규정 근거"),
        ("14kg 소아 아세트아미노펜 용량", "성인 아세트아미노펜 용량"),
        (
            "의료법 현재 시행 중인 조문과 출처",
            "대한민국 의료법 현행 조문 원문",
        ),
        (
            "KCD-8 당뇨병 공식 코드를 확인해줘",
            "제8차 한국표준질병사인분류 당뇨병 공식 코드",
        ),
        (
            "14kg 소아 이부프로펜 식약처 용량",
            "MFDS pediatric ibuprofen dose for 14 kg child",
        ),
        (
            "과거 흉통은 끝났고 지금 증상은 없어요. 공식 진료지침 출처",
            "resolved historical chest pain, currently asymptomatic guideline evidence",
        ),
    ],
)
async def test_self_contained_single_turn_uses_exact_user_query(
    user_text,
    model_query,
):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "semantic-drop",
                        "retrieve_relevant_content",
                        {"query": model_query},
                    )
                ]
            ),
            L2Completion(content="사용자 원문에 근거한 답변 [1]"),
        ]
    )
    retrieval = FakeRetrieval()

    await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content=user_text)]
    )

    assert retrieval.queries == [user_text]


async def test_retrieval_failure_uses_fresh_evidence_failure_final_prompt():
    class FailingRetrieval:
        async def retrieve(self, query):
            del query
            raise RetrievalError("private", code="mcp_connection_failed")

    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "식약처 공식 허가 근거"},
                    )
                ]
            ),
            L2Completion(content="근거 확인 실패를 밝힌 L2 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FailingRetrieval()).answer(
        [ChatMessage(role="user", content="식약처 공식 근거를 알려줘")]
    )

    assert answer == "근거 확인 실패를 밝힌 L2 답변"
    envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    context = envelope["final_phase_context"]
    assert context["schema_version"] == "generation-final-context-v1"
    assert context["phase"] == "mcp_failure"
    assert context["evidence_status"] == "unavailable"
    assert context["reason_code"] == "source_unavailable"
    assert "evidence" not in context
    assert "tools" not in l2.calls[1]
    assert "retrieve_relevant_content" not in l2.calls[1]["messages"][0]["content"]


async def test_emergency_guard_skips_retrieval_and_keeps_l2_as_author():
    l2 = ScriptedL2([L2Completion(content="119에 즉시 연락하세요.")])

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="가슴 통증이 심하고 숨을 못 쉬겠어요")]
    )

    assert answer == "119에 즉시 연락하세요."
    assert "tools" not in l2.calls[0]
    assert l2.calls[0]["attempt_timeout_seconds"] == 45
    assert l2.calls[0]["max_tokens"] == 2_048
    assert l2.calls[0]["allow_blank_recovery"] is False
    assert "시간 민감한 건강 위험" in l2.calls[0]["messages"][0]["content"]
    assert "retrieve_relevant_content" not in l2.calls[0]["messages"][0]["content"]
    envelope = json.loads(l2.calls[0]["messages"][-1]["content"])
    assert envelope["latest_user_message"]["content"] == "가슴 통증이 심하고 숨을 못 쉬겠어요"
    assert envelope["final_phase_context"]["phase"] == "emergency"
    assert envelope["final_phase_context"]["evidence_status"] == "not_requested"
    assert "request_id" not in envelope


async def test_emergency_unverified_source_and_new_oral_dose_get_clean_recovery():
    unsafe = (
        "최신 공식 학회 지침을 확인했습니다. 아스피린 300mg을 지금 복용하세요. "
        "https://example.org/guideline"
    )
    safe = (
        "대한민국에 계시면 지금 119에 연락하고 안전한 곳에서 상담원의 지시를 "
        "따르세요. 새 약을 임의로 복용하지 마세요."
    )
    l2 = ScriptedL2(
        [
            L2Completion(content=unsafe, finish_reason="stop"),
            L2Completion(content=safe, finish_reason="stop"),
        ]
    )

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="가슴 통증과 숨가쁨이 지금 있어요")]
    )

    assert answer == safe
    assert len(l2.calls) == 2
    assert unsafe not in json.dumps(l2.calls[1]["messages"], ensure_ascii=False)
    assert json.loads(l2.calls[1]["messages"][-1]["content"])[
        "final_phase_context"
    ] == {
        "schema_version": "generation-final-context-v1",
        "phase": "emergency",
        "evidence_status": "not_requested",
    }


async def test_emergency_unverified_source_and_new_oral_dose_twice_fail_closed():
    unsafe = "현행 법령과 공식 저널을 조회했습니다. 300mg을 지금 씹어 드세요."
    l2 = ScriptedL2(
        [
            L2Completion(content=unsafe, finish_reason="stop"),
            L2Completion(content=unsafe, finish_reason="stop"),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, None).direct_answer(
            [ChatMessage(role="user", content="숨을 못 쉬고 가슴이 아파요")]
        )

    assert len(l2.calls) == 2


@pytest.mark.parametrize(
    "unsafe",
    [
        "공식 학회 자료를 보면 이 처치가 권고됩니다.",
        "현행 법령상 반드시 이렇게 해야 합니다.",
        "Take 325 mg now.",
        "Take 325 mg now if the dispatcher tells you to.",
        "Take 325 mg regardless of your existing rescue plan.",
        "기존 처방 계획은 무시하고 이 약 500 mg을 복용하세요.",
        "이 약 두 알을 지금 드세요.",
        "상처 부위를 심장보다 높게 유지하세요.",
    ],
)
def test_emergency_output_guard_rejects_generalized_unsafe_variants(unsafe):
    assert generation_module._contains_unsupported_emergency_claim(unsafe) is True


@pytest.mark.parametrize(
    "allowed",
    [
        "최신 공식 자료는 이번 응급 응답에서 확인할 수 없습니다.",
        "새 약을 임의로 복용하지 마세요.",
        "119 상담원이 지시하면 이미 처방된 구조약을 처방대로 사용하세요.",
        "Take your prescribed rescue medicine exactly as directed in your existing rescue plan.",
        "Do not take 325 mg.",
        "지혈대를 임의로 사용하지 말고 계속 직접 압박하세요.",
    ],
)
def test_emergency_output_guard_allows_limitations_and_conservative_exceptions(allowed):
    assert generation_module._contains_unsupported_emergency_claim(allowed) is False


async def test_emergency_legitimate_concise_dispatcher_response_passes():
    safe = (
        "대한민국에 계시면 지금 119에 연락하세요. 안전한 곳에서 기다리며 "
        "119 상담원의 지시를 따르세요."
    )
    l2 = ScriptedL2([L2Completion(content=safe, finish_reason="stop")])

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="갑자기 말이 어눌하고 한쪽 팔이 약해졌어요")]
    )

    assert answer == safe
    assert len(l2.calls) == 1


async def test_active_bleeding_unsafe_procedural_extras_get_clean_recovery():
    unsafe = "팔을 심장보다 높이 올리고 지혈대를 단단히 묶으세요."
    safe = (
        "지금 현지 응급번호로 연락하세요. 깨끗한 천으로 상처를 계속 단단히 "
        "직접 압박하고, 떼어 확인하지 말고 상담원의 지시를 따르세요."
    )
    l2 = ScriptedL2(
        [
            L2Completion(content=unsafe, finish_reason="stop"),
            L2Completion(content=safe, finish_reason="stop"),
        ]
    )

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="상처에서 피가 계속 나고 지혈이 안 돼요")]
    )

    assert answer == safe
    assert len(l2.calls) == 2


async def test_uncontrolled_bleeding_paraphrase_uses_emergency_output_guard():
    unsafe = "팔을 심장보다 높이 올리고 지혈대를 단단히 묶으세요."
    safe = (
        "지금 현지 응급번호로 연락하세요. 깨끗한 천으로 상처를 계속 단단히 "
        "직접 압박하고, 떼어 확인하지 말고 상담원의 지시를 따르세요."
    )
    l2 = ScriptedL2(
        [
            L2Completion(content=unsafe, finish_reason="stop"),
            L2Completion(content=safe, finish_reason="stop"),
        ]
    )

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="상처에서 피가 계속 흐르고 멈추지 않아요")]
    )

    assert answer == safe
    assert len(l2.calls) == 2


async def test_nonemergency_topic_switch_is_unaffected_by_emergency_output_guard():
    messages = [
        ChatMessage(role="user", content="어제 흉통은 끝났고 지금은 증상이 없어요."),
        ChatMessage(role="assistant", content="새 증상이 생기면 응급 도움을 받으세요."),
        ChatMessage(role="user", content="이제 수면 습관만 간단히 설명해 주세요."),
    ]
    direct = "일정한 기상 시간을 유지하고 늦은 카페인을 줄여 보세요."
    l2 = ScriptedL2([L2Completion(content=direct, finish_reason="stop")])

    assert generation_module._is_emergency_turn(messages) is False
    assert await GenerationEngine(l2, None).direct_answer(messages) == direct
    assert len(l2.calls) == 1


async def test_emergency_guard_ignores_assistant_safety_quote_on_latest_source_question():
    messages = [
        ChatMessage(role="user", content="혈압에 대해 알려줘"),
        ChatMessage(
            role="assistant",
            content="가슴 통증이나 숨을 못 쉬면 즉시 도움을 요청하세요.",
        ),
        ChatMessage(role="user", content="그 설명의 출처를 알려줘"),
    ]

    assert generation_module._is_emergency_turn(messages) is False


@pytest.mark.parametrize(
    "content",
    [
        "흉통은 없어요. 공식 지침 출처만 궁금해요.",
        "어제 흉통은 끝났고 지금은 괜찮아요. 출처를 알려줘요.",
        "기사 속 환자가 흉통이라고 했어요. 최신 지침 출처는?",
        "10년 전 흉통 병력이 있어요. 일반 원리를 알려줘요.",
        "만약 흉통이 생기면 어떻게 해야 하나요?",
        "어제는 가슴 통증이 있었지만 지금은 통증도 호흡곤란도 없어요.",
    ],
)
async def test_emergency_guard_excludes_clear_negated_resolved_or_quoted_mentions(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is False


@pytest.mark.parametrize(
    "content",
    [
        "약 30알을 방금 복용했어요",
        "청색증이 있고 호흡곤란이 심해요",
        "한쪽 편측 위약과 발음 이상이 생겼어요",
        "목이 붓고 쌕쌕거려요",
        "상처에서 지혈이 안 돼요",
    ],
)
def test_emergency_guard_catches_current_high_risk_paraphrases(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is True


@pytest.mark.parametrize(
    "content",
    [
        "상처에서 피가 계속 흐르고 멈추지 않아요",
        "피가 안 멈춰요",
        "Blood keeps flowing and won't stop.",
    ],
)
def test_emergency_guard_catches_current_uncontrolled_bleeding(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is True


@pytest.mark.parametrize(
    "content",
    [
        "어제는 피가 계속 흘렀지만 지금은 멈췄어요",
        "만약 피가 계속 흐르면 어떻게 해야 하나요?",
        "'피가 계속 흐른다'라는 문구를 설명해 줘",
        "지금은 상처에서 피가 계속 흐르지 않아요",
    ],
)
def test_emergency_guard_excludes_noncurrent_bleeding_mentions(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is False


@pytest.mark.parametrize(
    "content",
    [
        "I have chest pressure with shortness of breath right now.",
        "Chest tightness and diaphoresis started suddenly.",
        "I have chest tightness but no shortness of breath or sweating.",
        "가슴을 짓누르는 느낌과 호흡곤란이 지금 같이 있어요.",
        "가슴 압박과 식은땀이 갑자기 생겼어요.",
    ],
)
def test_emergency_guard_catches_current_chest_pressure_clusters(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is True


@pytest.mark.parametrize(
    "content",
    [
        "현재는 가슴 압박, 호흡곤란, 식은땀 모두 전혀 없습니다.",
        "3년 전 흉통과 숨가쁨이 있었지만 지금은 흉통, 호흡곤란 모두 없습니다.",
        "가슴 압박이라는 표현의 뜻과 호흡곤란의 정의가 궁금합니다.",
    ],
)
def test_emergency_guard_does_not_promote_negated_or_noncurrent_clusters(content):
    assert generation_module._is_emergency_turn(
        [ChatMessage(role="user", content=content)]
    ) is False


async def test_emergency_protocol_output_gets_one_fresh_final_only_recovery():
    l2 = ScriptedL2(
        [
            L2Completion(
                content=json.dumps(
                    {
                        "schema_version": "generation-reroute-v2",
                        "action": "retry_normal_retrieval",
                    }
                )
            ),
            L2Completion(content="현재 증상이면 즉시 응급서비스에 연락하세요."),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [
            ChatMessage(
                role="user",
                content="흉통의 최신 공식 진료지침 출처는?",
            )
        ]
    )

    assert answer == "현재 증상이면 즉시 응급서비스에 연락하세요."
    assert [call["attempt_timeout_seconds"] for call in l2.calls] == [45, 30]
    assert l2.calls[0]["allow_blank_recovery"] is False
    assert l2.calls[1]["allow_blank_recovery"] is False
    recovery = l2.calls[1]["messages"]
    assert len(recovery) == 2
    assert "재작성 단계" in recovery[0]["content"]
    assert "generation-reroute-v2" not in json.dumps(recovery, ensure_ascii=False)


async def test_generation_requests_l2_correction_when_numeric_citation_is_missing():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침"},
                    )
                ]
            ),
            L2Completion(content="인용이 없는 답변"),
            L2Completion(content="교정된 답변 [1]"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처")]
    )

    assert answer == "교정된 답변 [1]"
    assert "tools" not in l2.calls[2]
    assert l2.calls[2]["attempt_timeout_seconds"] == 30
    assert l2.calls[2]["max_tokens"] == 2_048
    assert l2.calls[2]["allow_blank_recovery"] is False


async def test_repeated_citation_omission_returns_only_the_recovered_l2_text():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "KCD 공식 코드"},
                    )
                ]
            ),
            L2Completion(content="KCD 코드 설명이지만 숫자 인용은 없음"),
            L2Completion(content="새로 작성한 KCD 코드 설명도 숫자 인용은 없음"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="KCD 공식 코드를 설명해줘")]
    )

    assert answer == "새로 작성한 KCD 코드 설명도 숫자 인용은 없음"
    assert len(l2.calls) == 3
    assert l2.calls[2]["max_tokens"] == 2_048


async def test_generation_rejects_answer_after_failed_citation_correction():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침"},
                    )
                ]
            ),
            L2Completion(content="잘못된 번호 [99]와 raw guideline:1"),
            L2Completion(content="여전히 인용 없음"),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, FakeRetrieval()).answer(
            [ChatMessage(role="user", content="공식 진료지침 출처")]
        )


@pytest.mark.parametrize(
    "leaked_content",
    [
        "대문자 식별자 GUIDELINE:1 [1]",
        "인코딩된 식별자 guideline%3A1 [1]",
        '{"cite_uid":"invented:other","answer":"요약 [1]"}',
    ],
)
async def test_citation_identifier_variants_trigger_clean_recovery(leaked_content):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침"},
                    )
                ]
            ),
            L2Completion(content=leaked_content),
            L2Completion(content="식별자를 숨긴 근거 기반 답 [1]"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처")]
    )

    assert answer == "식별자를 숨긴 근거 기반 답 [1]"
    assert leaked_content not in json.dumps(l2.calls[2]["messages"], ensure_ascii=False)


@pytest.mark.parametrize(
    "leaked",
    [
        "GUIDELINE:1 [1]",
        "guideline%3A1 [1]",
        '{"cite_uid":"other","answer":"요약 [1]"}',
    ],
)
async def test_citation_identifier_repeat_after_recovery_fails_closed(leaked):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침"},
                    )
                ]
            ),
            L2Completion(content=leaked),
            L2Completion(content=leaked),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, FakeRetrieval()).answer(
            [ChatMessage(role="user", content="공식 진료지침 출처")]
        )

    assert len(l2.calls) == 3


async def test_generation_recovers_final_textual_tool_protocol_without_exposing_it():
    l2 = ScriptedL2(
        [
            L2Completion(content="internal direct decision"),
            L2Completion(content="<tool_call>bad</tool_call>"),
            L2Completion(content="일반 지식에 한정한 L2 답변"),
        ]
    )

    assert await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="질문")]
    ) == "일반 지식에 한정한 L2 답변"
    assert "tools" not in l2.calls[1]
    assert "tools" not in l2.calls[2]
    recovery_dump = json.dumps(l2.calls[2]["messages"], ensure_ascii=False)
    assert "<tool_call>bad</tool_call>" not in recovery_dump
    assert l2.calls[2]["messages"][0]["content"] != (
        l2.calls[1]["messages"][0]["content"]
    )


async def test_generation_drops_unexpected_final_tool_call_before_fresh_recovery():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침"},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    tool_call(
                        "unexpected-2",
                        "retrieve_relevant_content",
                        {"query": "반복 검색"},
                    )
                ]
            ),
            L2Completion(content="복구된 최종 답변 [1]"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처")]
    )

    assert answer == "복구된 최종 답변 [1]"
    recovery_messages = l2.calls[2]["messages"]
    assert len(recovery_messages) == 2
    assert all(message["role"] != "tool" for message in recovery_messages)
    recovery_dump = json.dumps(recovery_messages, ensure_ascii=False)
    assert "unexpected-2" not in recovery_dump
    assert "retrieve_relevant_content" not in recovery_messages[0]["content"]


@pytest.mark.parametrize(
    "invalid_content",
    [
        '<function_call name="retrieve_relevant_content" />',
        '{"name":"retrieve_relevant_content","arguments":{"query":"x"}}',
        "retrieve_relevant_content(query='x')",
    ],
)
async def test_direct_final_protocol_invalid_twice_fails_closed(invalid_content):
    l2 = ScriptedL2(
        [
            L2Completion(content=invalid_content),
            L2Completion(content=invalid_content),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, None).direct_answer(
            [ChatMessage(role="user", content="일반적인 건강 질문")]
        )

    assert len(l2.calls) == 2
    assert "retrieve_relevant_content" not in l2.calls[0]["messages"][0]["content"]
    assert invalid_content not in json.dumps(l2.calls[1]["messages"], ensure_ascii=False)


async def test_direct_final_allows_user_requested_nonprotocol_json():
    answer_json = '{"summary":"수분을 섭취하고 증상을 관찰하세요","urgent":false}'
    l2 = ScriptedL2([L2Completion(content=answer_json, finish_reason="stop")])

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="JSON으로 요약해줘")]
    )

    assert answer == answer_json
    assert len(l2.calls) == 1


async def test_initial_final_timeout_gets_one_fresh_bounded_recovery():
    class TimeoutThenAnswer:
        def __init__(self):
            self.calls = []

        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise UpstreamTimeoutError("synthetic initial final timeout")
            return L2Completion(content="복구된 최종 답변", finish_reason="stop")

    l2 = TimeoutThenAnswer()

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="일반적인 건강 질문")]
    )

    assert answer == "복구된 최종 답변"
    assert [call["attempt_timeout_seconds"] for call in l2.calls] == [45, 30]
    assert l2.calls[1]["max_tokens"] == 2_048
    assert l2.calls[1]["allow_blank_recovery"] is False
    assert len(l2.calls[1]["messages"]) == 2
    assert "재작성 단계" in l2.calls[1]["messages"][0]["content"]


@pytest.mark.parametrize(
    "protocol_text",
    [
        "<tool_call><arg_key>query</arg_key></tool_call>",
        '<function_call name="retrieve_relevant_content" />',
        "<retrieve_relevant_content>query</retrieve_relevant_content>",
        '{"name":"retrieve_relevant_content","arguments":{"query":"x"}}',
        '{"tool":"retrieve_relevant_content","arguments":{"query":"x"}}',
        '{"action":"retrieve_relevant_content","query":"x"}',
        "retrieve_relevant_content(query='x')",
        "retrieve_relevant_content",
        "retrieve_relevant_content를 실행함",
        'assistant emitted "tool_call" instead of an answer',
        "<tool_calls><call /></tool_calls>",
        '<tool_calls type="function">{"name":"unknown","arguments":{}}</tool_calls>',
        '```json\n{"tool_call":{"name":"unknown","arguments":{}}}\n```',
        '```json\n{"name":"unknown_lookup","arguments":{"q":"x"}}\n```',
    ],
)
def test_final_detector_rejects_pseudo_tool_shapes(protocol_text):
    assert generation_module._looks_like_tool_protocol(protocol_text) is True


@pytest.mark.parametrize(
    "plain_json",
    [
        '{"function":{"name":"kidney"},"summary":"신장 기능"}',
        '{"action":"hydrate","query":"오늘 할 일","urgent":false}',
        '[{"name":"blood pressure","value":"120/80"}]',
    ],
)
def test_final_detector_allows_benign_structured_user_answers(plain_json):
    assert generation_module._looks_like_tool_protocol(plain_json) is False


async def test_nonstop_finish_reason_uses_exactly_one_clean_recovery():
    l2 = ScriptedL2(
        [
            L2Completion(content="중간에 끊긴 답", finish_reason="length"),
            L2Completion(content="완결된 사용자 답", finish_reason="stop"),
        ]
    )

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "완결된 사용자 답"
    assert len(l2.calls) == 2
    assert "중간에 끊긴 답" not in json.dumps(l2.calls[1]["messages"], ensure_ascii=False)


async def test_nonstop_finish_reason_twice_fails_closed():
    l2 = ScriptedL2(
        [
            L2Completion(content="첫 중단", finish_reason="length"),
            L2Completion(content="두 번째 중단", finish_reason="length"),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, None).direct_answer(
            [ChatMessage(role="user", content="질문")]
        )

    assert len(l2.calls) == 2


async def test_final_tool_calls_twice_fail_closed_after_one_recovery():
    invalid_call = tool_call(
        "unexpected",
        "retrieve_relevant_content",
        {"query": "repeat"},
    )
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[invalid_call], finish_reason="tool_calls"),
            L2Completion(tool_calls=[invalid_call], finish_reason="tool_calls"),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, None).direct_answer(
            [ChatMessage(role="user", content="질문")]
        )

    assert len(l2.calls) == 2
    assert "unexpected" not in json.dumps(l2.calls[1]["messages"], ensure_ascii=False)


async def test_required_retrieval_no_tool_final_invalid_gets_one_clean_recovery():
    l2 = ScriptedL2(
        [
            L2Completion(content="decision ignored"),
            L2Completion(content="forced decision ignored"),
            L2Completion(content="retrieve_relevant_content(query='retry')"),
            L2Completion(content="공식 근거를 확인하지 못한 제한적 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="현행 공식 진료지침 출처를 알려줘")]
    )

    assert answer == "공식 근거를 확인하지 못한 제한적 답변"
    assert len(l2.calls) == 4
    assert "재작성 단계" in l2.calls[3]["messages"][0]["content"]
    assert "decision ignored" not in json.dumps(l2.calls[3]["messages"], ensure_ascii=False)


@pytest.mark.parametrize(
    "invalid_content",
    [
        "<tool_call>again</tool_call>",
        '{"tool_call":{"name":"unknown_lookup","arguments":{"query":"x"}}}',
    ],
)
async def test_mcp_error_final_invalid_gets_one_clean_recovery(invalid_content):
    class FailingRetrieval:
        async def retrieve(self, query):
            del query
            raise RetrievalError("private", code="mcp_timeout")

    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "식약처 공식 허가 근거"},
                    )
                ]
            ),
            L2Completion(content=invalid_content),
            L2Completion(content="근거 확인 실패와 확인 경로를 밝힌 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FailingRetrieval()).answer(
        [ChatMessage(role="user", content="식약처 공식 허가 근거")]
    )

    assert answer == "근거 확인 실패와 확인 경로를 밝힌 답변"
    assert len(l2.calls) == 3
    assert invalid_content not in json.dumps(l2.calls[2]["messages"], ensure_ascii=False)


@pytest.mark.parametrize(
    "unsupported_claim",
    [
        "현행 식약처 허가사항상 성인은 1회 500 mg을 4시간 간격으로 복용합니다.",
        "성인은 1회 500mg을 4시간 간격으로 복용합니다.",
        "현행 의료법 제21조에 따라 진료기록을 10년 보존해야 합니다.",
        "제품 라벨상 투여 시작 후 2주, 이후 매 3개월마다 간기능을 모니터링해야 합니다.",
        "공식 라벨상 임신부에게 금기입니다.",
    ],
)
async def test_mcp_failure_authoritative_claim_gets_clean_l2_recovery(
    unsupported_claim,
):
    class FailingRetrieval:
        async def retrieve(self, query):
            del query
            raise RetrievalError("private", code="mcp_timeout")

    safe_recovery = (
        "이번 시도에서는 현행 공식 근거를 확인하지 못해 구체 기준을 단정할 수 "
        "없습니다. 현재 제품 설명서와 담당 의료진에게 확인해 주세요."
    )
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 근거 확인"},
                    )
                ]
            ),
            L2Completion(content=unsupported_claim),
            L2Completion(content=safe_recovery),
        ]
    )

    answer = await GenerationEngine(l2, FailingRetrieval()).answer(
        [ChatMessage(role="user", content="현행 공식 기준을 알려줘")]
    )

    assert answer == safe_recovery
    assert len(l2.calls) == 3
    assert unsupported_claim not in json.dumps(
        l2.calls[2]["messages"], ensure_ascii=False
    )
    assert [call["attempt_timeout_seconds"] for call in l2.calls] == [25, 45, 30]


async def test_successful_retrieval_with_no_citable_content_uses_no_evidence_guard():
    retrieval = FakeRetrieval(
        RetrievalResult(
            status="partial",
            items=[
                EvidenceItem(
                    cite_uid="empty:1",
                    source_tool="kcd_get_name",
                    relevance_score=0.8,
                    content='{"cite_uid":"empty:1"}',
                )
            ],
            execution_status="ok",
            semantic_reason="coverage_gap",
        )
    )
    unsafe = "현행 KCD 공식 코드는 I10입니다 [1]."
    safe = (
        "이번 조회에서는 인용 가능한 공식 코드 근거를 확인하지 못했습니다. "
        "현재 KCD 분류표에서 확인해 주세요."
    )
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-empty",
                        "retrieve_relevant_content",
                        {"query": "현행 KCD 공식 코드"},
                    )
                ]
            ),
            L2Completion(content=unsafe),
            L2Completion(content=safe),
        ]
    )

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="현행 KCD 공식 코드를 알려줘")]
    )

    assert answer == safe
    envelope = json.loads(l2.calls[1]["messages"][-1]["content"])
    context = envelope["final_phase_context"]
    assert context["phase"] == "post_retrieval"
    assert context["evidence_status"] == "none"
    assert context["evidence"]["items"] == []
    assert context["evidence"]["evidence_status"] == "none"


async def test_no_evidence_authoritative_claim_twice_fails_closed():
    unsafe = "현행 의료법 제21조에 따라 10년 보관해야 합니다."
    l2 = ScriptedL2(
        [
            L2Completion(content="decision ignored"),
            L2Completion(content="forced decision ignored"),
            L2Completion(content=unsafe),
            L2Completion(content=unsafe),
        ]
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await GenerationEngine(l2, FakeRetrieval()).answer(
            [ChatMessage(role="user", content="현행 의료법 조문을 알려줘")]
        )

    assert len(l2.calls) == 4


@pytest.mark.parametrize(
    "limitation",
    [
        "1회 500mg인지 이번 시도에서는 확인할 수 없습니다. 제품 설명서를 확인해 주세요.",
        "현행 의료법 제21조 적용 여부는 확인하지 못했습니다.",
        "검사 주기는 근거가 없어 현재 자료에서 확인이 필요합니다.",
    ],
)
def test_no_evidence_guard_allows_explicit_limitations(limitation):
    assert generation_module._contains_unsupported_authoritative_claim(limitation) is False


def test_no_evidence_guard_does_not_match_is_inside_pharmacist():
    limitation = (
        "I could not verify the current package insert; "
        "check the official label or a pharmacist."
    )

    assert generation_module._contains_unsupported_authoritative_claim(limitation) is False


@pytest.mark.parametrize(
    "unsupported_claim",
    [
        "공식 라벨: 임신부 금기",
        "식약처 허가사항: 소아 사용 금기",
        "Current guideline: avoid in pregnancy",
        "공식 라벨상 용량은 500 mg이고 근거 확인 필요",
    ],
)
def test_no_evidence_guard_rejects_material_official_claims_and_suffix_disclaimers(
    unsupported_claim,
):
    assert (
        generation_module._contains_unsupported_authoritative_claim(unsupported_claim)
        is True
    )


async def test_explicit_direct_mode_keeps_official_label_question_on_one_call_direct_phase():
    answer_text = "FDA 승인 적응증에 관한 일반적인 안내입니다."
    l2 = ScriptedL2([L2Completion(content=answer_text, finish_reason="stop")])

    answer = await GenerationEngine(l2, None).direct_answer(
        [ChatMessage(role="user", content="What is the FDA-approved indication?")]
    )

    assert answer == answer_text
    assert len(l2.calls) == 1
    envelope = json.loads(l2.calls[0]["messages"][-1]["content"])
    assert envelope["final_phase_context"]["phase"] == "direct"
    assert envelope["final_phase_context"]["evidence_status"] == "not_requested"


async def test_evidence_unavailable_answer_uses_guarded_no_evidence_phase():
    limitation = "요청한 FDA-approved indication은 현재 근거로 확인할 수 없습니다."
    l2 = ScriptedL2([L2Completion(content=limitation, finish_reason="stop")])

    answer = await GenerationEngine(l2, None).evidence_unavailable_answer(
        [ChatMessage(role="user", content="What is the FDA-approved indication?")]
    )

    assert answer == limitation
    envelope = json.loads(l2.calls[0]["messages"][-1]["content"])
    assert envelope["final_phase_context"]["phase"] == "mcp_failure"
    assert envelope["final_phase_context"]["evidence_status"] == "unavailable"


async def test_grounded_authoritative_claim_and_direct_general_answer_are_unaffected():
    grounded = "현행 자료의 기준값은 140 mmHg입니다 [1]."
    rag_l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-grounded",
                        "retrieve_relevant_content",
                        {"query": "현행 공식 기준값"},
                    )
                ]
            ),
            L2Completion(content=grounded),
        ]
    )
    assert await GenerationEngine(rag_l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="현행 공식 기준값을 알려줘")]
    ) == grounded

    general = "일반적으로 성인은 하루 7~9시간 수면을 목표로 할 수 있습니다."
    direct_l2 = ScriptedL2([L2Completion(content=general)])
    assert await GenerationEngine(direct_l2, None).direct_answer(
        [ChatMessage(role="user", content="건강한 수면 습관을 알려줘")]
    ) == general
    assert len(direct_l2.calls) == 1


def test_worst_case_generation_path_stays_within_request_deadline():
    retrieval_hard_slice_seconds = 50
    worst_case = (
        generation_module._INITIAL_TOOL_TIMEOUT_SECONDS
        + generation_module._FORCED_TOOL_RETRY_TIMEOUT_SECONDS
        + retrieval_hard_slice_seconds
        + generation_module._FINAL_GENERATION_TIMEOUT_SECONDS
        + generation_module._RECOVERY_TIMEOUT_SECONDS
    )

    assert worst_case == 160
    assert worst_case <= 165


def test_generation_source_contains_no_extra_final_answer_tool():
    source = inspect.getsource(__import__("lunit_hackathon.generation", fromlist=["*"]))
    forbidden = "submit" + "_final_answer"
    assert forbidden not in source
