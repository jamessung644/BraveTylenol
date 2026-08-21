import json

from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.schemas import (
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    ToolCall,
)


def tool_call(call_id: str, name: str, arguments: dict | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(
        id=call_id,
        function={"name": name, "arguments": raw},
    )


class ScriptedL2:
    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


class FakeRetrieval:
    def __init__(self):
        self.queries = []

    async def retrieve(self, query):
        self.queries.append(query)
        return RetrievalResult(
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


async def test_generation_returns_direct_l2_text_verbatim():
    l2 = ScriptedL2([L2Completion(content="  L2 원문 답변  ")])
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "  L2 원문 답변  "
    assert retrieval.queries == []
    assert l2.calls[0]["tools"][0]["function"]["name"] == "retrieve_relevant_content"
    assert l2.calls[0]["messages"][-1]["content"] == "질문"


async def test_generation_retrieves_then_returns_second_l2_text():
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
            L2Completion(content="근거를 반영한 L2 최종 답변 guideline:1"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="목표 혈압은?")]
    )

    assert answer == "근거를 반영한 L2 최종 답변 guideline:1"
    assert retrieval.queries == ["한국 고혈압 목표 혈압 진료지침"]
    tool_message = l2.calls[1]["messages"][-2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "retrieve-1"
    assert "guideline:1" in tool_message["content"]
    grounding = l2.calls[1]["messages"][-1]
    assert grounding["role"] == "system"
    assert "guideline:1" in grounding["content"]
    assert "verbatim" in grounding["content"]
    assert l2.calls[1]["tools"][0]["function"]["name"] == "submit_final_answer"
    assert l2.calls[1]["tool_choice"]["function"]["name"] == "submit_final_answer"


async def test_generation_forces_retrieval_for_official_kcd_evidence():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "KCD-8 I10 공식 근거"},
                    )
                ]
            ),
            L2Completion(content="공식 근거 답변 guideline:1"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="KCD-8 I10의 공식 근거와 출처를 알려줘")]
    )

    assert answer == "공식 근거 답변 guideline:1"
    assert l2.calls[0]["tool_choice"]["function"]["name"] == ("retrieve_relevant_content")


async def test_generation_asks_l2_to_correct_missing_citation_identifier():
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
            L2Completion(content="번호 인용만 있는 답변 [1]"),
            L2Completion(content="교정된 최종 답변 guideline:1"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려줘")]
    )

    assert answer == "교정된 최종 답변 guideline:1"
    correction = l2.calls[2]["messages"][-1]
    assert correction["role"] == "system"
    assert "guideline:1" in correction["content"]
    assert "numbered citations" in correction["content"]


async def test_generation_accepts_one_exact_identifier_from_multiple_evidence_items():
    class MultipleEvidenceRetrieval:
        async def retrieve(self, query):
            del query
            return RetrievalResult(
                status="sufficient",
                items=[
                    EvidenceItem(
                        cite_uid=f"source:{index}",
                        source_tool="lookup",
                        relevance_score=0.9,
                        content=f'{{"cite_uid":"source:{index}"}}',
                    )
                    for index in range(2)
                ],
            )

    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 근거"},
                    )
                ]
            ),
            L2Completion(content="첫 근거만 사용한 답변 source:0"),
        ]
    )

    answer = await GenerationEngine(l2, MultipleEvidenceRetrieval()).answer(
        [ChatMessage(role="user", content="공식 근거를 알려줘")]
    )

    assert answer == "첫 근거만 사용한 답변 source:0"
    assert len(l2.calls) == 2


async def test_generation_returns_l2_correction_if_identifier_is_still_missing():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 근거"},
                    )
                ]
            ),
            L2Completion(content="번호 인용 답변 [1]"),
            L2Completion(content="식별자를 여전히 생략한 L2 교정 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 근거를 알려줘")]
    )

    assert answer == "식별자를 여전히 생략한 L2 교정 답변"


async def test_generation_rewrites_textual_tool_protocol_after_no_evidence():
    class NoEvidenceRetrieval:
        async def retrieve(self, query):
            del query
            return RetrievalResult(status="no_evidence", note="not found")

    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "식약처 제품 근거"},
                    )
                ]
            ),
            L2Completion(
                content=("<tool_call>retrieve_relevant_content<arg_key>query</arg_key></tool_call>")
            ),
            L2Completion(content="공식 제품 근거를 찾지 못했습니다."),
        ]
    )

    answer = await GenerationEngine(l2, NoEvidenceRetrieval()).answer(
        [ChatMessage(role="user", content="식약처 근거를 알려줘")]
    )

    assert answer == "공식 제품 근거를 찾지 못했습니다."
    rewrite = l2.calls[2]["messages"][-1]
    assert rewrite["role"] == "system"
    assert "without tool-call syntax" in rewrite["content"]


async def test_generation_returns_structured_l2_final_submission():
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
                        "submit-1",
                        "submit_final_answer",
                        {"answer": "L2 구조화 최종 답변 guideline:1"},
                    )
                ]
            ),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려줘")]
    )

    assert answer == "L2 구조화 최종 답변 guideline:1"
    assert len(l2.calls) == 2


async def test_generation_accepts_extra_fields_in_l2_final_submission():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "retrieve-1",
                        "retrieve_relevant_content",
                        {"query": "공식 진료지침", "ignored_hint": "extra"},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    tool_call(
                        "submit-1",
                        "submit_final_answer",
                        {
                            "answer": "여분 필드가 있는 최종 답변 guideline:1",
                            "ignored_metadata": True,
                        },
                    )
                ]
            ),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려줘")]
    )

    assert answer == "여분 필드가 있는 최종 답변 guideline:1"
    assert retrieval.queries == ["공식 진료지침"]


async def test_generation_uses_text_alongside_invalid_final_tool_call():
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
                content="도구 인자는 잘못됐지만 사용 가능한 답변 guideline:1",
                tool_calls=[tool_call("submit-bad", "submit_final_answer", {})],
            ),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려줘")]
    )

    assert answer == "도구 인자는 잘못됐지만 사용 가능한 답변 guideline:1"
    assert len(l2.calls) == 2


async def test_generation_retries_invalid_final_submission_without_tools():
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
            L2Completion(tool_calls=[tool_call("submit-bad", "submit_final_answer", {})]),
            L2Completion(content="일반 completion으로 복구한 답변 guideline:1"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="공식 진료지침 출처를 알려줘")]
    )

    assert answer == "일반 completion으로 복구한 답변 guideline:1"
    assert "tools" not in l2.calls[2]
    assert "tool_choice" not in l2.calls[2]


async def test_generation_handles_only_invalid_initial_tool_calls():
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[tool_call("bad", "unexpected", {})]),
            L2Completion(content="도구 없이 생성한 L2 최종 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "도구 없이 생성한 L2 최종 답변"


async def test_generation_does_not_expose_textual_protocol_from_initial_step():
    l2 = ScriptedL2(
        [
            L2Completion(
                content=("<tool_call>retrieve_relevant_content<arg_key>query</arg_key></tool_call>")
            ),
            L2Completion(
                tool_calls=[
                    tool_call(
                        "submit-1",
                        "submit_final_answer",
                        {"answer": "근거 검색 없이 생성한 L2 답변"},
                    )
                ]
            ),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "근거 검색 없이 생성한 L2 답변"
    assert len(l2.calls) == 2


async def test_generation_executes_only_first_valid_retrieval():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call("bad", "unknown_tool", {}),
                    tool_call("one", "retrieve_relevant_content", {"query": "query one"}),
                    tool_call("two", "retrieve_relevant_content", {"query": "query two"}),
                ]
            ),
            L2Completion(content="최종 guideline:1"),
        ]
    )
    retrieval = FakeRetrieval()

    assert (
        await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="질문")])
        == "최종 guideline:1"
    )
    assert retrieval.queries == ["query one"]
    tool_messages = [message for message in l2.calls[1]["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "bad",
        "one",
        "two",
    ]


async def test_direct_answer_uses_medical_prompt_but_no_tools():
    l2 = ScriptedL2([L2Completion(content="안전한 직접 답변")])
    engine = GenerationEngine(l2, FakeRetrieval())

    answer = await engine.direct_answer([ChatMessage(role="user", content="질문")])

    assert answer == "안전한 직접 답변"
    assert l2.calls[0]["messages"][0]["role"] == "system"
    assert "no more than 300 words" in l2.calls[0]["messages"][0]["content"]
    assert "No retrieval or other tools are available" in l2.calls[0]["messages"][0]["content"]
    assert "call retrieve_relevant_content" not in l2.calls[0]["messages"][0]["content"]
    assert l2.calls[0]["messages"][-1]["content"] == "질문"
    assert "tools" not in l2.calls[0]


async def test_direct_answer_rewrites_textual_tool_protocol_without_tools():
    l2 = ScriptedL2(
        [
            L2Completion(
                content="<tool_call>retrieve_relevant_content</tool_call>",
            ),
            L2Completion(content="도구 프로토콜을 제거한 최종 답변"),
        ]
    )

    answer = await GenerationEngine(l2, FakeRetrieval()).direct_answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "도구 프로토콜을 제거한 최종 답변"
    assert len(l2.calls) == 2
    assert all("tools" not in call for call in l2.calls)
    assert all("tool_choice" not in call for call in l2.calls)


async def test_direct_answer_recovers_structured_tool_call_without_executing_it():
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    tool_call(
                        "hallucinated-1",
                        "retrieve_relevant_content",
                        {"query": "사용하면 안 되는 검색"},
                    )
                ]
            ),
            L2Completion(content="도구 없이 복구한 최종 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).direct_answer(
        [ChatMessage(role="user", content="질문")]
    )

    assert answer == "도구 없이 복구한 최종 답변"
    assert retrieval.queries == []
    assert len(l2.calls) == 2
    assert all("tools" not in call for call in l2.calls)
    assert all("tool_choice" not in call for call in l2.calls)
