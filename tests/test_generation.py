import json
from collections.abc import Sequence
from typing import Any

import pytest

from harness.errors import MalformedUpstreamResponseError
from harness.generation import GenerationEngine
from harness.schemas import (
    ChatMessage,
    EvidenceItem,
    FunctionCall,
    L2Completion,
    RetrievalResult,
    ToolCall,
)


def tool_call(identifier: str, name: str, arguments: object) -> ToolCall:
    return ToolCall(id=identifier, function=FunctionCall(name=name, arguments=json.dumps(arguments)))


class ScriptedL2Client:
    def __init__(self, completions: list[L2Completion]) -> None:
        self.completions = completions
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: object) -> L2Completion:
        self.calls.append(kwargs)
        return self.completions.pop(0)


class FakeRetrievalEngine:
    def __init__(self, result: RetrievalResult | None = None) -> None:
        self.queries: list[str] = []
        self.result = result or RetrievalResult(status="no_evidence")

    async def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        return self.result


async def test_generation_returns_direct_l2_content_without_retrieval():
    l2 = ScriptedL2Client([L2Completion(content="  충분히 답할 수 있습니다.  ")])
    retrieval = FakeRetrievalEngine()

    answer = await GenerationEngine(l2, retrieval).answer([
        ChatMessage(role="user", content="물을 충분히 마셔야 하나요?")
    ])

    assert answer == "  충분히 답할 수 있습니다.  "
    assert retrieval.queries == []
    assert [tool["function"]["name"] for tool in l2.calls[0]["tools"]] == ["retrieve_relevant_content"]
    assert l2.calls[0]["tool_choice"] == "auto"


async def test_generation_feeds_retrieval_result_back_to_l2():
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("r1", "retrieve_relevant_content", {"query": "한국 고혈압 진료지침 목표 혈압"})]),
        L2Completion(content="근거를 반영한 최종 답변입니다."),
    ])
    retrieval = FakeRetrievalEngine(RetrievalResult(
        status="sufficient",
        items=[EvidenceItem(cite_uid="g:1", relevance_score=0.9, source_tool="index_get_page_content", content="근거")],
        note="지침 확인",
    ))

    answer = await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="목표 혈압은?")])

    assert answer == "근거를 반영한 최종 답변입니다."
    assert retrieval.queries == ["한국 고혈압 진료지침 목표 혈압"]
    assert l2.calls[1]["messages"][-1]["role"] == "tool"
    assert l2.calls[1]["messages"][-1]["tool_call_id"] == "r1"
    assert json.loads(l2.calls[1]["messages"][-1]["content"]) == {
        "items": [{"cite_uid": "g:1", "content": "근거", "relevance_score": 0.9, "source_tool": "index_get_page_content"}],
        "note": "지침 확인",
        "status": "sufficient",
    }
    assert "tools" not in l2.calls[1]


async def test_generation_prepends_prompt_and_preserves_complete_multiturn_conversation():
    l2 = ScriptedL2Client([L2Completion(content="답변")])
    messages: Sequence[ChatMessage] = [
        ChatMessage(role="system", content="evaluator system"),
        ChatMessage(role="user", content="첫 질문"),
        ChatMessage(role="assistant", content="첫 답"),
        ChatMessage(role="user", content="그럼 다음은?")
    ]

    await GenerationEngine(l2, FakeRetrievalEngine()).answer(messages)

    sent = l2.calls[0]["messages"]
    assert sent[1:] == [message.model_dump(exclude_none=True) for message in messages]


@pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "   "}, {"query": 42}, {"query": "x", "extra": "no"}])
async def test_generation_returns_protocol_error_for_invalid_retrieval_arguments(arguments: object):
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("bad", "retrieve_relevant_content", arguments)]),
        L2Completion(content="직접 답변"),
    ])
    retrieval = FakeRetrievalEngine()

    answer = await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="질문")])

    assert answer == "직접 답변"
    assert retrieval.queries == []
    assert json.loads(l2.calls[1]["messages"][-1]["content"])["error"] == "invalid retrieval request"


async def test_generation_returns_protocol_error_for_unexpected_tool_name():
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("bad", "unexpected_tool", {})]),
        L2Completion(content="직접 답변"),
    ])

    answer = await GenerationEngine(l2, FakeRetrievalEngine()).answer([ChatMessage(role="user", content="질문")])

    assert answer == "직접 답변"
    assert json.loads(l2.calls[1]["messages"][-1]["content"])["error"] == "unexpected tool"


async def test_generation_executes_only_first_valid_retrieval_and_reports_other_calls():
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[
            tool_call("bad", "unexpected_tool", {}),
            tool_call("good", "retrieve_relevant_content", {"query": "완전한 질문"}),
            tool_call("again", "retrieve_relevant_content", {"query": "다른 질문"}),
        ]),
        L2Completion(content="최종 답변"),
    ])
    retrieval = FakeRetrievalEngine()

    answer = await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="질문")])

    assert answer == "최종 답변"
    assert retrieval.queries == ["완전한 질문"]
    tool_messages = l2.calls[1]["messages"][-3:]
    assert [message["tool_call_id"] for message in tool_messages] == ["bad", "good", "again"]
    assert json.loads(tool_messages[0]["content"])["error"] == "unexpected tool"
    assert json.loads(tool_messages[2]["content"])["error"] == "retrieval already used"


async def test_generation_reuses_evidence_and_forces_text_after_second_retrieval_request():
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("first", "retrieve_relevant_content", {"query": "완전한 질문"})]),
        L2Completion(tool_calls=[tool_call("second", "retrieve_relevant_content", {"query": "또 다른 질문"})]),
        L2Completion(content="강제된 최종 답변"),
    ])
    retrieval = FakeRetrievalEngine(RetrievalResult(status="no_evidence", note="찾지 못함"))

    answer = await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="질문")])

    assert answer == "강제된 최종 답변"
    assert retrieval.queries == ["완전한 질문"]
    assert l2.calls[2]["messages"][-2]["role"] == "tool"
    assert l2.calls[2]["messages"][-2]["tool_call_id"] == "second"
    assert l2.calls[2]["messages"][-2]["content"] == l2.calls[1]["messages"][-1]["content"]
    assert l2.calls[2]["messages"][-1]["role"] == "system"
    assert "text answer" in l2.calls[2]["messages"][-1]["content"]


@pytest.mark.parametrize("completion", [L2Completion(), L2Completion(content="   "), L2Completion(tool_calls=[tool_call("again", "retrieve_relevant_content", {"query": "x"})])])
async def test_generation_rejects_blank_or_tool_only_final_content(completion: L2Completion):
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("first", "retrieve_relevant_content", {"query": "완전한 질문"})]),
        completion,
        L2Completion(content=" ") if completion.tool_calls else L2Completion(content="unused"),
    ])

    with pytest.raises(MalformedUpstreamResponseError):
        await GenerationEngine(l2, FakeRetrievalEngine()).answer([ChatMessage(role="user", content="질문")])
