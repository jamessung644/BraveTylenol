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
            L2Completion(content="근거를 반영한 L2 최종 답변"),
        ]
    )
    retrieval = FakeRetrieval()

    answer = await GenerationEngine(l2, retrieval).answer(
        [ChatMessage(role="user", content="목표 혈압은?")]
    )

    assert answer == "근거를 반영한 L2 최종 답변"
    assert retrieval.queries == ["한국 고혈압 목표 혈압 진료지침"]
    tool_message = l2.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "retrieve-1"
    assert "guideline:1" in tool_message["content"]
    assert "tools" not in l2.calls[1]


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
            L2Completion(content="최종"),
        ]
    )
    retrieval = FakeRetrieval()

    assert (
        await GenerationEngine(l2, retrieval).answer([ChatMessage(role="user", content="질문")])
        == "최종"
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
    assert "HealthBench" in l2.calls[0]["messages"][0]["content"]
    assert l2.calls[0]["messages"][-1]["content"] == "질문"
    assert l2.calls[0]["max_tokens"] == 1024
    assert "tools" not in l2.calls[0]


async def test_direct_answer_uses_exactly_one_l2_call_for_emergency():
    l2 = ScriptedL2([L2Completion(content="Call emergency services now.")])
    engine = GenerationEngine(l2, FakeRetrieval())

    answer = await engine.direct_answer(
        [ChatMessage(role="user", content="They are unconscious and not breathing normally")]
    )

    assert answer == "Call emergency services now."
    assert len(l2.calls) == 1
    assert l2.calls[0]["max_tokens"] == 1536
    assert "urgent action first" in l2.calls[0]["messages"][0]["content"]
