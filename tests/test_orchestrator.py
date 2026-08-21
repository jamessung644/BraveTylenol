import asyncio
import logging

import pytest

from harness.errors import RetrievalError, UpstreamTimeoutError
from harness.orchestrator import ChatOrchestrator
from harness.schemas import ChatMessage, L2Completion


class RecordingL2:
    def __init__(self, responses: list[L2Completion | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    async def complete(self, **kwargs: object) -> L2Completion:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingGeneration:
    def __init__(self, answer: str | Exception) -> None:
        self._answer = answer
        self.calls: list[list[ChatMessage]] = []

    async def answer(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


async def test_passthrough_sends_original_conversation_without_tools():
    messages = [ChatMessage(role="user", content="질문")]
    l2 = RecordingL2([L2Completion(content="직접 답변")])
    generation = RecordingGeneration(AssertionError("generation must not be called"))

    answer = await ChatOrchestrator(l2=l2, generation=generation, mode="passthrough").answer(messages)

    assert answer == "직접 답변"
    assert l2.calls == [{"messages": messages}]
    assert generation.calls == []


async def test_rag_uses_generation_engine():
    messages = [ChatMessage(role="user", content="질문")]
    l2 = RecordingL2([])
    generation = RecordingGeneration("근거 기반 답변")

    answer = await ChatOrchestrator(l2=l2, generation=generation, mode="rag").answer(messages)

    assert answer == "근거 기반 답변"
    assert generation.calls == [messages]
    assert l2.calls == []


async def test_rag_retrieval_error_retries_original_conversation_directly():
    messages = [ChatMessage(role="user", content="원문 질문")]
    l2 = RecordingL2([L2Completion(content="폴백 답변")])
    generation = RecordingGeneration(RetrievalError("MCP unavailable"))

    answer = await ChatOrchestrator(l2=l2, generation=generation, mode="rag").answer(messages)

    assert answer == "폴백 답변"
    assert generation.calls == [messages]
    assert l2.calls == [{"messages": messages}]


async def test_retrieval_fallback_log_does_not_include_request_or_error_contents(caplog):
    messages = [ChatMessage(role="user", content="private question")]
    l2 = RecordingL2([L2Completion(content="폴백 답변")])
    generation = RecordingGeneration(RetrievalError("secret evidence and api-key"))

    with caplog.at_level(logging.WARNING):
        await ChatOrchestrator(l2=l2, generation=generation, mode="rag").answer(messages)

    assert "private question" not in caplog.text
    assert "secret evidence" not in caplog.text
    assert "api-key" not in caplog.text


async def test_l2_timeout_is_not_a_retrieval_fallback():
    messages = [ChatMessage(role="user", content="질문")]
    l2 = RecordingL2([UpstreamTimeoutError("timeout")])
    generation = RecordingGeneration(RetrievalError("retrieval failed"))

    with pytest.raises(UpstreamTimeoutError):
        await ChatOrchestrator(l2=l2, generation=generation, mode="rag").answer(messages)

    assert l2.calls == [{"messages": messages}]


async def test_concurrent_requests_do_not_share_state():
    entered = 0
    both_entered = asyncio.Event()

    def orchestrator_factory(question: str) -> ChatOrchestrator:
        class BlockingGeneration:
            async def answer(self, messages: list[ChatMessage]) -> str:
                nonlocal entered
                entered += 1
                if entered == 2:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=1)
                return f"답변: {messages[0].content}"

        return ChatOrchestrator(l2=RecordingL2([]), generation=BlockingGeneration(), mode="rag")

    async def worker(question: str) -> str:
        return await orchestrator_factory(question).answer([ChatMessage(role="user", content=question)])

    answers = await asyncio.gather(worker("질문 A"), worker("질문 B"))

    assert answers == ["답변: 질문 A", "답변: 질문 B"]
