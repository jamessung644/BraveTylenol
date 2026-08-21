import pytest

from lunit_hackathon.errors import RetrievalError, UpstreamTimeoutError
from lunit_hackathon.orchestrator import ChatOrchestrator
from lunit_hackathon.schemas import ChatMessage, L2Completion, TokenUsage


class FakeL2:
    def __init__(self):
        self.calls = []
        self.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return L2Completion(content="L2 passthrough", usage=self.last_usage)


class UnusedGeneration:
    async def answer(self, messages):
        del messages
        raise AssertionError("generation must not run")

    async def direct_answer(self, messages):
        del messages
        raise AssertionError("generation must not run")


async def test_passthrough_sends_original_messages_directly_to_l2():
    l2 = FakeL2()
    messages = [
        ChatMessage(role="user", content="첫 질문"),
        ChatMessage(role="assistant", content="이전 답"),
        ChatMessage(role="user", content="후속 질문"),
    ]
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=UnusedGeneration(),
        mode="passthrough",
    )

    answer = await orchestrator.answer(messages)

    assert answer == "L2 passthrough"
    assert l2.calls[0]["messages"] == messages
    assert orchestrator.last_usage.total_tokens == 8


async def test_direct_mode_uses_one_medically_prompted_l2_generation():
    class DirectGeneration:
        def __init__(self):
            self.calls = []

        async def direct_answer(self, messages):
            self.calls.append(messages)
            return "L2 의료 최종 답변"

        async def answer(self, messages):
            del messages
            raise AssertionError("retrieval path must not run without MCP")

    l2 = FakeL2()
    generation = DirectGeneration()
    messages = [ChatMessage(role="user", content="질문")]
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=generation,
        mode="direct",
    )

    assert await orchestrator.answer(messages) == "L2 의료 최종 답변"
    assert generation.calls == [messages]
    assert l2.calls == []
    assert orchestrator.last_usage.total_tokens == 8


async def test_retrieval_failure_falls_back_to_medical_l2_generation():
    class FailingGeneration:
        def __init__(self):
            self.direct_calls = []

        async def answer(self, messages):
            del messages
            raise RetrievalError("private detail")

        async def direct_answer(self, messages):
            self.direct_calls.append(messages)
            return "L2 안전 폴백"

    l2 = FakeL2()
    generation = FailingGeneration()
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=generation,
        mode="rag",
    )
    messages = [ChatMessage(role="user", content="질문")]

    assert await orchestrator.answer(messages) == "L2 안전 폴백"
    assert generation.direct_calls == [messages]


async def test_l2_failure_is_not_hidden_as_retrieval_fallback():
    class TimeoutGeneration:
        async def answer(self, messages):
            del messages
            raise UpstreamTimeoutError("timeout")

        async def direct_answer(self, messages):
            del messages
            raise AssertionError("must not fall back on L2 failures")

    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=TimeoutGeneration(),
        mode="rag",
    )

    with pytest.raises(UpstreamTimeoutError):
        await orchestrator.answer([ChatMessage(role="user", content="질문")])
