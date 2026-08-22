import asyncio

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamTimeoutError,
)
from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.orchestrator import ChatOrchestrator
from lunit_hackathon.schemas import ChatMessage, L2Completion, TokenUsage


class FakeL2:
    def __init__(self, completions=None):
        self.calls = []
        self.completions = list(completions or [])
        self.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.completions:
            return self.completions.pop(0)
        return L2Completion(content="L2 passthrough", usage=self.last_usage)


class UnusedGeneration:
    async def answer(self, messages):
        del messages
        raise AssertionError("generation must not run")

    async def direct_answer(self, messages):
        del messages
        raise AssertionError("generation must not run")


async def test_passthrough_uses_the_same_direct_final_prompt_and_validator():
    l2 = FakeL2()
    messages = [
        ChatMessage(role="user", content="첫 질문"),
        ChatMessage(role="assistant", content="이전 답"),
        ChatMessage(role="user", content="후속 질문"),
    ]
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=None,
        mode="passthrough",
    )

    answer = await orchestrator.answer(messages)

    assert answer == "L2 passthrough"
    sent_messages = l2.calls[0]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "의료정보 어시스턴트" in sent_messages[0]["content"]
    assert "retrieve_relevant_content" not in sent_messages[0]["content"]
    assert sent_messages[-1]["role"] == "user"
    assert orchestrator.last_usage.total_tokens == 8


async def test_passthrough_invalid_twice_fails_the_plain_answer_invariant():
    l2 = FakeL2(
        [
            L2Completion(content="retrieve_relevant_content(query='x')"),
            L2Completion(content='<tool_call name="retry" />'),
        ]
    )
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=None,
        mode="passthrough",
    )

    with pytest.raises(MalformedUpstreamResponseError, match="bounded recovery"):
        await orchestrator.answer([ChatMessage(role="user", content="질문")])

    assert len(l2.calls) == 2
    assert "retrieve_relevant_content(query='x')" not in str(l2.calls[1]["messages"])


@pytest.mark.parametrize("mode", ["direct", "hybrid", "passthrough"])
async def test_shortcut_modes_preserve_the_emergency_final_gate(mode):
    l2 = FakeL2()
    generation = None if mode == "passthrough" else GenerationEngine(l2, None)
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=generation,
        mode=mode,
    )

    answer = await orchestrator.answer(
        [ChatMessage(role="user", content="가슴을 짓누르는 느낌과 숨참이 지금 같이 있어요")]
    )

    assert answer == "L2 passthrough"
    assert len(l2.calls) == 1
    system_prompt = l2.calls[0]["messages"][0]["content"]
    assert "시간 민감한 건강 위험" in system_prompt
    assert "retrieve_relevant_content" not in system_prompt


@pytest.mark.parametrize("mode", ["hybrid", "rag"])
async def test_rag_admission_full_direct_fallback_preserves_emergency_gate(mode):
    l2 = FakeL2()
    generation = GenerationEngine(l2, None)
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        generation_engine=generation,
        mode=mode,
        rag_semaphore=semaphore,
    )

    answer = await orchestrator.answer(
        [
            ChatMessage(
                role="user",
                content="현재 공식 진료지침이 궁금하고 가슴이 쥐어짜이면서 숨이 차요",
            )
        ]
    )

    assert answer == "L2 passthrough"
    assert len(l2.calls) == 1
    assert "시간 민감한 건강 위험" in l2.calls[0]["messages"][0]["content"]


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


async def test_retrieval_failure_does_not_start_a_new_direct_trajectory():
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

    with pytest.raises(RetrievalError):
        await orchestrator.answer(messages)
    assert generation.direct_calls == []


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


class HybridGeneration:
    def __init__(self):
        self.direct_calls = []
        self.rag_calls = []
        self.evidence_unavailable_calls = []

    async def direct_answer(self, messages):
        self.direct_calls.append(messages)
        return "direct L2 answer"

    async def answer(self, messages):
        self.rag_calls.append(messages)
        return "RAG L2 answer"

    async def evidence_unavailable_answer(self, messages):
        self.evidence_unavailable_calls.append(messages)
        return "guarded no-evidence L2 answer"


async def test_hybrid_keeps_routine_and_noisy_medical_turns_on_one_call_direct_path():
    generation = HybridGeneration()
    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=generation,
        mode="hybrid",
    )
    messages = [ChatMessage(role="user", content="머리아픔 어제부터 약 뭐먹지")]

    assert await orchestrator.answer(messages) == "direct L2 answer"
    assert generation.direct_calls == [messages]
    assert generation.rag_calls == []


async def test_hybrid_routes_source_dependent_keyword_fragments_to_rag():
    generation = HybridGeneration()
    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=generation,
        mode="hybrid",
    )
    messages = [ChatMessage(role="user", content="타이레놀 식약처 허가사항 최신")]

    assert await orchestrator.answer(messages) == "RAG L2 answer"
    assert generation.rag_calls == [messages]
    assert generation.direct_calls == []


async def test_hybrid_rag_admission_never_waits_into_the_final_answer_reserve():
    generation = HybridGeneration()
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=generation,
        mode="hybrid",
        rag_semaphore=semaphore,
    )
    messages = [ChatMessage(role="user", content="KCD I10 공식 명칭")]

    assert await orchestrator.answer(messages) == "guarded no-evidence L2 answer"
    assert generation.evidence_unavailable_calls == [messages]
    assert generation.direct_calls == []
    assert generation.rag_calls == []


async def test_default_c16_rag_admission_has_no_artificial_evidence_failure(
    monkeypatch,
):
    monkeypatch.delenv("MAX_CONCURRENT_RAG_REQUESTS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.max_concurrent_rag_requests == 16

    class ConcurrentGeneration(HybridGeneration):
        def __init__(self):
            super().__init__()
            self.all_admitted = asyncio.Event()
            self.release = asyncio.Event()

        async def answer(self, messages):
            self.rag_calls.append(messages)
            if len(self.rag_calls) == settings.max_concurrent_rag_requests:
                self.all_admitted.set()
            await self.release.wait()
            return "RAG L2 answer"

    generation = ConcurrentGeneration()
    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=generation,
        mode="hybrid",
        rag_semaphore=asyncio.Semaphore(settings.max_concurrent_rag_requests),
    )
    messages = [ChatMessage(role="user", content="KCD I10 공식 명칭")]
    tasks = [
        asyncio.create_task(orchestrator.answer(messages))
        for _ in range(settings.max_concurrent_rag_requests)
    ]

    try:
        await asyncio.wait_for(generation.all_admitted.wait(), timeout=1)
    finally:
        generation.release.set()
        answers = await asyncio.gather(*tasks)

    assert answers == ["RAG L2 answer"] * 16
    assert len(generation.rag_calls) == 16
    assert generation.evidence_unavailable_calls == []
    assert generation.direct_calls == []


async def test_forced_rag_mode_uses_the_same_nonblocking_admission_limit():
    generation = HybridGeneration()
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    orchestrator = ChatOrchestrator(
        l2_client=FakeL2(),
        generation_engine=generation,
        mode="rag",
        rag_semaphore=semaphore,
    )
    messages = [ChatMessage(role="user", content="건강한 수면 습관을 알려줘")]

    assert await orchestrator.answer(messages) == "direct L2 answer"
    assert generation.direct_calls == [messages]
    assert generation.evidence_unavailable_calls == []
    assert generation.rag_calls == []
