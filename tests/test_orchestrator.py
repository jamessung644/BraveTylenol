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
    assert "sole author of the final user-facing medical answer" in sent_messages[0][
        "content"
    ]
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
    assert "possibly time-critical health risk" in system_prompt
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
    assert "possibly time-critical health risk" in l2.calls[0]["messages"][0][
        "content"
    ]


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


async def test_c16_admits_only_four_rag_without_queueing_direct_or_no_evidence(
    monkeypatch,
):
    monkeypatch.delenv("MAX_CONCURRENT_RAG_REQUESTS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.max_concurrent_rag_requests == 4

    class QueueDetectingSemaphore(asyncio.Semaphore):
        def __init__(self, value):
            super().__init__(value)
            self.acquire_while_locked = 0

        async def acquire(self):
            if self.locked():
                self.acquire_while_locked += 1
            return await super().acquire()

    class ConcurrentGeneration(HybridGeneration):
        def __init__(self):
            super().__init__()
            self.active_rag = 0
            self.max_active_rag = 0
            self.four_rag_admitted = asyncio.Event()
            self.twelve_spillovers_finished = asyncio.Event()
            self.release_rag = asyncio.Event()

        def _record_spillover(self):
            if len(self.direct_calls) + len(self.evidence_unavailable_calls) == 12:
                self.twelve_spillovers_finished.set()

        async def direct_answer(self, messages):
            answer = await super().direct_answer(messages)
            self._record_spillover()
            return answer

        async def evidence_unavailable_answer(self, messages):
            answer = await super().evidence_unavailable_answer(messages)
            self._record_spillover()
            return answer

        async def answer(self, messages):
            self.rag_calls.append(messages)
            self.active_rag += 1
            self.max_active_rag = max(self.max_active_rag, self.active_rag)
            if len(self.rag_calls) == settings.max_concurrent_rag_requests:
                self.four_rag_admitted.set()
            try:
                await self.release_rag.wait()
            finally:
                self.active_rag -= 1
            return "RAG L2 answer"

    generation = ConcurrentGeneration()
    semaphore = QueueDetectingSemaphore(settings.max_concurrent_rag_requests)

    def request(messages):
        return ChatOrchestrator(
            l2_client=FakeL2(),
            generation_engine=generation,
            mode="hybrid",
            rag_semaphore=semaphore,
        ).answer(messages)

    source_messages = [ChatMessage(role="user", content="KCD I10 공식 명칭")]
    direct_messages = [ChatMessage(role="user", content="건강한 수면 습관을 알려줘")]
    admitted = [
        asyncio.create_task(request(source_messages))
        for _ in range(settings.max_concurrent_rag_requests)
    ]
    spillovers = []

    try:
        await asyncio.wait_for(generation.four_rag_admitted.wait(), timeout=1)

        # Keep four slow, multi-call-capable RAG trajectories blocked while
        # twelve more requests make this a deterministic C16 cohort.
        spillovers = [asyncio.create_task(request(source_messages)) for _ in range(4)]
        spillovers += [asyncio.create_task(request(direct_messages)) for _ in range(8)]
        await asyncio.wait_for(
            generation.twelve_spillovers_finished.wait(),
            timeout=1,
        )
        spillover_answers = await asyncio.gather(*spillovers)

        assert generation.max_active_rag == 4
        assert len(generation.rag_calls) == 4
        assert len(generation.evidence_unavailable_calls) == 4
        assert len(generation.direct_calls) == 8
        assert semaphore.acquire_while_locked == 0
        assert spillover_answers.count("guarded no-evidence L2 answer") == 4
        assert spillover_answers.count("direct L2 answer") == 8
        assert all(not task.done() for task in admitted)
    finally:
        generation.release_rag.set()
        admitted_answers = await asyncio.gather(*admitted)
        if spillovers:
            await asyncio.gather(*spillovers)

    assert admitted_answers == ["RAG L2 answer"] * 4
    assert len(generation.rag_calls) == 4


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
