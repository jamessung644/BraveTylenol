import asyncio

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.orchestrator import MEDICAL_SAFETY_FALLBACK, ChatOrchestrator
from lunit_hackathon.schemas import (
    ChatMessage,
    EvidenceItem,
    MedicalDomain,
    RetrievalResult,
    RouteDecision,
    TokenUsage,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


def _deadline(seconds: float = 60.0) -> RequestDeadline:
    return RequestDeadline.start(total_seconds=seconds)


def _route(*, verify: bool = True, retrieval: bool = True) -> RouteDecision:
    return RouteDecision(
        domains=frozenset({MedicalDomain.DRUG_SAFETY}),
        tool_names=("adr_retrieve_drug_info",),
        retrieval_required=retrieval,
        verification_required=verify,
    )


def _evidence(status: str = "sufficient") -> RetrievalResult:
    return RetrievalResult(
        status=status,
        items=[
            EvidenceItem(
                cite_uid="mfds:acetaminophen",
                source_tool="adr_retrieve_drug_info",
                relevance_score=1.0,
                content="공식 의약품 안전성 정보",
            )
        ],
    )


class FakeL2:
    def __init__(self) -> None:
        self.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        self.last_finish_reason = "length"


class RecordingRetrieval:
    def __init__(self, outcome: RetrievalResult | BaseException) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, RouteDecision, RequestDeadline]] = []

    async def retrieve(
        self,
        query: str,
        route: RouteDecision,
        deadline: RequestDeadline,
    ) -> RetrievalResult:
        self.calls.append((query, route, deadline))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class RecordingGeneration:
    def __init__(
        self,
        *,
        grounded: str | BaseException = "근거 기반 후보 [mfds:acetaminophen]",
        direct: str | BaseException = "직접 L2 답변",
        events: list[str] | None = None,
    ) -> None:
        self.grounded = grounded
        self.direct = direct
        self.events = events if events is not None else []
        self.grounded_calls: list[tuple[object, RetrievalResult, RequestDeadline]] = []
        self.direct_calls: list[tuple[object, RequestDeadline]] = []

    async def grounded_answer(self, messages, retrieval, deadline):
        self.events.append("grounded")
        self.grounded_calls.append((messages, retrieval, deadline))
        if isinstance(self.grounded, BaseException):
            raise self.grounded
        return self.grounded

    async def direct_answer(self, messages, deadline):
        self.events.append("direct")
        self.direct_calls.append((messages, deadline))
        if isinstance(self.direct, BaseException):
            raise self.direct
        return self.direct


class RecordingVerifier:
    def __init__(
        self,
        outcome: str | BaseException = "검증 후 답변 [mfds:acetaminophen]",
        events: list[str] | None = None,
    ) -> None:
        self.outcome = outcome
        self.events = events if events is not None else []
        self.calls: list[tuple[object, str, RetrievalResult, RequestDeadline]] = []

    async def verify(self, messages, candidate, retrieval, deadline):
        self.events.append("verify")
        self.calls.append((messages, candidate, retrieval, deadline))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _orchestrator(
    *,
    retrieval: RecordingRetrieval | None,
    generation: RecordingGeneration,
    verifier: RecordingVerifier | None,
    mode: str = "rag",
) -> ChatOrchestrator:
    return ChatOrchestrator(
        l2_client=FakeL2(),
        retrieval_engine=retrieval,
        generation_engine=generation,
        answer_verifier=verifier,
        settings=_settings(),
        mode=mode,
    )


async def test_drug_safety_pipeline_routes_retrieves_generates_and_verifies_original_history(
    monkeypatch,
):
    events: list[str] = []
    retrieval = RecordingRetrieval(_evidence())
    generation = RecordingGeneration(events=events)
    verifier = RecordingVerifier(events=events)
    messages = [
        ChatMessage(role="user", content="아세트아미노펜을 복용했습니다."),
        ChatMessage(role="assistant", content="복용량을 확인해 보겠습니다."),
        ChatMessage(role="user", content="발진이 생겼는데 부작용인가요?"),
    ]
    route = _route()
    deadline = _deadline()
    monkeypatch.setattr("lunit_hackathon.orchestrator.route_messages", lambda value: route)

    answer = await _orchestrator(
        retrieval=retrieval,
        generation=generation,
        verifier=verifier,
    ).answer(messages, deadline)

    assert answer == "검증 후 답변 [mfds:acetaminophen]"
    assert events == ["grounded", "verify"]
    query, received_route, received_deadline = retrieval.calls[0]
    assert "아세트아미노펜" in query
    assert "발진이 생겼는데 부작용인가요?" in query
    assert received_route is route
    assert received_deadline is deadline
    assert generation.grounded_calls[0][0] is messages
    assert verifier.calls[0][0] is messages
    assert generation.grounded_calls[0][1] == verifier.calls[0][2] == _evidence()


async def test_retrieval_error_uses_one_direct_l2_attempt_with_original_history(monkeypatch):
    retrieval = RecordingRetrieval(RetrievalError("retrieval unavailable", code="mcp_failed"))
    generation = RecordingGeneration()
    messages = [ChatMessage(role="user", content="약물 상호작용을 알려 주세요.")]
    deadline = _deadline()
    monkeypatch.setattr("lunit_hackathon.orchestrator.route_messages", lambda value: _route())

    answer = await _orchestrator(
        retrieval=retrieval,
        generation=generation,
        verifier=RecordingVerifier(),
    ).answer(messages, deadline)

    assert answer == "직접 L2 답변"
    assert generation.grounded_calls == []
    assert generation.direct_calls == [(messages, deadline)]


async def test_partial_retrieval_evidence_still_uses_grounded_generation(monkeypatch):
    retrieval = RecordingRetrieval(_evidence(status="partial"))
    generation = RecordingGeneration()
    monkeypatch.setattr(
        "lunit_hackathon.orchestrator.route_messages", lambda value: _route(verify=False)
    )

    answer = await _orchestrator(
        retrieval=retrieval,
        generation=generation,
        verifier=None,
    ).answer([ChatMessage(role="user", content="약물 부작용?")], _deadline())

    assert answer == "근거 기반 후보 [mfds:acetaminophen]"
    assert len(generation.grounded_calls) == 1
    assert generation.direct_calls == []


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(20.0, "직접 L2 답변"), (9.0, MEDICAL_SAFETY_FALLBACK)],
)
async def test_grounded_timeout_uses_direct_recovery_only_when_budget_remains(
    monkeypatch,
    seconds,
    expected,
):
    generation = RecordingGeneration(grounded=UpstreamTimeoutError("timed out"))
    monkeypatch.setattr(
        "lunit_hackathon.orchestrator.route_messages", lambda value: _route(verify=False)
    )

    answer = await _orchestrator(
        retrieval=RecordingRetrieval(_evidence()),
        generation=generation,
        verifier=None,
    ).answer([ChatMessage(role="user", content="약물 부작용?")], _deadline(seconds))

    assert answer == expected
    assert len(generation.grounded_calls) == 1
    assert len(generation.direct_calls) == int(seconds >= 10.0)


async def test_verifier_expected_failure_keeps_grounded_candidate(monkeypatch):
    candidate = "근거 기반 후보 [mfds:acetaminophen]"
    generation = RecordingGeneration(grounded=candidate)
    verifier = RecordingVerifier(UpstreamTransportError("unavailable"))
    monkeypatch.setattr("lunit_hackathon.orchestrator.route_messages", lambda value: _route())

    answer = await _orchestrator(
        retrieval=RecordingRetrieval(_evidence()),
        generation=generation,
        verifier=verifier,
    ).answer([ChatMessage(role="user", content="약물 부작용?")], _deadline())

    assert answer == candidate
    assert len(verifier.calls) == 1


async def test_verification_is_skipped_when_deadline_has_less_than_minimum_budget(monkeypatch):
    generation = RecordingGeneration()
    verifier = RecordingVerifier()
    monkeypatch.setattr("lunit_hackathon.orchestrator.route_messages", lambda value: _route())

    answer = await _orchestrator(
        retrieval=RecordingRetrieval(_evidence()),
        generation=generation,
        verifier=verifier,
    ).answer([ChatMessage(role="user", content="약물 부작용?")], _deadline(24.0))

    assert answer == "근거 기반 후보 [mfds:acetaminophen]"
    assert verifier.calls == []


@pytest.mark.parametrize(
    "outcome",
    [
        ConfigurationError("missing configuration"),
        UpstreamTransportError("transport"),
        UpstreamResponseError("status"),
        MalformedUpstreamResponseError("malformed"),
        UpstreamTimeoutError("timeout"),
        TimeoutError(),
        "   ",
    ],
)
async def test_direct_expected_failures_and_blank_content_use_static_safety_fallback(
    outcome,
):
    generation = RecordingGeneration(direct=outcome)

    answer = await _orchestrator(
        retrieval=None,
        generation=generation,
        verifier=None,
        mode="direct",
    ).answer([ChatMessage(role="user", content="질문")], _deadline())

    assert answer == MEDICAL_SAFETY_FALLBACK
    assert len(generation.direct_calls) == 1


@pytest.mark.parametrize("failure", [ValueError("bug"), asyncio.CancelledError()])
async def test_programmer_errors_and_cancellation_propagate(failure):
    generation = RecordingGeneration(direct=failure)
    orchestrator = _orchestrator(
        retrieval=None,
        generation=generation,
        verifier=None,
        mode="direct",
    )

    with pytest.raises(type(failure)):
        await orchestrator.answer([ChatMessage(role="user", content="질문")], _deadline())


async def test_direct_route_and_public_state_are_request_local(monkeypatch):
    l2 = FakeL2()
    l2.last_usage = TokenUsage(prompt_tokens=-3, completion_tokens=2, total_tokens=-1)
    generation = RecordingGeneration()
    monkeypatch.setattr(
        "lunit_hackathon.orchestrator.route_messages",
        lambda value: _route(retrieval=False, verify=False),
    )
    orchestrator = ChatOrchestrator(
        l2_client=l2,
        retrieval_engine=None,
        generation_engine=generation,
        answer_verifier=None,
        settings=_settings(),
        mode="rag",
    )
    deadline = _deadline()
    messages = [ChatMessage(role="user", content="일반 건강 질문")]

    assert await orchestrator.answer(messages, deadline) == "직접 L2 답변"
    assert generation.direct_calls == [(messages, deadline)]
    assert orchestrator.last_usage.model_dump() == {
        "prompt_tokens": 0,
        "completion_tokens": 2,
        "total_tokens": 2,
    }
    assert orchestrator.last_finish_reason == "stop"
