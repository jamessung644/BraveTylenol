import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.orchestrator import MEDICAL_SAFETY_FALLBACK
from lunit_hackathon.schemas import EvidenceItem, RetrievalResult, TokenUsage


def _settings(**updates) -> Settings:
    return Settings(_env_file=None, agent_mode="direct", **updates)


async def _post(app, payload=None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            json=payload or {"messages": [{"role": "user", "content": "질문"}]},
        )


class RaisingOrchestrator:
    last_usage = TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)

    def __init__(self, failure: BaseException) -> None:
        self.failure = failure

    async def answer(self, messages, deadline):
        del messages, deadline
        raise self.failure


async def test_credential_configuration_failure_returns_a_complete_safe_envelope(monkeypatch):
    def fail_resolution(authorization, environment_key):
        del authorization, environment_key
        raise ConfigurationError("credential resolution failed")

    monkeypatch.setattr("app.resolve_lunit_api_key", fail_resolution)

    response = await _post(create_app(_settings()))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == MEDICAL_SAFETY_FALLBACK


@pytest.mark.parametrize(
    "failure",
    [
        ConfigurationError("missing key"),
        RetrievalError("MCP unavailable", code="mcp_connection_failed"),
        UpstreamTransportError("transport"),
        UpstreamResponseError("status"),
        MalformedUpstreamResponseError("malformed"),
        UpstreamTimeoutError("timeout"),
    ],
)
async def test_expected_pipeline_failures_return_a_complete_safe_200_envelope(failure):
    response = await _post(create_app(_settings(), RaisingOrchestrator(failure)))

    payload = response.json()
    assert response.status_code == 200
    assert payload["model"] == "team-chatbot"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": MEDICAL_SAFETY_FALLBACK,
    }
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


async def test_request_wide_timeout_returns_the_safe_completion_envelope():
    class SlowOrchestrator:
        last_usage = TokenUsage()

        async def answer(self, messages, deadline):
            del messages, deadline
            await asyncio.sleep(1.0)
            return "unreachable"

    response = await _post(
        create_app(
            _settings(request_timeout_seconds=0.01),
            SlowOrchestrator(),
        )
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == MEDICAL_SAFETY_FALLBACK
    assert response.json()["choices"][0]["finish_reason"] == "stop"


async def test_real_rag_path_passes_route_and_deadline_to_three_argument_retrieval(
    monkeypatch,
):
    evidence = RetrievalResult(
        status="sufficient",
        items=[
            EvidenceItem(
                cite_uid="mfds:1",
                source_tool="adr_retrieve_drug_info",
                relevance_score=1.0,
                content="공식 의약품 정보",
            )
        ],
    )

    class FakeL2:
        def __init__(self, settings, *, http_client=None):
            del settings, http_client
            self.last_usage = TokenUsage(prompt_tokens=6, completion_tokens=4, total_tokens=10)

    class FakeMCP:
        def __init__(self, settings):
            del settings

    class RecordingRetrieval:
        instances = []

        def __init__(self, l2, mcp, settings):
            del l2, mcp, settings
            self.calls = []
            type(self).instances.append(self)

        async def retrieve(self, query, route, deadline):
            self.calls.append((query, route, deadline))
            return evidence

    class RecordingGeneration:
        instances = []

        def __init__(self, l2, settings):
            del l2, settings
            self.grounded_calls = []
            self.direct_calls = []
            type(self).instances.append(self)

        async def grounded_answer(self, messages, retrieval, deadline):
            self.grounded_calls.append((messages, retrieval, deadline))
            return "근거 기반 답변 [mfds:1]"

        async def direct_answer(self, messages, deadline):
            self.direct_calls.append((messages, deadline))
            return "직접 답변"

    class RecordingVerifier:
        instances = []

        def __init__(self, l2, settings):
            del l2, settings
            self.calls = []
            type(self).instances.append(self)

        async def verify(self, messages, candidate, retrieval, deadline):
            self.calls.append((messages, candidate, retrieval, deadline))
            return "검증된 답변 [mfds:1]"

    monkeypatch.setattr("app.L2Client", FakeL2)
    monkeypatch.setattr("app.MCPClient", FakeMCP)
    monkeypatch.setattr("app.RetrievalEngine", RecordingRetrieval)
    monkeypatch.setattr("app.GenerationEngine", RecordingGeneration)
    monkeypatch.setattr("app.AnswerVerifier", RecordingVerifier)
    app = create_app(
        Settings(
            _env_file=None,
            agent_mode="rag",
            mcp_url="https://mcp.example.test/mcp",
        )
    )
    history = [
        {"role": "user", "content": "아세트아미노펜을 복용했습니다."},
        {"role": "assistant", "content": "복용량을 확인해 보겠습니다."},
        {"role": "user", "content": "발진은 부작용인가요?"},
    ]

    response = await _post(app, {"messages": history})

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "검증된 답변 [mfds:1]"
    retrieval = RecordingRetrieval.instances[0]
    generation = RecordingGeneration.instances[0]
    verifier = RecordingVerifier.instances[0]
    query, route, deadline = retrieval.calls[0]
    assert "아세트아미노펜" in query
    assert "발진은 부작용인가요?" in query
    assert route.retrieval_required
    assert generation.grounded_calls[0][0] is verifier.calls[0][0]
    assert generation.grounded_calls[0][1] is evidence
    assert generation.grounded_calls[0][2] is deadline
    assert verifier.calls[0][2] is evidence
    assert verifier.calls[0][3] is deadline
    assert generation.direct_calls == []
