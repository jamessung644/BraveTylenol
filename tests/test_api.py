import logging
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import UpstreamResponseError
from lunit_hackathon.schemas import L2Completion, TokenUsage


class FakeOrchestrator:
    def __init__(self, answer="L2 최종 답변"):
        self.answer_text = answer
        self.calls = []
        self.last_usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        )

    async def answer(self, messages):
        self.calls.append(messages)
        return self.answer_text


def with_key(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    return Settings(_env_file=None)


async def test_health_and_models_work_without_key(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    app = create_app(Settings(_env_file=None))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        healthz = await client.get("/healthz")
        models = await client.get("/v1/models")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert healthz.status_code == 200
    assert healthz.json() == {"status": "ok"}
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "team-chatbot"


async def test_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    orchestrator = FakeOrchestrator()
    app = create_app(Settings(_env_file=None), orchestrator)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "질문"}],
            },
        )

    assert response.status_code == 503
    assert orchestrator.calls == []


async def test_chat_accepts_missing_model_and_forwards_evaluator_bearer_key(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)

    class RecordingL2:
        received_api_key = None
        calls = []

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).received_api_key = settings.api_key
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            type(self).calls.append(kwargs)
            assert kwargs["messages"][0]["role"] == "system"
            assert "tools" not in kwargs
            return L2Completion(content="Bearer 인증 L2 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    app = create_app(Settings(_env_file=None))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer evaluator-secret"},
            json={"messages": [{"role": "user", "content": "질문"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Bearer 인증 L2 답변"
    assert response.json()["model"] == "team-chatbot"
    assert RecordingL2.received_api_key == "evaluator-secret"
    assert len(RecordingL2.calls) == 1


async def test_chat_preserves_multi_turn_history_after_direct_system_prompt(monkeypatch):
    monkeypatch.setenv("LUNIT_MCP_URL", "https://mcp.injected-by-pipeline.test")

    class RecordingL2:
        calls = []

        def __init__(self, settings, *, http_client=None):
            del settings, http_client
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            type(self).calls.append(kwargs)
            return L2Completion(content="문맥을 반영한 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    history = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
        {"role": "user", "content": "그럼 지금은요?"},
    ]
    app = create_app(with_key(monkeypatch))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": history},
        )

    assert response.status_code == 200
    assert len(RecordingL2.calls) == 1
    upstream_messages = RecordingL2.calls[0]["messages"]
    assert upstream_messages[0]["role"] == "system"
    assert upstream_messages[1:] == history


@pytest.mark.parametrize(
    ("requested_max_tokens", "expected_max_tokens"),
    [(700, 700), (5_000, 5_000), (7_000, 6_144)],
)
async def test_chat_applies_requested_max_tokens_with_server_cap(
    monkeypatch,
    requested_max_tokens,
    expected_max_tokens,
):
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)

    class RecordingL2:
        configured_max_tokens = None

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).configured_max_tokens = settings.max_completion_tokens
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            return L2Completion(content="토큰 제한 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    app = create_app(with_key(monkeypatch))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "질문"}],
                "max_tokens": requested_max_tokens,
            },
        )

    assert response.status_code == 200
    assert RecordingL2.configured_max_tokens == expected_max_tokens


async def test_chat_prefers_evaluator_bearer_over_environment_key(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "stale-deployment-key")

    class RecordingL2:
        received_api_key = None

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).received_api_key = settings.api_key
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            return L2Completion(content="요청 키 사용 성공")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    app = create_app(Settings(_env_file=None))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer evaluator-secret"},
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "질문"}]},
        )

    assert response.status_code == 200
    assert RecordingL2.received_api_key == "evaluator-secret"


async def test_chat_rejects_non_bearer_authorization(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    app = create_app(Settings(_env_file=None), FakeOrchestrator())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Basic invalid"},
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "질문"}]},
        )

    assert response.status_code == 503


async def test_chat_returns_openai_compatible_l2_completion(monkeypatch):
    orchestrator = FakeOrchestrator()
    app = create_app(with_key(monkeypatch), orchestrator)
    before = int(time.time())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "질문"}],
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["id"].startswith("chatcmpl-")
    assert before <= payload["created"] <= int(time.time())
    assert payload["model"] == "team-chatbot"
    assert payload["choices"][0]["message"]["content"] == "L2 최종 답변"
    assert payload["usage"]["total_tokens"] == 14
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("internal_finish_reason", "public_finish_reason"),
    [("length", "length"), ("tool_calls", "stop"), ("function_call", "stop")],
)
async def test_chat_exposes_consistent_finish_reason(
    monkeypatch,
    internal_finish_reason,
    public_finish_reason,
):
    class FinishingOrchestrator(FakeOrchestrator):
        last_finish_reason = internal_finish_reason

    app = create_app(with_key(monkeypatch), FinishingOrchestrator("잘린 답변"))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "질문"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == public_finish_reason


async def test_chat_rejects_streaming(monkeypatch):
    orchestrator = FakeOrchestrator()
    app = create_app(with_key(monkeypatch), orchestrator)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "질문"}],
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert orchestrator.calls == []


async def test_chat_rejects_empty_messages(monkeypatch):
    app = create_app(with_key(monkeypatch), FakeOrchestrator())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": []},
        )

    assert response.status_code == 422


async def test_upstream_failure_is_sanitized(monkeypatch):
    class FailingOrchestrator(FakeOrchestrator):
        async def answer(self, messages):
            del messages
            raise UpstreamResponseError("private upstream detail")

    app = create_app(with_key(monkeypatch), FailingOrchestrator())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "질문"}],
            },
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "L2 upstream request failed"}
    assert "private upstream detail" not in response.text


async def test_request_logs_exclude_medical_text_and_credentials(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    app = create_app(with_key(monkeypatch), FakeOrchestrator())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [
                    {
                        "role": "user",
                        "content": "PRIVATE-MEDICAL-QUESTION",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert "request_complete" in caplog.text
    assert "PRIVATE-MEDICAL-QUESTION" not in caplog.text
    assert "test-key" not in caplog.text
