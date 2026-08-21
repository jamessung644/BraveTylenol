import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from app import create_app
from harness.errors import UpstreamResponseError, UpstreamTimeoutError
from harness.schemas import L2Completion, TokenUsage


async def test_imported_app_is_ready_without_key_or_upstream_calls(monkeypatch):
    def upstream_must_not_be_constructed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("readiness must not construct an L2 or MCP client")

    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "L2Client", upstream_must_not_be_constructed)
    monkeypatch.setattr(app_module, "MCPClient", upstream_must_not_be_constructed)

    assert app_module.app is not None
    async with AsyncClient(transport=ASGITransport(app=app_module.app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "team-chatbot"


async def test_models_available_without_api_key(settings_without_key):
    app = create_app(settings=settings_without_key)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "team-chatbot", "object": "model", "owned_by": "brave-tylenol"}],
    }


async def test_streaming_is_rejected(settings_with_key, fake_orchestrator):
    app = create_app(settings=settings_with_key, orchestrator=fake_orchestrator)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}], "stream": True},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Streaming is not supported"
    assert fake_orchestrator.calls == []


async def test_chat_requires_configured_key(settings_without_key, fake_orchestrator):
    app = create_app(settings=settings_without_key, orchestrator=fake_orchestrator)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "LUNIT_FM_API_KEY is not configured"
    assert fake_orchestrator.calls == []


async def test_chat_forwards_request_bearer_token_to_l2_when_environment_key_is_missing(
    settings_without_key, monkeypatch
):
    class BearerAuthenticatedL2:
        def __init__(self, settings, *, http_client=None):
            del http_client
            if settings.api_key != "lunit_request_key":
                raise AssertionError("request bearer token was not forwarded to L2")
            self.last_usage = TokenUsage()

        async def complete(self, *, messages, tools=None, tool_choice=None):
            del messages, tools, tool_choice
            return L2Completion(content="인증된 L2 응답")

    monkeypatch.setattr(app_module, "L2Client", BearerAuthenticatedL2)
    app = create_app(settings=settings_without_key)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer lunit_request_key"},
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "인증된 L2 응답"


async def test_chat_returns_openai_compatible_completion(settings_with_key, fake_orchestrator):
    app = create_app(settings=settings_with_key, orchestrator=fake_orchestrator)
    before = int(time.time())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["id"].startswith("chatcmpl-")
    assert before <= payload["created"] <= int(time.time())
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "team-chatbot"
    assert payload["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "반갑습니다."}, "finish_reason": "stop"}
    ]
    assert payload["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
async def test_chat_accepts_all_supported_message_roles(settings_with_key, fake_orchestrator, role):
    app = create_app(settings=settings_with_key, orchestrator=fake_orchestrator)
    message = {"role": role, "content": "context"}
    if role == "tool":
        message["tool_call_id"] = "call-1"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "team-chatbot", "messages": [message]})

    assert response.status_code == 200
    assert fake_orchestrator.calls[0][0].role == role


async def test_chat_rejects_empty_messages(settings_with_key, fake_orchestrator):
    app = create_app(settings=settings_with_key, orchestrator=fake_orchestrator)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "team-chatbot", "messages": []})

    assert response.status_code == 422
    assert fake_orchestrator.calls == []


async def test_chat_ignores_optional_openai_fields(settings_with_key, fake_orchestrator):
    app = create_app(settings=settings_with_key, orchestrator=fake_orchestrator)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "team-chatbot",
                "messages": [{"role": "user", "content": "안녕"}],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 50,
                "response_format": {"type": "text"},
            },
        )

    assert response.status_code == 200
    assert fake_orchestrator.calls[0][0].content == "안녕"


class _FailingOrchestrator:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def answer(self, messages):
        del messages
        raise self._error


async def test_l2_timeout_maps_to_gateway_timeout(settings_with_key):
    app = create_app(settings=settings_with_key, orchestrator=_FailingOrchestrator(UpstreamTimeoutError("secret")))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]})

    assert response.status_code == 504
    assert response.json() == {"detail": "L2 upstream timed out"}


async def test_request_deadline_cancels_orchestration_and_maps_to_gateway_timeout(settings_with_key):
    class SlowOrchestrator:
        def __init__(self):
            self.cancelled = False

        async def answer(self, messages):
            del messages
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    slow = SlowOrchestrator()
    app = create_app(settings=settings_with_key.model_copy(update={"upstream_timeout_seconds": 0.01}), orchestrator=slow)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]})

    assert response.status_code == 504
    assert response.json() == {"detail": "L2 upstream timed out"}
    assert slow.cancelled


async def test_l2_failure_maps_to_bad_gateway(settings_with_key):
    app = create_app(settings=settings_with_key, orchestrator=_FailingOrchestrator(UpstreamResponseError("secret")))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "team-chatbot", "messages": [{"role": "user", "content": "안녕"}]})

    assert response.status_code == 502
    assert response.json() == {"detail": "L2 upstream request failed"}


async def test_default_passthrough_does_not_construct_mcp_and_copies_final_usage(settings_with_key, monkeypatch):
    class DirectL2:
        def __init__(self, settings, *, http_client=None):
            del settings, http_client
            self.last_usage = TokenUsage()

        async def complete(self, *, messages, tools=None, tool_choice=None):
            assert tools is None
            assert tool_choice is None
            assert messages[0].role == "system"
            assert "no more than 300 words" in messages[0].content
            assert messages[1].content == "원문 질문"
            self.last_usage = TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
            return L2Completion(content="L2 원문 응답", usage=self.last_usage)

    def mcp_must_not_be_constructed(*args, **kwargs):
        del args, kwargs
        raise AssertionError("passthrough must not construct MCP retrieval")

    monkeypatch.setattr(app_module, "L2Client", DirectL2)
    monkeypatch.setattr(app_module, "MCPClient", mcp_must_not_be_constructed)
    settings = settings_with_key.model_copy(update={"harness_mode": "passthrough"})
    app = create_app(settings=settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "team-chatbot", "messages": [{"role": "user", "content": "원문 질문"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "L2 원문 응답"
    assert response.json()["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


async def test_lifespan_closes_shared_l2_http_client(settings_with_key):
    app = create_app(settings=settings_with_key)

    async with app.router.lifespan_context(app):
        shared_client = app.state.l2_http_client
        assert not shared_client.is_closed

    assert shared_client.is_closed
