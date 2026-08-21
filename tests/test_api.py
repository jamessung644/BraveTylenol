import time

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app


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
