import logging
import time

from httpx import ASGITransport, AsyncClient

from app import create_app
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import UpstreamResponseError
from lunit_hackathon.schemas import TokenUsage


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
        models = await client.get("/v1/models")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
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
