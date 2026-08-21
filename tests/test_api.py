import asyncio
import logging
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.schemas import ChatMessage, L2Completion, TokenUsage


class FakeOrchestrator:
    def __init__(self, answer: str = "L2 최종 답변") -> None:
        self.answer_text = answer
        self.calls: list[tuple[list[ChatMessage], RequestDeadline]] = []
        self.last_usage = TokenUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
        self.last_finish_reason = "length"

    async def answer(self, messages, deadline) -> str:
        self.calls.append((messages, deadline))
        return self.answer_text


def _settings(**updates) -> Settings:
    return Settings(_env_file=None, agent_mode="direct", **updates)


async def _post(app, payload, *, headers=None):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.post("/v1/chat/completions", json=payload, headers=headers)


async def test_health_models_and_valid_completion_keep_evaluator_contract():
    orchestrator = FakeOrchestrator()
    app = create_app(_settings(), orchestrator)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        health = await client.get("/health")
        healthz = await client.get("/healthz")
        models = await client.get("/v1/models")
        before = int(time.time())
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "another-evaluator-model",
                "messages": [{"role": "user", "content": "질문"}],
            },
        )

    payload = response.json()
    assert health.json() == {"status": "ok"}
    assert healthz.json() == {"status": "ok"}
    assert models.json()["data"][0]["id"] == "team-chatbot"
    assert response.status_code == 200
    assert payload["id"].startswith("chatcmpl-")
    assert before <= payload["created"] <= int(time.time())
    assert payload["model"] == "team-chatbot"
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "L2 최종 답변"},
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert response.headers["x-request-id"]
    assert len(orchestrator.calls) == 1
    assert isinstance(orchestrator.calls[0][1], RequestDeadline)


async def test_injected_orchestrator_blank_answer_is_normalized_to_a_valid_completion():
    response = await _post(
        create_app(_settings(), FakeOrchestrator("   ")),
        {"messages": [{"role": "user", "content": "질문"}]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["model"] == "team-chatbot"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["message"]["content"].strip()


async def test_chat_validation_preserves_invalid_json_empty_messages_and_streaming_errors():
    app = create_app(_settings(), FakeOrchestrator())
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        invalid_json = await client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        empty_messages = await client.post(
            "/v1/chat/completions",
            json={"messages": []},
        )
        invalid_field = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "질문"}], "max_tokens": 0},
        )
        streaming = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "질문"}], "stream": True},
        )

    assert invalid_json.status_code == 422
    assert empty_messages.status_code == 422
    assert invalid_field.status_code == 422
    assert streaming.status_code == 400


async def test_arbitrary_evaluator_bearer_is_never_used_as_lunit_credential(monkeypatch):
    class RecordingL2:
        credentials: list[str | None] = []

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).credentials.append(settings.api_key)
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            assert kwargs["messages"][0]["role"] == "system"
            return L2Completion(content="직접 의료 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    response = await _post(
        create_app(_settings()),
        {"messages": [{"role": "user", "content": "질문"}]},
        headers={"Authorization": "Bearer evaluator-placeholder"},
    )

    assert response.status_code == 200
    assert RecordingL2.credentials[0] != "evaluator-placeholder"
    assert RecordingL2.credentials[0].startswith("lunit_")


async def test_environment_credential_precedes_valid_lunit_bearer(monkeypatch):
    class RecordingL2:
        credential = None

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).credential = settings.api_key
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            return L2Completion(content="직접 의료 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    response = await _post(
        create_app(_settings(lunit_fm_api_key="lunit_environment_test")),
        {"messages": [{"role": "user", "content": "질문"}]},
        headers={"Authorization": "Bearer lunit_request_test"},
    )

    assert response.status_code == 200
    assert RecordingL2.credential == "lunit_environment_test"


async def test_valid_lunit_bearer_and_non_bearer_requests_remain_safe(monkeypatch):
    class RecordingL2:
        credentials: list[str | None] = []

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).credentials.append(settings.api_key)
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            return L2Completion(content="직접 의료 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    app = create_app(_settings())
    valid = await _post(
        app,
        {"messages": [{"role": "user", "content": "질문"}]},
        headers={"Authorization": "Bearer lunit_request_test"},
    )
    non_bearer = await _post(
        app,
        {"messages": [{"role": "user", "content": "질문"}]},
        headers={"Authorization": "Basic evaluator-placeholder"},
    )

    assert valid.status_code == non_bearer.status_code == 200
    assert RecordingL2.credentials[0] == "lunit_request_test"
    assert RecordingL2.credentials[1] != "evaluator-placeholder"


@pytest.mark.parametrize(
    ("requested_max_tokens", "expected_max_tokens"),
    [(700, 700), (5_000, 4_096)],
)
async def test_requested_max_tokens_is_capped_per_request(
    monkeypatch,
    requested_max_tokens,
    expected_max_tokens,
):
    class RecordingL2:
        configured_tokens: list[int] = []

        def __init__(self, settings, *, http_client=None):
            del http_client
            type(self).configured_tokens.append(settings.max_completion_tokens)
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            return L2Completion(content="토큰 제한 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    response = await _post(
        create_app(_settings()),
        {
            "messages": [{"role": "user", "content": "질문"}],
            "max_tokens": requested_max_tokens,
        },
    )

    assert response.status_code == 200
    assert RecordingL2.configured_tokens == [expected_max_tokens]


async def test_direct_upstream_receives_the_full_multi_turn_history(monkeypatch):
    class RecordingL2:
        calls: list[dict] = []

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
    response = await _post(create_app(_settings()), {"messages": history})

    assert response.status_code == 200
    assert RecordingL2.calls[0]["messages"][1:] == history


async def test_concurrent_requests_use_distinct_l2_settings_without_token_leakage(monkeypatch):
    class RecordingL2:
        configured_tokens: list[int] = []

        def __init__(self, settings, *, http_client=None):
            del http_client
            self.max_tokens = settings.max_completion_tokens
            type(self).configured_tokens.append(self.max_tokens)
            self.last_usage = TokenUsage()

        async def complete(self, **kwargs):
            del kwargs
            await asyncio.sleep(0)
            return L2Completion(content=f"{self.max_tokens} 토큰 답변")

    monkeypatch.setattr("app.L2Client", RecordingL2)
    app = create_app(_settings())
    first, second = await asyncio.gather(
        _post(app, {"messages": [{"role": "user", "content": "첫 요청"}], "max_tokens": 700}),
        _post(app, {"messages": [{"role": "user", "content": "둘째 요청"}], "max_tokens": 5_000}),
    )

    assert first.status_code == second.status_code == 200
    assert sorted(RecordingL2.configured_tokens) == [700, 4_096]
    assert {
        first.json()["choices"][0]["message"]["content"],
        second.json()["choices"][0]["message"]["content"],
    } == {"700 토큰 답변", "4096 토큰 답변"}


async def test_request_logs_exclude_medical_text_and_credentials(caplog):
    caplog.set_level(logging.INFO)
    response = await _post(
        create_app(_settings(), FakeOrchestrator()),
        {"messages": [{"role": "user", "content": "PRIVATE-MEDICAL-QUESTION"}]},
        headers={"Authorization": "Bearer evaluator-placeholder"},
    )

    assert response.status_code == 200
    assert "request_complete" in caplog.text
    assert "PRIVATE-MEDICAL-QUESTION" not in caplog.text
    assert "evaluator-placeholder" not in caplog.text
