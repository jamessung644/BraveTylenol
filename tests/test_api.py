from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.fast_harness import L2TimeoutError
from app.main import create_app


class FakeHarness:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[list[dict[str, Any]], int | None]] = []

    async def answer(
        self,
        messages: list[dict[str, Any]],
        *,
        requested_max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append((messages, requested_max_tokens))
        if self.error:
            raise self.error
        assert self.result is not None
        return self.result


def settings(api_key: str = "test-key") -> Settings:
    return Settings(api_key=api_key, _env_file=None)


def test_healthz():
    with TestClient(create_app(settings=settings(api_key=""))) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/health").json() == {"status": "ok"}


def test_models_is_openai_compatible():
    with TestClient(create_app(settings=settings(api_key=""))) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert response.json()["data"][0]["id"] == "Lunit/L2-preview"


def test_missing_key_fails_closed():
    with TestClient(create_app(settings=settings(api_key=""))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "안녕하세요"}]},
        )
    assert response.status_code == 503


def test_evaluator_bearer_token_is_forwarded_to_l2(monkeypatch):
    received_keys: list[str] = []

    class RecordingHarness(FakeHarness):
        def __init__(self, configured, *, http_client):
            del http_client
            received_keys.append(configured.api_key)
            super().__init__(
                {
                    "id": "chatcmpl-auth",
                    "object": "chat.completion",
                    "created": 1,
                    "model": configured.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "인증 성공"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": None,
                }
            )

    monkeypatch.setattr("app.main.FastL2Harness", RecordingHarness)
    application = create_app(settings=settings(api_key=""))

    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer evaluator-secret"},
            json={"messages": [{"role": "user", "content": "질문"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "인증 성공"
    assert received_keys == ["evaluator-secret"]


def test_non_bearer_authorization_is_rejected():
    application = create_app(settings=settings(api_key=""))
    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Basic invalid"},
            json={"messages": [{"role": "user", "content": "질문"}]},
        )
    assert response.status_code == 503


def test_chat_preserves_history_and_returns_exact_l2_answer():
    upstream = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "Lunit/L2-preview",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "L2 최종 답변"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    driver = FakeHarness(upstream)
    application = create_app(settings=settings(), harness=driver)
    history = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
        {"role": "user", "content": "그럼 지금은요?"},
    ]

    with TestClient(application) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": history, "max_tokens": 700},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "L2 최종 답변"
    assert driver.calls == [(history, 700)]


def test_streaming_is_rejected_before_calling_l2():
    driver = FakeHarness({})
    with TestClient(create_app(settings=settings(), harness=driver)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "질문"}], "stream": True},
        )
    assert response.status_code == 400
    assert driver.calls == []


def test_l2_timeout_maps_to_504():
    driver = FakeHarness(error=L2TimeoutError("slow"))
    with TestClient(create_app(settings=settings(), harness=driver)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "질문"}]},
        )
    assert response.status_code == 504
