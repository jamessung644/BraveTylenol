import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app as app_module
from lunit_hackathon.config import Settings
from lunit_hackathon.schemas import L2Completion, TokenUsage


@dataclass(frozen=True)
class RecordedCall:
    api_key: str | None
    max_completion_tokens: int
    messages: tuple[dict[str, Any], ...]
    argument_names: frozenset[str]


class RecordingL2:
    calls: list[RecordedCall] = []
    semaphores: list[asyncio.Semaphore | None] = []

    def __init__(self, settings: Settings, *, http_client=None, request_semaphore=None):
        del http_client
        type(self).semaphores.append(request_semaphore)
        self._api_key = settings.api_key
        self._max_completion_tokens = settings.max_completion_tokens
        self.last_usage = TokenUsage()
        self.last_finish_reason: str | None = None

    async def complete(self, **kwargs: Any) -> L2Completion:
        messages = tuple(dict(message) for message in kwargs["messages"])
        question = next(
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        await asyncio.sleep(int(question.rsplit("-", 1)[-1]) % 3 / 1_000)
        type(self).calls.append(
            RecordedCall(
                api_key=self._api_key,
                max_completion_tokens=self._max_completion_tokens,
                messages=messages,
                argument_names=frozenset(kwargs),
            )
        )
        self.last_usage = TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
        self.last_finish_reason = "stop"
        return L2Completion(
            content=f"answer-for-{question}",
            finish_reason="stop",
            usage=self.last_usage,
        )


def _clear_evaluator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LUNIT_FM_API_KEY",
        "AGENT_MODE",
        "HARNESS_MODE",
        "LUNIT_MCP_URL",
        "ENABLE_RAG",
        "MAX_COMPLETION_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def _install_recording_l2(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingL2.calls.clear()
    RecordingL2.semaphores.clear()
    monkeypatch.setattr(app_module, "L2Client", RecordingL2)


async def test_coeval_no_env_preflight_accepts_bearer_and_omitted_model(monkeypatch):
    _clear_evaluator_environment(monkeypatch)
    _install_recording_l2(monkeypatch)
    application = app_module.create_app(Settings(_env_file=None))

    async with application.router.lifespan_context(application):
        async with AsyncClient(
            transport=ASGITransport(app=application),
            base_url="http://test",
        ) as client:
            health, healthz, models = await asyncio.gather(
                client.get("/health"),
                client.get("/healthz"),
                client.get("/v1/models"),
            )
            chat = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer preflight-secret-0"},
                json={"messages": [{"role": "user", "content": "preflight-question-0"}]},
            )

    assert [health.status_code, healthz.status_code, models.status_code, chat.status_code] == [
        200,
        200,
        200,
        200,
    ]
    assert health.json() == healthz.json() == {"status": "ok"}
    assert models.json()["data"][0]["id"] == "team-chatbot"
    assert chat.json()["model"] == "team-chatbot"
    assert chat.json()["choices"][0]["message"]["content"] == (
        "answer-for-preflight-question-0"
    )
    assert len(RecordingL2.calls) == 1
    assert RecordingL2.semaphores[0] is application.state.l2_request_semaphore
    assert RecordingL2.calls[0].api_key == "preflight-secret-0"
    assert RecordingL2.calls[0].argument_names.isdisjoint({"tools", "tool_choice"})


async def test_coeval_parallel_batch_is_request_isolated_and_logs_are_sanitized(
    monkeypatch,
    caplog,
):
    _clear_evaluator_environment(monkeypatch)
    _install_recording_l2(monkeypatch)
    application = app_module.create_app(Settings(_env_file=None))
    keys = [f"batch-secret-key-{index}" for index in range(16)]
    questions = [f"private-batch-question-{index}" for index in range(16)]
    histories = [
        [
            {"role": "user", "content": f"history-user-{index}"},
            {"role": "assistant", "content": f"history-assistant-{index}"},
            {"role": "user", "content": questions[index]},
        ]
        for index in range(16)
    ]

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        with caplog.at_level(logging.INFO):
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {keys[index]}"},
                        json={
                            "messages": histories[index],
                            "max_tokens": 700 if index % 2 == 0 else 5_000,
                        },
                    )
                    for index in range(16)
                )
            )

    assert all(response.status_code == 200 for response in responses)
    chat_ids = [response.json()["id"] for response in responses]
    request_ids = [response.headers["x-request-id"] for response in responses]
    assert len(set(chat_ids)) == len(chat_ids) == 16
    assert len(set(request_ids)) == len(request_ids) == 16

    calls_by_question = {
        call.messages[-1]["content"]: call
        for call in RecordingL2.calls
    }
    assert len(calls_by_question) == 16
    for index, response in enumerate(responses):
        question = questions[index]
        call = calls_by_question[question]
        assert response.json()["model"] == "team-chatbot"
        assert response.json()["choices"][0]["message"]["content"] == (
            f"answer-for-{question}"
        )
        assert call.api_key == keys[index]
        assert list(call.messages[1:]) == histories[index]
        assert call.max_completion_tokens == (700 if index % 2 == 0 else 1_024)
        assert call.argument_names.isdisjoint({"tools", "tool_choice"})

    for sensitive_value in [*keys, *questions]:
        assert sensitive_value not in caplog.text


@pytest.mark.parametrize("mode_variable", ["AGENT_MODE", "HARNESS_MODE"])
async def test_coeval_rag_environment_pollution_stays_one_call_direct_without_opt_in(
    monkeypatch,
    mode_variable,
):
    _clear_evaluator_environment(monkeypatch)
    monkeypatch.setenv(mode_variable, "rag")
    monkeypatch.setenv("LUNIT_MCP_URL", "https://mcp.injected-by-pipeline.test")
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    _install_recording_l2(monkeypatch)
    application = app_module.create_app(Settings(_env_file=None))

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer pollution-secret-0"},
            json={"messages": [{"role": "user", "content": "pollution-question-0"}]},
        )

    assert response.status_code == 200
    assert len(RecordingL2.calls) == 1
    assert RecordingL2.calls[0].api_key == "pollution-secret-0"
    assert RecordingL2.calls[0].argument_names.isdisjoint({"tools", "tool_choice"})


async def test_one_timed_out_request_does_not_poison_parallel_or_followup_requests(monkeypatch):
    _clear_evaluator_environment(monkeypatch)

    class MixedSpeedL2(RecordingL2):
        async def complete(self, **kwargs: Any) -> L2Completion:
            question = kwargs["messages"][-1]["content"]
            if question == "slow-question":
                await asyncio.sleep(0.03)
            return await super().complete(**kwargs)

    RecordingL2.calls.clear()
    monkeypatch.setattr(app_module, "L2Client", MixedSpeedL2)
    settings = Settings(_env_file=None).model_copy(
        update={"request_timeout_seconds": 0.01}
    )
    application = app_module.create_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        slow, fast = await asyncio.gather(
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer timeout-secret-slow"},
                json={"messages": [{"role": "user", "content": "slow-question"}]},
            ),
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer timeout-secret-fast"},
                json={"messages": [{"role": "user", "content": "fast-question-1"}]},
            ),
        )
        followup = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer timeout-secret-followup"},
            json={"messages": [{"role": "user", "content": "fast-question-2"}]},
        )
        health = await client.get("/health")

    assert slow.status_code == 504
    assert fast.status_code == 200
    assert followup.status_code == 200
    assert health.status_code == 200
