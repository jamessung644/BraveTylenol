import json

import httpx
import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
)
from lunit_hackathon.l2_client import L2Client


def settings_with_key(monkeypatch) -> Settings:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    return Settings(_env_file=None)


async def test_complete_calls_l2_chat_completions(monkeypatch):
    settings = settings_with_key(monkeypatch)
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "연결 성공"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "테스트"}]
        )

    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"] == {
        "model": "Lunit/L2-preview",
        "messages": [{"role": "user", "content": "테스트"}],
        "max_tokens": 6144,
        "reasoning_effort": "low",
        "temperature": 0.0,
    }
    assert result.content == "연결 성공"
    assert result.finish_reason == "length"
    assert result.usage.total_tokens == 6


async def test_complete_parses_tool_calls(monkeypatch):
    settings = settings_with_key(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "테스트"}]
        )

    assert result.tool_calls[0].function.name == "lookup"


async def test_complete_requires_real_api_key(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError):
        await L2Client(settings).complete(messages=[{"role": "user", "content": "테스트"}])


async def test_complete_retries_transient_status_once(monkeypatch):
    monkeypatch.setenv("L2_RETRY_ATTEMPTS", "1")
    settings = settings_with_key(monkeypatch)
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"type": "overloaded"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "성공"}}]})

    async def no_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr("lunit_hackathon.l2_client.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "테스트"}]
        )

    assert attempts == 2
    assert result.content == "성공"


async def test_complete_retries_two_consecutive_bad_gateways(monkeypatch):
    settings = settings_with_key(monkeypatch).model_copy(update={"retry_attempts": 2})
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(502, json={"error": {"type": "bad_gateway"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "성공"}}]})

    async def no_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr("lunit_hackathon.l2_client.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "테스트"}]
        )

    assert attempts == 3
    assert result.content == "성공"


async def test_retry_and_blank_recovery_share_one_absolute_deadline(monkeypatch):
    settings = settings_with_key(monkeypatch).model_copy(
        update={"request_timeout_seconds": 65, "retry_attempts": 1}
    )
    responses = iter(
        [
            httpx.Response(502, json={"error": {"type": "bad_gateway"}}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": " ",
                                "reasoning": "A short final answer is ready.",
                            },
                            "finish_reason": "length",
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"choices": [{"message": {"content": "복구 성공"}}]}),
        ]
    )
    observed_timeouts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_timeouts.append(request.extensions["timeout"])
        return next(responses)

    async def no_sleep(delay: float) -> None:
        del delay

    clock = iter((100.0, 100.0, 101.0, 110.0, 110.0, 111.0, 120.0, 120.0, 121.0))
    monkeypatch.setattr("lunit_hackathon.l2_client.perf_counter", lambda: next(clock))
    monkeypatch.setattr("lunit_hackathon.l2_client.asyncio.sleep", no_sleep)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        result = await client.complete(
            messages=[{"role": "user", "content": "질문"}]
        )

    assert result.content == "복구 성공"
    assert [timeout["read"] for timeout in observed_timeouts] == [65, 55, 45]
    assert [timeout["write"] for timeout in observed_timeouts] == [65, 55, 45]
    assert [timeout["connect"] for timeout in observed_timeouts] == [5, 5, 5]
    assert [timeout["pool"] for timeout in observed_timeouts] == [5, 5, 5]


async def test_complete_accumulates_usage_across_generation_steps(monkeypatch):
    settings = settings_with_key(monkeypatch)
    responses = iter(
        [
            {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "성공"}}],
                "usage": next(responses),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        await client.complete(messages=[{"role": "user", "content": "첫 단계"}])
        await client.complete(messages=[{"role": "user", "content": "둘째 단계"}])

    assert client.last_usage.prompt_tokens == 11
    assert client.last_usage.completion_tokens == 5
    assert client.last_usage.total_tokens == 16


async def test_complete_rejects_malformed_success(monkeypatch):
    settings = settings_with_key(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings, http_client=http_client).complete(
                messages=[{"role": "user", "content": "테스트"}]
            )


async def test_complete_recovers_blank_direct_completion_once(monkeypatch):
    settings = settings_with_key(monkeypatch)
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "   ",
                                "reasoning": "The patient needs clear red flags and next steps.",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "최종 답변"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        result = await client.complete(messages=[{"role": "user", "content": "질문"}])

    assert result.content == "최종 답변"
    assert len(requests) == 2
    assert requests[1]["max_tokens"] == 1536
    assert "The patient needs clear red flags" in requests[1]["messages"][-2]["content"]
    assert "final user-facing answer" in requests[1]["messages"][-1]["content"]
    assert result.usage.total_tokens == 7
    assert client.last_usage.total_tokens == 18


async def test_complete_rejects_blank_after_one_recovery(monkeypatch):
    settings = settings_with_key(monkeypatch)
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings, http_client=http_client).complete(
                messages=[{"role": "user", "content": "질문"}]
            )

    assert attempts == 2


async def test_complete_recovers_blank_final_submission_once(monkeypatch):
    settings = settings_with_key(monkeypatch)
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": "A short grounded answer is ready.",
                            },
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "submit-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_final_answer",
                                        "arguments": '{"answer":"복구된 최종 답변"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    final_tool = {
        "type": "function",
        "function": {
            "name": "submit_final_answer",
            "parameters": {"type": "object"},
        },
    }
    forced_choice = {
        "type": "function",
        "function": {"name": "submit_final_answer"},
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "질문"}],
            tools=[final_tool],
            tool_choice=forced_choice,
        )

    assert result.tool_calls[0].function.name == "submit_final_answer"
    assert len(requests) == 2
    assert requests[1]["tool_choice"] == forced_choice
    assert "Call submit_final_answer exactly once" in requests[1]["messages"][-1]["content"]


async def test_complete_does_not_recover_blank_tool_planner(monkeypatch):
    settings = settings_with_key(monkeypatch)
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings, http_client=http_client).complete(
                messages=[{"role": "user", "content": "질문"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "retrieve",
                            "description": "search",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )

    assert attempts == 1


async def test_upstream_error_does_not_leak_response_message(monkeypatch):
    settings = settings_with_key(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"type": "auth_error", "message": "sensitive upstream detail"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(UpstreamResponseError) as caught:
            await L2Client(settings, http_client=http_client).complete(
                messages=[{"role": "user", "content": "테스트"}]
            )

    assert "401" in str(caught.value)
    assert "sensitive upstream detail" not in str(caught.value)
