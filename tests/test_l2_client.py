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
                "choices": [{"message": {"role": "assistant", "content": "연결 성공"}}],
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
        "max_tokens": 3072,
        "reasoning_effort": "low",
        "temperature": 0.0,
    }
    assert result.content == "연결 성공"
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
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "최종 답변"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "질문"}]
        )

    assert result.content == "최종 답변"
    assert len(requests) == 2
    assert requests[1]["max_tokens"] == 1536
    assert "The patient needs clear red flags" in requests[1]["messages"][-2]["content"]
    assert "final user-facing answer" in requests[1]["messages"][-1]["content"]


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
