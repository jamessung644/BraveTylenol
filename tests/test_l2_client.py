import json

import httpx
import pytest

from harness.errors import (
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from harness.l2_client import L2Client


def completion_payload(message: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


async def test_complete_preserves_messages_and_tools(settings_with_key):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion_payload({"role": "assistant", "content": "확인했습니다."}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings_with_key, http_client=http_client)
        result = await client.complete(
            messages=[{"role": "user", "content": "질문"}],
            tools=[{"type": "function", "function": {"name": "retrieve_relevant_content", "description": "검색", "parameters": {"type": "object"}}}],
            tool_choice="auto",
        )

    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"]["model"] == "Lunit/L2-preview"
    assert seen["body"]["messages"] == [{"role": "user", "content": "질문"}]
    assert seen["body"]["tool_choice"] == "auto"
    assert result.content == "확인했습니다."
    assert result.usage.total_tokens == 14


async def test_complete_parses_assistant_tool_calls(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=completion_payload(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "retrieve_relevant_content", "arguments": "{\"query\": \"혈압\"}"}}],
                }
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])

    assert result.content is None
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].function.arguments == '{"query": "혈압"}'


async def test_complete_retries_retryable_response_once(settings_with_key, monkeypatch):
    attempts = 0
    sleeps = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"type": "rate_limit"}})
        return httpx.Response(200, json=completion_payload({"role": "assistant", "content": "재시도 성공"}))

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("harness.l2_client.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        result = await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])

    assert result.content == "재시도 성공"
    assert attempts == 2
    assert sleeps == [0.25]


async def test_complete_does_not_retry_non_retryable_response(settings_with_key):
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"type": "invalid_request"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(UpstreamResponseError, match="400"):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])

    assert attempts == 1


async def test_complete_translates_timeout(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(UpstreamTimeoutError):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])


async def test_complete_translates_transport_error(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(UpstreamTransportError):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])


async def test_complete_rejects_invalid_json(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])


async def test_complete_rejects_missing_choices(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])


async def test_complete_rejects_blank_completion_without_tool_calls(settings_with_key):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_payload({"role": "assistant", "content": "   "}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError):
            await L2Client(settings_with_key, http_client=http_client).complete(messages=[{"role": "user", "content": "질문"}])
