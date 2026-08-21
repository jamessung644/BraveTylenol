import json

import httpx
import pytest

from app.config import Settings
from app.fast_harness import FastL2Harness, L2ResponseError


def settings() -> Settings:
    return Settings(api_key="secret", _env_file=None)


@pytest.mark.asyncio
async def test_one_turn_makes_exactly_one_tool_free_low_effort_l2_call():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "upstream-id",
                "created": 123,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "빠른 답변"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = FastL2Harness(settings(), http_client=client)
    history = [
        {"role": "user", "content": "질문 1"},
        {"role": "assistant", "content": "답변 1"},
        {"role": "user", "content": "후속 질문"},
    ]

    result = await driver.answer(history, requested_max_tokens=5000)
    await client.aclose()

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert "tools" not in payload
    assert payload["reasoning_effort"] == "low"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 1024
    assert payload["messages"][1:] == history
    assert result["choices"][0]["message"]["content"] == "빠른 답변"


@pytest.mark.asyncio
async def test_blank_l2_completion_fails_instead_of_retrying():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    driver = FastL2Harness(settings(), http_client=client)

    with pytest.raises(L2ResponseError, match="no final text"):
        await driver.answer([{"role": "user", "content": "질문"}])
    await client.aclose()

    assert calls == 1
