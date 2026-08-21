import asyncio
import json

import httpx
import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
)
from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.schemas import ChatMessage


def settings_with_key(monkeypatch) -> Settings:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
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

    assert seen["authorization"] == "Bearer lunit_test_team_key"
    assert seen["body"] == {
        "model": "Lunit/L2-preview",
        "messages": [{"role": "user", "content": "테스트"}],
        "max_tokens": 4096,
        "reasoning_effort": "low",
        "temperature": 0.0,
    }
    assert result.content == "연결 성공"
    assert result.finish_reason == "length"
    assert result.usage.total_tokens == 6


async def test_attempt_budget_includes_waiting_for_the_model_semaphore(monkeypatch):
    settings = settings_with_key(monkeypatch)
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    called = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        del request
        called = True
        return httpx.Response(200, json={"choices": [{"message": {"content": "late"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client, semaphore=semaphore)
        with pytest.raises(UpstreamTimeoutError):
            await client.complete(
                messages=[{"role": "user", "content": "질문"}],
                attempt_timeout_seconds=0.01,
            )

    assert called is False
    semaphore.release()


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


async def test_complete_rejects_legacy_function_call_in_tool_phase(monkeypatch):
    settings = settings_with_key(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "lookup",
                                "arguments": '{"q":"x"}',
                            },
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(MalformedUpstreamResponseError, match="legacy function"):
            await L2Client(settings, http_client=http_client).complete(
                messages=[{"role": "user", "content": "테스트"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                allow_blank_recovery=False,
            )


async def test_complete_requires_real_api_key(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError):
        await L2Client(settings).complete(messages=[{"role": "user", "content": "테스트"}])


async def test_complete_retries_transient_status_once(monkeypatch):
    settings = settings_with_key(monkeypatch).model_copy(update={"retry_attempts": 1})
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


def test_remaining_request_seconds_uses_the_shared_absolute_deadline(monkeypatch):
    settings = settings_with_key(monkeypatch).model_copy(
        update={"request_timeout_seconds": 165}
    )
    clock = iter((120.0,))
    monkeypatch.setattr("lunit_hackathon.l2_client.perf_counter", lambda: next(clock))
    client = L2Client(settings)

    assert client.remaining_request_seconds() == 165
    client._deadline = 265.0
    assert client.remaining_request_seconds() == 145


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


async def test_complete_can_defer_one_blank_final_to_generation_validator(monkeypatch):
    settings = settings_with_key(monkeypatch)
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "   "},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        completion = await L2Client(settings, http_client=http_client).complete(
            messages=[{"role": "user", "content": "질문"}],
            allow_blank_recovery=False,
            allow_empty_completion=True,
        )

    assert completion.content == "   "
    assert completion.finish_reason == "length"
    assert attempts == 1


async def test_generation_with_real_l2_client_recovers_blank_from_frozen_inbound(monkeypatch):
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
                                "content": " ",
                                "reasoning": "private interrupted draft",
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
                        "message": {"content": "완결된 L2 사용자 답변"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        answer = await GenerationEngine(client, None).direct_answer(
            [ChatMessage(role="user", content="두통이 있을 때 일반적으로 뭘 확인하나요?")]
        )

    assert answer == "완결된 L2 사용자 답변"
    assert len(requests) == 2
    assert "private interrupted draft" not in json.dumps(
        requests[1]["messages"],
        ensure_ascii=False,
    )
    assert "재작성 단계" in requests[1]["messages"][0]["content"]


async def test_generation_recovers_legacy_function_call_from_real_parser(monkeypatch):
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
                                "function_call": {
                                    "name": "retrieve_relevant_content",
                                    "arguments": '{"query":"x"}',
                                },
                            },
                            "finish_reason": "function_call",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "복구된 평문 답변"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        answer = await GenerationEngine(client, None).direct_answer(
            [ChatMessage(role="user", content="일반적인 건강 질문")]
        )

    assert answer == "복구된 평문 답변"
    assert len(requests) == 2
    assert "legacy-function-call" not in json.dumps(requests[1], ensure_ascii=False)


async def test_generation_recovers_malformed_modern_tool_call_from_real_parser(
    monkeypatch,
):
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
                                "tool_calls": [
                                    {
                                        "id": "invalid call id",
                                        "type": "function",
                                        "function": {
                                            "name": "retrieve_relevant_content",
                                            "arguments": '{"query":"x"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "복구된 평문 답변"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        answer = await GenerationEngine(client, None).direct_answer(
            [ChatMessage(role="user", content="일반적인 건강 질문")]
        )

    assert answer == "복구된 평문 답변"
    assert len(requests) == 2
    assert "invalid call id" not in json.dumps(requests[1], ensure_ascii=False)


async def test_generation_rejects_legacy_function_call_in_recovery(monkeypatch):
    settings = settings_with_key(monkeypatch)
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "function_call": {
                                "name": "retrieve_relevant_content",
                                "arguments": '{"query":"x"}',
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        with pytest.raises(MalformedUpstreamResponseError, match="legacy function"):
            await GenerationEngine(client, None).direct_answer(
                [ChatMessage(role="user", content="일반적인 건강 질문")]
            )

    assert len(requests) == 2


async def test_generation_rejects_malformed_modern_tool_call_in_recovery(monkeypatch):
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
                            "message": {"content": "<tool_call>invalid</tool_call>"},
                            "finish_reason": "stop",
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
                                    "id": "invalid call id",
                                    "type": "function",
                                    "function": {
                                        "name": "retrieve_relevant_content",
                                        "arguments": '{"query":"x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = L2Client(settings, http_client=http_client)
        with pytest.raises(MalformedUpstreamResponseError, match="invalid completion"):
            await GenerationEngine(client, None).direct_answer(
                [ChatMessage(role="user", content="일반적인 건강 질문")]
            )

    assert len(requests) == 2


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
