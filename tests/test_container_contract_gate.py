import asyncio
import json
import sys
import types

import httpx
import pytest

import scripts.container_contract_gate as gate_module
from scripts.container_contract_gate import (
    COEVAL_COMMIT,
    COEVAL_CONCURRENCY,
    COEVAL_MAX_TOKENS,
    COEVAL_TIMEOUT_SECONDS,
    LIVE_VECTORS,
    _release_contract_metadata,
    _validate_loopback_base_url,
    run_boundary_gate,
    run_live_http_gate,
    run_official_coeval_gate,
)


def test_pinned_public_contract_metadata_is_stable():
    metadata = _release_contract_metadata()

    assert COEVAL_COMMIT == "741263cfafba687f8baeb7422c747ef9557df1c4"
    assert metadata["official_conquer_val"] == {
        "examples": 301,
        "concurrency": 16,
        "inference_attempts": 2,
        "retry_delay_seconds": 2.0,
        "max_tokens": 6_144,
        "timeout_seconds": 180.0,
        "score_inference_failures_as_zero": True,
    }
    assert metadata["gate_policy"]["max_live_calls"] == 18
    assert metadata["gate_policy"]["internal_retries"] == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://model.hackathon.lunit.io",
        "http://127.0.0.1:8000/v1",
        "http://user:secret@127.0.0.1:8000",
    ],
)
def test_gate_refuses_non_loopback_or_credential_bearing_targets(url):
    with pytest.raises(ValueError):
        _validate_loopback_base_url(url)


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_gate_accepts_only_loopback_container_targets(url, normalized):
    assert _validate_loopback_base_url(url) == normalized


async def test_boundary_gate_checks_failure_contract_without_upstream_calls():
    observed_paths = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_paths.append(request.url.path)
        body = await request.aread()
        if request.method == "GET" and request.url.path in {"/health", "/healthz"}:
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/readyz":
            return httpx.Response(503, json={"detail": "credential required"})
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": []})
        if len(body) > 600_000:
            return httpx.Response(413, json={"detail": "too large"})
        if request.headers.get("content-encoding") == "gzip":
            return httpx.Response(400, json={"detail": "compressed"})
        if request.headers.get("content-type") != "application/json":
            return httpx.Response(415, json={"detail": "content type"})
        if body in {b'{"messages": [', (
            b'{"messages":[{"role":"user","content":"a"}],'
            b'"messages":[{"role":"user","content":"b"}]}'
        )}:
            return httpx.Response(400, json={"detail": "invalid json"})
        payload = json.loads(body)
        if payload.get("stream") is True or "tools" in payload:
            return httpx.Response(400, json={"detail": "unsupported"})
        if payload.get("messages") == [] or any(
            isinstance(message.get("content"), list)
            for message in payload.get("messages", [])
        ):
            return httpx.Response(422, json={"detail": "validation"})
        return httpx.Response(503, json={"detail": "credential required"})

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        report = await run_boundary_gate("http://127.0.0.1:8000", client=client)

    assert report["passed"] is True
    assert report["case_count"] == 14
    assert report["external_calls_expected"] == 0
    assert report["failures"] == []
    assert "/v1/chat/completions" in observed_paths


async def test_live_gate_is_exact_c16_single_attempt_and_never_persists_content_or_key():
    active = 0
    maximum_active = 0
    seen_authorization = []
    seen_payloads = []
    secret = "lunit_TEST_SECRET_SENTINEL"
    response_sentinel = "PRIVATE_RESPONSE_SENTINEL"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            seen_authorization.append(request.headers.get("authorization"))
            seen_payloads.append(json.loads(await request.aread()))
            await asyncio.sleep(0.01)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "team-chatbot",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_sentinel,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        finally:
            active -= 1

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        report = await run_live_http_gate(
            "http://127.0.0.1:8000",
            api_key=secret,
            client=client,
        )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["passed"] is True
    assert report["vector_count"] == COEVAL_CONCURRENCY == len(LIVE_VECTORS) == 16
    assert report["attempts_per_vector"] == 1
    assert report["runner_retry_count"] == 0
    assert report["route_shape_counts"] == {"direct": 8, "rag": 8}
    assert report["nonempty_rate"] == 1.0
    assert maximum_active == COEVAL_CONCURRENCY
    assert len(seen_payloads) == COEVAL_CONCURRENCY
    assert all(payload["max_tokens"] == COEVAL_MAX_TOKENS for payload in seen_payloads)
    assert all(payload["temperature"] == 0 for payload in seen_payloads)
    assert seen_authorization == [f"Bearer {secret}"] * COEVAL_CONCURRENCY
    assert secret not in serialized
    assert response_sentinel not in serialized
    assert "와파린" not in serialized
    assert COEVAL_TIMEOUT_SECONDS == 180.0


async def test_official_client_gate_uses_two_passthrough_calls_without_content_persistence(
    monkeypatch,
    tmp_path,
):
    captured_configs = []
    captured_messages = []
    response_sentinel = "OFFICIAL_RESPONSE_SENTINEL"

    class FakeConfig:
        def __init__(self, **kwargs):
            captured_configs.append(kwargs)

    class FakePassthroughClient:
        def __init__(self, llm):
            assert llm == "official-llm"

        async def generate(self, messages):
            captured_messages.append(messages)
            return response_sentinel

    passthrough_module = types.ModuleType("coeval.clients.passthrough")
    passthrough_module.PassthroughClient = FakePassthroughClient
    config_module = types.ModuleType("coeval.llm.config")
    config_module.LLMConfig = FakeConfig
    factory_module = types.ModuleType("coeval.llm.factory")
    factory_module.create_llm_client = lambda config: "official-llm"
    monkeypatch.setitem(sys.modules, "coeval.clients.passthrough", passthrough_module)
    monkeypatch.setitem(sys.modules, "coeval.llm.config", config_module)
    monkeypatch.setitem(sys.modules, "coeval.llm.factory", factory_module)
    monkeypatch.setattr(
        gate_module,
        "verify_coeval_checkout",
        lambda root, require_imports: {
            "phase": "coeval_pin_preflight",
            "passed": True,
            "commit": COEVAL_COMMIT,
        },
    )

    report = await run_official_coeval_gate(
        "http://127.0.0.1:8000",
        tmp_path,
        api_key="lunit_TEST_SECRET_SENTINEL",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["passed"] is True
    assert report["vector_count"] == 2
    assert len(captured_messages) == 2
    assert captured_configs == [
        {
            "api_base": "http://127.0.0.1:8000/v1",
            "model": "team-chatbot",
            "api_key": "lunit_TEST_SECRET_SENTINEL",
            "temperature": 0.0,
            "max_tokens": 6_144,
            "timeout": 180.0,
            "max_retries": 0,
            "is_function_calling_model": False,
        }
    ]
    assert response_sentinel not in serialized
    assert "lunit_TEST_SECRET_SENTINEL" not in serialized
