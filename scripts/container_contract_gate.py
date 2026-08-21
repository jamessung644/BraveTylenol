#!/usr/bin/env python3
"""Secret-safe HTTP release gate for the pinned CoEval client contract.

This script deliberately prints only case identifiers, status codes, aggregate
latencies, and structural pass/fail reasons.  It never prints request messages,
response content, or credentials.

The boundary phase is safe to run against a credential-free, network-disabled
container.  Live phases require ``--authorized-live`` and are restricted to a
loopback container endpoint.  ``release-live`` performs exactly 18 inference
requests: a C=16 direct HTTP matrix plus two requests through the official
CoEval ``PassthroughClient`` from the pinned checkout.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import ipaddress
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

COEVAL_COMMIT = "741263cfafba687f8baeb7422c747ef9557df1c4"
COEVAL_PASSTHROUGH_SHA256 = (
    "1eefc9e0c7fa7cb9ffd6bd733a484756cafb982095e0005068d3883ebd994b1b"
)
COEVAL_CONQUER_VAL_SHA256 = (
    "8f03ae15a657dd36a82e1c9c8d2969fb1a2ba423be504cb093751cf0ade676d7"
)
COEVAL_UV_LOCK_SHA256 = (
    "801c6880754d65bcd94f08471d0d0febde46cb295ee61377e5bf8aedc28828e7"
)
COEVAL_CONCURRENCY = 16
COEVAL_MAX_TOKENS = 6_144
COEVAL_TIMEOUT_SECONDS = 180.0
COEVAL_RUNNER_ATTEMPTS = 2
GATE_ATTEMPTS_PER_VECTOR = 1
MODEL_ID = "team-chatbot"
_API_KEY = re.compile(r"\Alunit_[A-Za-z0-9._~+/-]+={0,2}\Z")


@dataclass(frozen=True)
class LiveVector:
    case_id: str
    route_class: str
    messages: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    status: int | None
    latency_ms: float
    reason: str


# Dataset-independent synthetic prompts.  They test transport and routing-shaped
# concurrency only; their answer text is never graded or persisted by this gate.
LIVE_VECTORS: tuple[LiveVector, ...] = (
    LiveVector(
        "single_direct_general",
        "direct",
        ({"role": "user", "content": "감기 증상이 있을 때 집에서 어떻게 쉬면 좋아?"},),
    ),
    LiveVector(
        "single_direct_noisy",
        "direct",
        ({"role": "user", "content": "혈압 정상범위 대략 머야"},),
    ),
    LiveVector(
        "single_direct_general_principle",
        "direct",
        (
            {
                "role": "user",
                "content": "특정 기관 자료 말고 수분 섭취의 일반 원리만 설명해줘.",
            },
        ),
    ),
    LiveVector(
        "single_direct_hypothetical",
        "direct",
        ({"role": "user", "content": "만약 가슴이 아프면 어떤 증상을 살펴야 해?"},),
    ),
    LiveVector(
        "single_direct_past_history",
        "direct",
        ({"role": "user", "content": "10년 전 흉통 병력이 있는데 평소 운동 원칙이 궁금해."},),
    ),
    LiveVector(
        "single_direct_emergency",
        "direct",
        ({"role": "user", "content": "약을 30알 방금 먹었어. 지금 무엇을 해야 해?"},),
    ),
    LiveVector(
        "content_parts_direct",
        "direct",
        (
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "두통이 있을 때 "},
                    {"type": "text", "text": "일반적인 생활 관리를 알려줘."},
                ],
            },
        ),
    ),
    LiveVector(
        "multi_stale_source_ack",
        "direct",
        (
            {"role": "user", "content": "식약처 자료로 약 설명을 부탁해."},
            {"role": "assistant", "content": "요청하신 내용을 설명했습니다."},
            {"role": "user", "content": "고마워."},
        ),
    ),
    LiveVector(
        "single_rag_drug_pregnancy",
        "rag",
        (
            {
                "role": "user",
                "content": "와파린을 임신 중 복용해도 되는지 허가자료 근거와 함께 알려줘.",
            },
        ),
    ),
    LiveVector(
        "single_rag_interaction",
        "rag",
        (
            {
                "role": "user",
                "content": "아픽사반과 이트라코나졸의 허가상 상호작용을 확인해줘.",
            },
        ),
    ),
    LiveVector(
        "single_rag_pediatric_dose",
        "rag",
        (
            {
                "role": "user",
                "content": "체중 14 kg 소아의 아세트아미노펜 허가 용량 근거를 알려줘.",
            },
        ),
    ),
    LiveVector(
        "single_rag_indication",
        "rag",
        ({"role": "user", "content": "타이레놀의 국내 허가 적응증을 출처와 함께 알려줘."},),
    ),
    LiveVector(
        "single_rag_current_criteria",
        "rag",
        ({"role": "user", "content": "현재 당뇨병 진단 기준을 공식 근거로 확인해줘."},),
    ),
    LiveVector(
        "single_rag_english_source",
        "rag",
        ({"role": "user", "content": "Give official sources for aspirin contraindications."},),
    ),
    LiveVector(
        "single_rag_noisy_guideline",
        "rag",
        ({"role": "user", "content": "최신 가이드라잉 근거로 고혈압 진단기준 알려줘"},),
    ),
    LiveVector(
        "multi_rag_coreference",
        "rag",
        (
            {"role": "user", "content": "아픽사반을 복용하고 있어."},
            {"role": "assistant", "content": "복용 중인 약을 확인했습니다."},
            {"role": "user", "content": "무좀 치료도 고려 중이야."},
            {"role": "assistant", "content": "치료제에 따라 확인할 점이 다릅니다."},
            {
                "role": "user",
                "content": "그 약과 이트라코나졸 상호작용을 공식 자료로 확인해줘.",
            },
        ),
    ),
)


def _validate_loopback_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base URL must use HTTP or HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("base URL must not contain a path")
    host = parsed.hostname
    if host is None:
        raise ValueError("base URL must include a host")
    if host.casefold() != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("release gate target must be loopback")
        except ValueError as error:
            if str(error) == "release gate target must be loopback":
                raise
            raise ValueError("release gate target must be loopback") from error
    return base_url.rstrip("/")


def _read_optional_api_key(environment_name: str) -> str | None:
    value = os.getenv(environment_name)
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if len(normalized) > 4_096 or _API_KEY.fullmatch(normalized) is None:
        raise ValueError(f"{environment_name} is not a format-valid Lunit key")
    return normalized


def _headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 1)


def _latency_summary(results: Sequence[CaseResult]) -> dict[str, float]:
    values = [result.latency_ms for result in results]
    if not values:
        return {"total": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "total": round(sum(values), 1),
        "p50": _percentile_nearest_rank(values, 0.50),
        "p95": _percentile_nearest_rank(values, 0.95),
        "max": round(max(values), 1),
    }


def _result(
    case_id: str,
    started: float,
    *,
    status: int | None,
    passed: bool,
    reason: str,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        passed=passed,
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1_000, 1),
        reason=reason,
    )


def _safe_structure_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


def _valid_chat_structure(response: httpx.Response) -> tuple[bool, str]:
    payload = _safe_structure_json(response)
    if not isinstance(payload, Mapping):
        return False, "response_not_json_object"
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "choices_missing"
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return False, "choice_not_object"
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return False, "assistant_message_missing"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return False, "assistant_content_empty"
    if message.get("tool_calls"):
        return False, "public_tool_call_exposed"
    if choice.get("finish_reason") not in {"stop", "length"}:
        return False, "finish_reason_invalid"
    if payload.get("object") != "chat.completion":
        return False, "object_invalid"
    if not isinstance(payload.get("id"), str):
        return False, "id_missing"
    return True, "ok"


async def _expected_status_case(
    client: httpx.AsyncClient,
    case_id: str,
    method: str,
    path: str,
    expected_status: int,
    **kwargs: Any,
) -> CaseResult:
    started = time.perf_counter()
    try:
        response = await client.request(method, path, **kwargs)
    except Exception as error:  # noqa: BLE001 - the report exposes type only
        return _result(
            case_id,
            started,
            status=None,
            passed=False,
            reason=f"transport_{type(error).__name__}",
        )
    passed = response.status_code == expected_status
    return _result(
        case_id,
        started,
        status=response.status_code,
        passed=passed,
        reason="ok" if passed else f"expected_{expected_status}",
    )


async def run_boundary_gate(
    base_url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Exercise only paths that must not call L2 or MCP.

    The target container must have no valid credential.  This makes the phase
    suitable for ``--network=none`` and proves that health is not a fake chat
    success signal.
    """

    normalized_url = _validate_loopback_base_url(base_url)
    own_client = client is None
    active_client = client or httpx.AsyncClient(
        base_url=normalized_url,
        timeout=10.0,
        trust_env=False,
        follow_redirects=False,
    )
    cases: list[Awaitable[CaseResult]] = [
        _expected_status_case(active_client, "health", "GET", "/health", 200),
        _expected_status_case(active_client, "healthz", "GET", "/healthz", 200),
        _expected_status_case(active_client, "ready_without_key", "GET", "/readyz", 503),
        _expected_status_case(active_client, "models", "GET", "/v1/models", 200),
        _expected_status_case(
            active_client,
            "chat_without_key",
            "POST",
            "/v1/chat/completions",
            503,
            headers=_headers(),
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "credential-free probe"}],
            },
        ),
        _expected_status_case(
            active_client,
            "malformed_json",
            "POST",
            "/v1/chat/completions",
            400,
            headers=_headers(),
            content=b'{"messages": [',
        ),
        _expected_status_case(
            active_client,
            "duplicate_json_key",
            "POST",
            "/v1/chat/completions",
            400,
            headers=_headers(),
            content=(
                b'{"messages":[{"role":"user","content":"a"}],'
                b'"messages":[{"role":"user","content":"b"}]}'
            ),
        ),
        _expected_status_case(
            active_client,
            "wrong_content_type",
            "POST",
            "/v1/chat/completions",
            415,
            headers={"Content-Type": "text/plain"},
            content=b"not-json",
        ),
        _expected_status_case(
            active_client,
            "compressed_body",
            "POST",
            "/v1/chat/completions",
            400,
            headers={**_headers(), "Content-Encoding": "gzip"},
            content=b"not-compressed",
        ),
        _expected_status_case(
            active_client,
            "oversize_body",
            "POST",
            "/v1/chat/completions",
            413,
            headers=_headers(),
            content=b"{" + (b" " * 600_001),
        ),
        _expected_status_case(
            active_client,
            "stream_rejected",
            "POST",
            "/v1/chat/completions",
            400,
            headers=_headers(),
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "stream probe"}],
                "stream": True,
            },
        ),
        _expected_status_case(
            active_client,
            "empty_messages",
            "POST",
            "/v1/chat/completions",
            422,
            headers=_headers(),
            json={"model": MODEL_ID, "messages": []},
        ),
        _expected_status_case(
            active_client,
            "non_text_content_part",
            "POST",
            "/v1/chat/completions",
            422,
            headers=_headers(),
            json={
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": "blocked"}],
                    }
                ],
            },
        ),
        _expected_status_case(
            active_client,
            "caller_tools_rejected",
            "POST",
            "/v1/chat/completions",
            400,
            headers=_headers(),
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "tool probe"}],
                "tools": [],
            },
        ),
    ]
    try:
        results = list(await asyncio.gather(*cases))
    finally:
        if own_client:
            await active_client.aclose()
    status_counts = Counter(
        "transport_error" if item.status is None else str(item.status) for item in results
    )
    return {
        "phase": "credential_free_boundary",
        "passed": all(item.passed for item in results),
        "case_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "latency_ms": _latency_summary(results),
        "failures": [asdict(item) for item in results if not item.passed],
        "network_expectation": "container_network_none",
        "external_calls_expected": 0,
    }


async def _run_live_vector(
    client: httpx.AsyncClient,
    vector: LiveVector,
    start_event: asyncio.Event,
    api_key: str | None,
) -> CaseResult:
    await start_event.wait()
    started = time.perf_counter()
    try:
        response = await client.post(
            "/v1/chat/completions",
            headers=_headers(api_key),
            json={
                "model": MODEL_ID,
                "messages": [dict(message) for message in vector.messages],
                "max_tokens": COEVAL_MAX_TOKENS,
                "temperature": 0,
            },
        )
    except Exception as error:  # noqa: BLE001 - the report exposes type only
        return _result(
            vector.case_id,
            started,
            status=None,
            passed=False,
            reason=f"transport_{type(error).__name__}",
        )
    if response.status_code != 200:
        return _result(
            vector.case_id,
            started,
            status=response.status_code,
            passed=False,
            reason="non_200",
        )
    passed, reason = _valid_chat_structure(response)
    return _result(
        vector.case_id,
        started,
        status=response.status_code,
        passed=passed,
        reason=reason,
    )


async def run_live_http_gate(
    base_url: str,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Run the fixed C=16 live HTTP matrix once with no internal retry."""

    normalized_url = _validate_loopback_base_url(base_url)
    own_client = client is None
    active_client = client or httpx.AsyncClient(
        base_url=normalized_url,
        timeout=COEVAL_TIMEOUT_SECONDS,
        limits=httpx.Limits(
            max_connections=COEVAL_CONCURRENCY,
            max_keepalive_connections=COEVAL_CONCURRENCY,
        ),
        trust_env=False,
        follow_redirects=False,
    )
    start_event = asyncio.Event()
    tasks = [
        asyncio.create_task(_run_live_vector(active_client, vector, start_event, api_key))
        for vector in LIVE_VECTORS
    ]
    started = time.perf_counter()
    start_event.set()
    try:
        results = list(await asyncio.gather(*tasks))
    finally:
        if own_client:
            await active_client.aclose()
    wall_ms = round((time.perf_counter() - started) * 1_000, 1)
    status_counts = Counter(
        "transport_error" if item.status is None else str(item.status) for item in results
    )
    nonempty_count = sum(item.passed for item in results)
    return {
        "phase": "live_http_c16",
        "passed": nonempty_count == len(LIVE_VECTORS),
        "vector_count": len(LIVE_VECTORS),
        "concurrency": COEVAL_CONCURRENCY,
        "attempts_per_vector": GATE_ATTEMPTS_PER_VECTOR,
        "runner_retry_count": 0,
        "route_shape_counts": dict(
            sorted(Counter(vector.route_class for vector in LIVE_VECTORS).items())
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "nonempty_count": nonempty_count,
        "nonempty_rate": round(nonempty_count / len(LIVE_VECTORS), 4),
        "wall_ms": wall_ms,
        "latency_ms": _latency_summary(results),
        "failures": [asdict(item) for item in results if not item.passed],
        "response_content_persisted": False,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_coeval_checkout(root: Path, *, require_imports: bool) -> dict[str, Any]:
    """Verify only the public pin, PassthroughClient, val config, and lock file."""

    resolved = root.resolve(strict=True)
    completed = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    targets = {
        "passthrough_client": (
            resolved / "src/coeval/clients/passthrough.py",
            COEVAL_PASSTHROUGH_SHA256,
        ),
        "conquer_val_config": (
            resolved / "src/coeval/conf/datasets/conquer_val.yaml",
            COEVAL_CONQUER_VAL_SHA256,
        ),
        "uv_lock": (resolved / "uv.lock", COEVAL_UV_LOCK_SHA256),
    }
    hashes = {name: _file_sha256(path) for name, (path, _) in targets.items()}
    hash_matches = {
        name: hashes[name] == expected for name, (_, expected) in targets.items()
    }
    imports_available = False
    import_error_type: str | None = None
    if require_imports and head == COEVAL_COMMIT and all(hash_matches.values()):
        sys.path.insert(0, str(resolved / "src"))
        try:
            importlib.import_module("coeval.clients.passthrough")
            importlib.import_module("coeval.llm.config")
            importlib.import_module("coeval.llm.factory")
            imports_available = True
        except Exception as error:  # noqa: BLE001 - report exposes type only
            import_error_type = type(error).__name__
        finally:
            sys.path.pop(0)
    elif not require_imports:
        imports_available = True
    passed = head == COEVAL_COMMIT and all(hash_matches.values()) and imports_available
    return {
        "phase": "coeval_pin_preflight",
        "passed": passed,
        "commit": head,
        "expected_commit": COEVAL_COMMIT,
        "hash_matches": hash_matches,
        "official_imports_available": imports_available,
        "import_error_type": import_error_type,
        "inspected_scope": ["passthrough_client", "conquer_val_config", "uv_lock"],
        "held_out_accessed": False,
    }


async def run_official_coeval_gate(
    base_url: str,
    coeval_root: Path,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send two calls through the exact pinned official PassthroughClient."""

    normalized_url = _validate_loopback_base_url(base_url)
    preflight = verify_coeval_checkout(coeval_root, require_imports=True)
    if not preflight["passed"]:
        return {
            "phase": "official_coeval_passthrough",
            "passed": False,
            "preflight": preflight,
            "vector_count": 0,
            "failures": [{"case_id": "preflight", "reason": "preflight_failed"}],
        }

    source_path = str(coeval_root.resolve() / "src")
    sys.path.insert(0, source_path)
    try:
        passthrough_module = importlib.import_module("coeval.clients.passthrough")
        config_module = importlib.import_module("coeval.llm.config")
        factory_module = importlib.import_module("coeval.llm.factory")
        config = config_module.LLMConfig(
            api_base=f"{normalized_url}/v1",
            model=MODEL_ID,
            api_key=api_key,
            temperature=0.0,
            max_tokens=COEVAL_MAX_TOKENS,
            timeout=COEVAL_TIMEOUT_SECONDS,
            max_retries=0,
            is_function_calling_model=False,
        )
        official_client = passthrough_module.PassthroughClient(
            llm=factory_module.create_llm_client(config)
        )
        official_vectors = (LIVE_VECTORS[0], LIVE_VECTORS[-1])

        async def invoke(vector: LiveVector) -> CaseResult:
            started = time.perf_counter()
            try:
                content = await official_client.generate(
                    [dict(message) for message in vector.messages]
                )
            except Exception as error:  # noqa: BLE001 - report exposes type only
                return _result(
                    vector.case_id,
                    started,
                    status=None,
                    passed=False,
                    reason=f"official_client_{type(error).__name__}",
                )
            passed = isinstance(content, str) and bool(content.strip())
            return _result(
                vector.case_id,
                started,
                status=200 if passed else None,
                passed=passed,
                reason="ok" if passed else "empty_content",
            )

        started = time.perf_counter()
        results = list(await asyncio.gather(*(invoke(item) for item in official_vectors)))
        wall_ms = round((time.perf_counter() - started) * 1_000, 1)
    finally:
        sys.path.pop(0)
    return {
        "phase": "official_coeval_passthrough",
        "passed": all(item.passed for item in results),
        "preflight": preflight,
        "vector_count": len(results),
        "attempts_per_vector": GATE_ATTEMPTS_PER_VECTOR,
        "runner_retry_count": 0,
        "wall_ms": wall_ms,
        "latency_ms": _latency_summary(results),
        "failures": [asdict(item) for item in results if not item.passed],
        "response_content_persisted": False,
    }


def _release_contract_metadata() -> dict[str, Any]:
    return {
        "coeval_commit": COEVAL_COMMIT,
        "official_conquer_val": {
            "examples": 301,
            "concurrency": COEVAL_CONCURRENCY,
            "inference_attempts": COEVAL_RUNNER_ATTEMPTS,
            "retry_delay_seconds": 2.0,
            "max_tokens": COEVAL_MAX_TOKENS,
            "timeout_seconds": COEVAL_TIMEOUT_SECONDS,
            "score_inference_failures_as_zero": True,
        },
        "gate_policy": {
            "internal_retries": 0,
            "max_live_calls": 18,
            "target_scope": "loopback_container_only",
            "questions_or_responses_logged": False,
        },
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    metadata = _release_contract_metadata()
    if args.phase == "boundary":
        report = await run_boundary_gate(args.base_url)
        return {"contract": metadata, "reports": [report], "passed": report["passed"]}

    if args.phase == "coeval-preflight":
        if args.coeval_root is None:
            raise ValueError("--coeval-root is required for coeval-preflight")
        report = verify_coeval_checkout(args.coeval_root, require_imports=True)
        return {"contract": metadata, "reports": [report], "passed": report["passed"]}

    if not args.authorized_live:
        raise ValueError("live phases require explicit --authorized-live")
    api_key = _read_optional_api_key(args.api_key_env)
    if args.phase == "live-http":
        report = await run_live_http_gate(args.base_url, api_key=api_key)
        return {"contract": metadata, "reports": [report], "passed": report["passed"]}

    if args.coeval_root is None:
        raise ValueError("--coeval-root is required for official CoEval phases")
    if args.phase == "official-coeval":
        report = await run_official_coeval_gate(
            args.base_url,
            args.coeval_root,
            api_key=api_key,
        )
        return {"contract": metadata, "reports": [report], "passed": report["passed"]}

    http_report = await run_live_http_gate(args.base_url, api_key=api_key)
    if not http_report["passed"]:
        return {"contract": metadata, "reports": [http_report], "passed": False}
    official_report = await run_official_coeval_gate(
        args.base_url,
        args.coeval_root,
        api_key=api_key,
    )
    reports = [http_report, official_report]
    return {
        "contract": metadata,
        "reports": reports,
        "live_call_count": 18,
        "passed": all(report["passed"] for report in reports),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "boundary",
            "coeval-preflight",
            "live-http",
            "official-coeval",
            "release-live",
        ),
        required=True,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--coeval-root", type=Path)
    parser.add_argument(
        "--api-key-env",
        default="COEVAL_GATE_API_KEY",
        help="Environment variable name only; never pass a credential on the command line.",
    )
    parser.add_argument(
        "--authorized-live",
        action="store_true",
        help="Required acknowledgement before a phase that can trigger upstream L2/MCP.",
    )
    return parser


def main() -> int:
    # Third-party clients must not emit headers, prompts, or response bodies.
    # The gate's single structured JSON record is the complete release evidence.
    logging.disable(logging.CRITICAL)
    parser = _parser()
    args = parser.parse_args()
    try:
        report = asyncio.run(_run(args))
    except Exception as error:  # noqa: BLE001 - no exception details can leak secrets
        report = {
            "contract": _release_contract_metadata(),
            "passed": False,
            "preflight_error_type": type(error).__name__,
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
