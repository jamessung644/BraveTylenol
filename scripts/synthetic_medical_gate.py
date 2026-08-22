#!/usr/bin/env python3
"""Run an append-only, dataset-independent synthetic medical response gate.

Only a loopback OpenAI-compatible ``/v1/chat/completions`` endpoint is allowed.
Response text is evaluated in memory and is never included in the returned or
written report.  The report is intentionally limited to identifiers, pass
flags, HTTP status, latency, content length, and predeclared route expectations.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPOSITORY_ROOT / "benchmarks/fixtures/synthetic_medical_gate_v1.json"
DEFAULT_MODEL = "team-chatbot"
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_TOKENS = 2_048
MAX_CONCURRENCY = 8
REPORT_SCHEMA_VERSION = "synthetic-medical-live-report-v1"
_ALLOWED_ROUTES = frozenset({"direct", "emergency", "rag"})
_ALLOWED_DIMENSIONS = frozenset(
    {"accuracy", "completeness", "context", "communication", "instruction_adherence"}
)
_SAFE_REPORT_STATUSES = frozenset(
    {
        "ok",
        "http_error",
        "request_error",
        "invalid_response",
        "empty_response",
        "invariant_failure",
    }
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    primary_category: str
    expected_route: str
    passed: bool
    status: str
    http_status: int | None
    latency_ms: float
    content_length: int
    invariant_flags: Mapping[str, bool]
    failed_invariants: tuple[str, ...]


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
            address = ipaddress.ip_address(host)
        except ValueError as error:
            raise ValueError("gate target must be loopback") from error
        if not address.is_loopback:
            raise ValueError("gate target must be loopback")
    return base_url.rstrip("/")


def _read_api_key(environment_name: str) -> str | None:
    value = os.getenv(environment_name)
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if len(normalized) > 4_096 or any(ord(character) < 0x20 for character in normalized):
        raise ValueError(f"{environment_name} is not a valid single-line secret")
    return normalized


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _fixture_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict):
        raise ValueError("fixture root must be an object")
    required_top_level = {
        "schema_version",
        "fixture_series",
        "generation",
        "generation_status",
        "append_only_policy",
        "category_contract",
        "concepts",
        "cases",
    }
    missing = sorted(required_top_level - fixture.keys())
    if missing:
        raise ValueError(f"fixture is missing required fields: {', '.join(missing)}")
    if fixture["generation_status"] != "frozen_before_observation":
        raise ValueError("fixture generation must be frozen before observation")
    if not isinstance(fixture["generation"], int) or fixture["generation"] < 1:
        raise ValueError("fixture generation must be a positive integer")
    append_only_policy = fixture["append_only_policy"]
    if not isinstance(append_only_policy, dict) or not all(
        append_only_policy.get(flag) is True
        for flag in (
            "case_ids_are_permanent",
            "observed_case_semantics_must_not_change",
            "observed_category_assignments_must_not_change",
            "new_failures_create_new_case_ids",
            "extensions_require_a_new_frozen_generation",
        )
    ):
        raise ValueError("fixture must declare the complete append-only policy")

    concepts = fixture["concepts"]
    cases = fixture["cases"]
    if not isinstance(concepts, dict) or not concepts:
        raise ValueError("fixture concepts must be a nonempty object")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture cases must be a nonempty array")

    expected_counts = fixture["category_contract"]
    observed_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every fixture case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every fixture case requires a nonempty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate fixture case id: {case_id}")
        seen_ids.add(case_id)
        introduced = case.get("introduced_in_generation")
        if not isinstance(introduced, int) or not 1 <= introduced <= fixture["generation"]:
            raise ValueError(f"{case_id} has an invalid introduced_in_generation")
        category = case.get("primary_category")
        if not isinstance(category, str):
            raise ValueError(f"{case_id} requires a primary_category")
        observed_counts[category] += 1
        if case.get("expected_route") not in _ALLOWED_ROUTES:
            raise ValueError(f"{case_id} has an invalid expected_route")
        dimensions = case.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"{case_id} requires dimensions")
        if not set(dimensions).issubset(_ALLOWED_DIMENSIONS):
            raise ValueError(f"{case_id} has an unsupported dimension")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case_id} requires messages")
        if not any(
            isinstance(message, dict) and message.get("role") == "user" for message in messages
        ):
            raise ValueError(f"{case_id} requires a user message")
        invariants = case.get("invariants")
        if not isinstance(invariants, dict):
            raise ValueError(f"{case_id} requires invariants")
        concept_ids = invariants.get("required_concepts", []) + invariants.get(
            "forbidden_concepts", []
        )
        unknown = sorted(set(concept_ids) - concepts.keys())
        if unknown:
            raise ValueError(f"{case_id} references unknown concepts: {', '.join(unknown)}")

    if dict(sorted(observed_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError("fixture category counts do not match category_contract")
    observed_dimensions = {dimension for case in cases for dimension in case["dimensions"]}
    if observed_dimensions != _ALLOWED_DIMENSIONS:
        raise ValueError("fixture does not cover every declared evaluation dimension")
    return fixture


def _concept_present(content: str, concept: Mapping[str, Any]) -> bool:
    patterns = concept.get("any_patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("concept any_patterns must be a nonempty array")
    return any(
        re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns
    )


def _check_json_object(content: str, format_spec: Mapping[str, Any]) -> bool:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    expected_keys = format_spec.get("exact_keys")
    if not isinstance(expected_keys, list) or set(value) != set(expected_keys):
        return False
    return all(
        (isinstance(item, str) and bool(item.strip()))
        or (isinstance(item, list) and bool(item) and all(str(entry).strip() for entry in item))
        for item in value.values()
    )


def _check_labeled_bullets(content: str, format_spec: Mapping[str, Any]) -> bool:
    labels = format_spec.get("labels")
    if not isinstance(labels, list) or not labels:
        return False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != len(labels):
        return False
    observed: list[str] = []
    for line in lines:
        match = re.fullmatch(r"[-*]\s+([^:]+):\s+.+", line)
        if match is None:
            return False
        observed.append(match.group(1).strip().casefold())
    return observed == [str(label).strip().casefold() for label in labels]


def _check_format(content: str, format_spec: Mapping[str, Any]) -> bool:
    kind = format_spec.get("kind", "none")
    if kind == "none":
        return True
    if kind == "json_object":
        return _check_json_object(content, format_spec)
    if kind == "labeled_bullets":
        return _check_labeled_bullets(content, format_spec)
    raise ValueError(f"unsupported format invariant kind: {kind}")


def evaluate_content(
    content: str,
    case: Mapping[str, Any],
    concepts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], tuple[str, ...]]:
    """Evaluate text without returning or persisting it."""

    invariants = case["invariants"]
    flags: dict[str, bool] = {
        "nonempty": bool(content.strip()),
        "minimum_length": len(content) >= int(invariants["min_content_chars"]),
        "maximum_length": len(content) <= int(invariants["max_content_chars"]),
    }
    for concept_id in invariants.get("required_concepts", []):
        flags[f"required:{concept_id}"] = _concept_present(content, concepts[concept_id])
    for concept_id in invariants.get("forbidden_concepts", []):
        flags[f"forbidden:{concept_id}"] = not _concept_present(content, concepts[concept_id])
    flags["format"] = _check_format(content, invariants.get("format", {"kind": "none"}))
    return flags, tuple(flag for flag, passed in flags.items() if not passed)


def _extract_content(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


async def _run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: Mapping[str, Any],
    concepts: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
    max_tokens: int,
    headers: Mapping[str, str],
) -> CaseResult:
    started = time.perf_counter()
    http_status: int | None = None
    content_length = 0
    flags: dict[str, bool] = {"transport": False}
    failed: tuple[str, ...] = ("transport",)
    status = "request_error"
    try:
        async with semaphore:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": case["messages"],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
        http_status = response.status_code
        if response.status_code != 200:
            status = "http_error"
        else:
            content = _extract_content(response)
            if content is None:
                status = "invalid_response"
            else:
                content_length = len(content)
                flags, failed = evaluate_content(content, case, concepts)
                if not content.strip():
                    status = "empty_response"
                elif failed:
                    status = "invariant_failure"
                else:
                    status = "ok"
    except (TimeoutError, httpx.HTTPError):
        # Exception strings can contain URLs, headers, or body fragments.  The
        # safe report records only this fixed status vocabulary.
        status = "request_error"
    latency_ms = round((time.perf_counter() - started) * 1_000, 1)
    if status not in _SAFE_REPORT_STATUSES:
        status = "request_error"
    return CaseResult(
        case_id=case["id"],
        primary_category=case["primary_category"],
        expected_route=case["expected_route"],
        passed=status == "ok",
        status=status,
        http_status=http_status,
        latency_ms=latency_ms,
        content_length=content_length,
        invariant_flags=flags,
        failed_invariants=failed,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 1)


def _build_report(
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    concurrency: int,
    results: Sequence[CaseResult],
) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    statuses = Counter(result.status for result in results)
    categories = Counter(result.primary_category for result in results)
    routes = Counter(result.expected_route for result in results)
    passed = sum(result.passed for result in results)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "fixture": {
            "series": fixture["fixture_series"],
            "generation": fixture["generation"],
            "sha256": fixture_sha256,
            "generation_status": fixture["generation_status"],
        },
        "retention": {
            "request_messages_in_report": False,
            "response_bodies_in_report": False,
            "credentials_in_report": False,
        },
        "execution": {
            "target_scope": "loopback_only",
            "concurrency": concurrency,
            "attempts_per_case": 1,
            "actual_route_collection": "not_collected_use_sanitized_container_log",
        },
        "summary": {
            "passed": passed == len(results),
            "case_count": len(results),
            "passed_count": passed,
            "failed_count": len(results) - passed,
            "pass_rate": round(passed / len(results), 4) if results else 0.0,
            "status_counts": dict(sorted(statuses.items())),
            "category_counts": dict(sorted(categories.items())),
            "expected_route_counts": dict(sorted(routes.items())),
            "latency_ms": {
                "total": round(sum(latencies), 1),
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": round(max(latencies), 1) if latencies else 0.0,
            },
        },
        "cases": [asdict(result) for result in results],
    }


async def run_gate(
    base_url: str,
    fixture_path: Path = DEFAULT_FIXTURE,
    *,
    concurrency: int = 4,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    normalized_base_url = _validate_loopback_base_url(base_url)
    fixture = load_fixture(fixture_path)
    semaphore = asyncio.Semaphore(concurrency)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=normalized_base_url,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )
    try:
        tasks = [
            _run_case(
                client,
                semaphore,
                case,
                fixture["concepts"],
                model=model,
                max_tokens=max_tokens,
                headers=_headers(api_key),
            )
            for case in fixture["cases"]
        ]
        results = await asyncio.gather(*tasks)
    finally:
        if owns_client:
            await client.aclose()
    return _build_report(fixture, _fixture_sha256(fixture_path), concurrency, results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--concurrency", type=int, choices=range(1, MAX_CONCURRENCY + 1), default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="LUNIT_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = asyncio.run(
        run_gate(
            args.base_url,
            args.fixture,
            concurrency=args.concurrency,
            model=args.model,
            api_key=_read_api_key(args.api_key_env),
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_tokens,
        )
    )
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    if args.report is not None:
        args.report.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
