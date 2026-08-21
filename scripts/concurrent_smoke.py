#!/usr/bin/env python3
"""Bounded, aggregate-only CoEval completion smoke checker."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_ID = "team-chatbot"
MAX_LATENCY_SECONDS = 165.0
SAFETY_FALLBACK = (
    "질문을 확인했습니다. 증상이 심하거나 갑자기 악화되면 즉시 119 또는 "
    "응급실의 도움을 받고, 정확한 판단을 위해 의료 전문가와 상담해 주세요."
)
REQUEST_PAYLOAD = {
    "model": MODEL_ID,
    "messages": [
        {"role": "user", "content": "감기 증상으로 약을 복용 중입니다."},
        {
            "role": "assistant",
            "content": "복용 중인 약과 증상을 확인한 뒤 의료진 또는 약사와 상담하세요.",
        },
        {
            "role": "user",
            "content": "그렇다면 아세트아미노펜을 함께 복용할 때 주의할 점은 무엇인가요?",
        },
    ],
}


@dataclass(frozen=True)
class RequestResult:
    status: int
    latency_seconds: float
    success: bool
    fallback: bool


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_completion(payload: Any, latency_seconds: float) -> tuple[bool, bool]:
    """Validate only the public response envelope and return (success, fallback)."""
    if latency_seconds > MAX_LATENCY_SECONDS or not isinstance(payload, dict):
        return False, False
    if payload.get("model") != MODEL_ID:
        return False, False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False, False
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        return False, False
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not all(
        _nonnegative_int(usage.get(name))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return False, False
    return True, message["content"].strip() == SAFETY_FALLBACK


def _read_json(request: Request, timeout: float) -> tuple[int, Any | None]:
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller supplies local URL.
            status = response.status
            try:
                return status, json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return status, None
    except HTTPError as error:
        return error.code, None
    except (TimeoutError, URLError, OSError):
        return 0, None


def _get_json(base_url: str, path: str) -> tuple[int, Any | None]:
    return _read_json(Request(f"{base_url.rstrip('/')}{path}", method="GET"), timeout=5.0)


def check_preflight(base_url: str) -> list[int]:
    health_status, _ = _get_json(base_url, "/health")
    models_status, models = _get_json(base_url, "/v1/models")
    models_valid = (
        models_status == 200
        and isinstance(models, dict)
        and isinstance(models.get("data"), list)
        and any(isinstance(item, dict) and item.get("id") == MODEL_ID for item in models["data"])
    )
    return [] if health_status == 200 and models_valid else [health_status, models_status]


def send_completion(base_url: str) -> RequestResult:
    started = time.perf_counter()
    body = json.dumps(REQUEST_PAYLOAD, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, payload = _read_json(request, timeout=MAX_LATENCY_SECONDS)
    latency = time.perf_counter() - started
    if status != 200:
        return RequestResult(status=status, latency_seconds=latency, success=False, fallback=False)
    success, fallback = validate_completion(payload, latency)
    return RequestResult(status=status, latency_seconds=latency, success=success, fallback=fallback)


def aggregate(results: list[RequestResult]) -> dict[str, Any]:
    return {
        "requests": len(results),
        "success": sum(result.success for result in results),
        "fallback": sum(result.fallback for result in results),
        "failure": sum(not result.success for result in results),
        "status_counts": dict(Counter(result.status for result in results)),
        "latencies": [result.latency_seconds for result in results],
    }


def exit_code(summary: dict[str, Any]) -> int:
    return int(summary["failure"] != 0 or summary["success"] != summary["requests"])


def print_summary(summary: dict[str, Any]) -> None:
    latencies = summary["latencies"] or [0.0]
    statuses = ",".join(
        f"{status}:{count}" for status, count in sorted(summary["status_counts"].items())
    ) or "none"
    print(
        "requests={requests} success={success} fallback={fallback} failure={failure}".format(
            **summary
        )
    )
    print(f"http_status_counts={statuses}")
    median = statistics.median(latencies)
    print(
        "latency_seconds="
        f"min={min(latencies):.3f} median={median:.3f} max={max(latencies):.3f}"
    )


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate concurrent public completion envelopes.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--requests", type=_positive, default=16)
    parser.add_argument("--concurrency", type=_positive, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preflight_failures = check_preflight(args.base_url)
    if preflight_failures:
        summary = {
            "requests": 0,
            "success": 0,
            "fallback": 0,
            "failure": 1,
            "status_counts": dict(Counter(preflight_failures)),
            "latencies": [],
        }
        print_summary(summary)
        return 1
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(lambda _: send_completion(args.base_url), range(args.requests)))
    summary = aggregate(results)
    print_summary(summary)
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
