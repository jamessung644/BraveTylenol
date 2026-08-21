#!/usr/bin/env python3
"""Bounded, aggregate-only CoEval completion smoke checker."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODEL_ID = "team-chatbot"
MAX_LATENCY_SECONDS = 165.0
PREFLIGHT_TIMEOUT_SECONDS = 5.0
MAX_REQUESTS = 256
MAX_CONCURRENCY = 32
_PROCESS_JOIN_SECONDS = 0.1
_BATCH_OVERHEAD_SECONDS = 2.0
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


def validate_completion(
    payload: Any,
    latency_seconds: float,
    deadline_seconds: float = MAX_LATENCY_SECONDS,
) -> tuple[bool, bool]:
    """Validate only the public response envelope and return (success, fallback)."""
    if latency_seconds > deadline_seconds or not isinstance(payload, dict):
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


def _read_json_worker(
    connection: Any,
    base_url: str,
    path: str,
    body: bytes | None,
    method: str,
    timeout_seconds: float,
) -> None:
    """Perform one potentially blocking HTTP read in a process the caller can terminate."""
    try:
        request = Request(
            f"{base_url.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method=method,
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - supplied CLI endpoint.
            status = response.status
            try:
                result = status, json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                result = status, None
    except HTTPError as error:
        result = error.code, None
    except Exception:
        result = 0, None
    try:
        connection.send(result)
    except Exception:
        pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _read_json(
    base_url: str,
    path: str,
    body: bytes | None,
    method: str,
    timeout_seconds: float,
) -> tuple[int, Any | None]:
    """Use a monotonic deadline around a killable HTTP worker process."""
    started = time.perf_counter()
    process = None
    receiver = None
    sender = None
    process_started = False
    result: tuple[int, Any | None] = (0, None)
    cleanup_failed = False
    try:
        receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
        process = multiprocessing.get_context("spawn").Process(
            target=_read_json_worker,
            args=(sender, base_url, path, body, method, timeout_seconds),
            daemon=True,
        )
        process.start()
        process_started = True
        sender.close()
        sender = None
        remaining = max(0.0, timeout_seconds - (time.perf_counter() - started))
        if receiver.poll(remaining):
            try:
                status, payload = receiver.recv()
                if isinstance(status, int):
                    result = status, payload
            except (EOFError, OSError):
                pass
    except Exception:
        result = 0, None
    finally:
        if sender is not None:
            try:
                sender.close()
            except Exception:
                cleanup_failed = True
        if receiver is not None:
            try:
                receiver.close()
            except Exception:
                cleanup_failed = True
        if process is not None and process_started:
            alive: bool | None = None
            try:
                alive = process.is_alive()
            except (AssertionError, OSError):
                cleanup_failed = True
            if alive is not False:
                try:
                    process.terminate()
                except (AssertionError, OSError):
                    cleanup_failed = True
            try:
                process.join(_PROCESS_JOIN_SECONDS)
            except (AssertionError, OSError):
                cleanup_failed = True
            alive_after: bool | None = None
            try:
                alive_after = process.is_alive()
            except (AssertionError, OSError):
                cleanup_failed = True
            if alive_after is not False:
                try:
                    process.kill()
                except (AssertionError, OSError):
                    cleanup_failed = True
            try:
                process.join(_PROCESS_JOIN_SECONDS)
            except (AssertionError, OSError):
                cleanup_failed = True
            final_alive: bool | None = None
            try:
                final_alive = process.is_alive()
            except (AssertionError, OSError):
                cleanup_failed = True
            if final_alive is not False:
                cleanup_failed = True
    return (0, None) if cleanup_failed else result


def _get_json(base_url: str, path: str, timeout_seconds: float) -> tuple[int, Any | None]:
    return _read_json(base_url, path, body=None, method="GET", timeout_seconds=timeout_seconds)


def check_preflight(base_url: str, timeout_seconds: float = PREFLIGHT_TIMEOUT_SECONDS) -> list[int]:
    health_status, _ = _get_json(base_url, "/health", timeout_seconds)
    models_status, models = _get_json(base_url, "/v1/models", timeout_seconds)
    models_valid = (
        models_status == 200
        and isinstance(models, dict)
        and isinstance(models.get("data"), list)
        and any(isinstance(item, dict) and item.get("id") == MODEL_ID for item in models["data"])
    )
    return [] if health_status == 200 and models_valid else [health_status, models_status]


def send_completion(base_url: str, deadline_seconds: float = MAX_LATENCY_SECONDS) -> RequestResult:
    started = time.perf_counter()
    try:
        body = json.dumps(REQUEST_PAYLOAD, ensure_ascii=False).encode("utf-8")
        status, payload = _read_json(
            base_url,
            "/v1/chat/completions",
            body=body,
            method="POST",
            timeout_seconds=deadline_seconds,
        )
    except Exception:
        status, payload = 0, None
    latency = time.perf_counter() - started
    if status != 200:
        return RequestResult(status=status, latency_seconds=latency, success=False, fallback=False)
    success, fallback = validate_completion(payload, latency, deadline_seconds)
    return RequestResult(status=status, latency_seconds=latency, success=success, fallback=fallback)


def _failed_result(started: float) -> RequestResult:
    return RequestResult(
        status=0,
        latency_seconds=time.perf_counter() - started,
        success=False,
        fallback=False,
    )


def run_completions(
    base_url: str,
    requests: int,
    concurrency: int,
    deadline_seconds: float,
) -> list[RequestResult]:
    """Bound total wait while each network operation has its own killable absolute deadline."""
    started = time.perf_counter()
    workers = effective_workers(requests, concurrency)
    batch_budget = math.ceil(requests / workers) * deadline_seconds + _BATCH_OVERHEAD_SECONDS
    batch_deadline = started + batch_budget
    executor = None
    futures = []
    try:
        executor = ThreadPoolExecutor(max_workers=workers)
        for _ in range(requests):
            futures.append(executor.submit(send_completion, base_url, deadline_seconds))
    except Exception:
        for future in futures:
            try:
                future.cancel()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        return [_failed_result(started) for _ in range(requests)]
    results: list[RequestResult] = []
    shutdown_failed = False
    try:
        for future in futures:
            remaining = batch_deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                results.append(future.result(timeout=remaining))
            except Exception:
                results.append(_failed_result(started))
    finally:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            shutdown_failed = True
    if shutdown_failed:
        return [_failed_result(started) for _ in range(requests)]
    results.extend(_failed_result(started) for _ in range(requests - len(results)))
    return results


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


def _bounded_positive(value: str, maximum: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= maximum else None


def _deadline(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not 0 < parsed <= MAX_LATENCY_SECONDS:
        return 0.0
    return parsed


def effective_workers(requests: int, concurrency: int) -> int:
    return min(requests, concurrency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate concurrent public completion envelopes.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--requests", default="16")
    parser.add_argument("--concurrency", default="16")
    parser.add_argument("--deadline-seconds", default=str(MAX_LATENCY_SECONDS))
    raw = parser.parse_args()
    requests = _bounded_positive(raw.requests, MAX_REQUESTS)
    concurrency = _bounded_positive(raw.concurrency, MAX_CONCURRENCY)
    deadline_seconds = _deadline(raw.deadline_seconds)
    return argparse.Namespace(
        base_url=raw.base_url,
        requests=requests,
        concurrency=concurrency,
        deadline_seconds=deadline_seconds,
        valid=requests is not None and concurrency is not None and deadline_seconds > 0,
    )


def main() -> int:
    args = parse_args()
    if not args.valid:
        print_summary(
            {
                "requests": 0,
                "success": 0,
                "fallback": 0,
                "failure": 1,
                "status_counts": {0: 1},
                "latencies": [],
            }
        )
        return 1
    preflight_failures = check_preflight(
        args.base_url,
        timeout_seconds=min(PREFLIGHT_TIMEOUT_SECONDS, args.deadline_seconds),
    )
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
    results = run_completions(
        args.base_url,
        requests=args.requests,
        concurrency=args.concurrency,
        deadline_seconds=args.deadline_seconds,
    )
    summary = aggregate(results)
    print_summary(summary)
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
