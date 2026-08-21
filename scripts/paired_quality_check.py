#!/usr/bin/env python3
"""Aggregate-only paired direct versus score-first completion checker."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from absolute_http import post_json
except ModuleNotFoundError:
    from scripts.absolute_http import post_json

MODEL_ID = "team-chatbot"
MAX_LATENCY_SECONDS = 165.0
SAFETY_FALLBACK = (
    "질문을 확인했습니다. 증상이 심하거나 갑자기 악화되면 즉시 119 또는 "
    "응급실의 도움을 받고, 정확한 판단을 위해 의료 전문가와 상담해 주세요."
)
MANUAL_RUBRIC = (
    "correctness",
    "relevance",
    "safety",
    "actionable_next_steps",
    "appropriate_uncertainty",
    "source_grounding",
)
FIXED_SCENARIOS = (
    (
        "drug_indication_safety",
        "drug_safety",
        (
            {
                "role": "user",
                "content": "아세트아미노펜의 적응증과 주요 부작용, 복용 시 주의점을 알려주세요.",
            },
        ),
    ),
    (
        "emergency_triage",
        "emergency",
        (
            {
                "role": "user",
                "content": "갑작스러운 흉통과 호흡곤란이 있을 때 지금 무엇을 해야 하나요?",
            },
        ),
    ),
    (
        "guideline_recommendation",
        "guideline",
        (
            {
                "role": "user",
                "content": (
                    "성인 고혈압의 생활습관 관리에 대한 최신 가이드라인 권고를 요약해 주세요."
                ),
            },
        ),
    ),
    (
        "reimbursement",
        "reimbursement",
        (
            {
                "role": "user",
                "content": "이 치료가 건강보험 급여 대상인지 확인할 때 어떤 기준을 봐야 하나요?",
            },
        ),
    ),
    (
        "korean_law",
        "law",
        (
            {
                "role": "user",
                "content": "한국 의료법상 진료기록 열람과 관련한 기본 원칙을 알려주세요.",
            },
        ),
    ),
    (
        "general_health",
        "general_health",
        (
            {
                "role": "user",
                "content": "감기 증상이 있을 때 집에서 할 수 있는 일반적인 관리 방법은 무엇인가요?",
            },
        ),
    ),
    (
        "pronoun_follow_up",
        "drug",
        (
            {"role": "user", "content": "두통 때문에 아세트아미노펜을 복용하려고 합니다."},
            {
                "role": "assistant",
                "content": "다른 복용 약과 간 질환 여부를 의료진 또는 약사와 확인하세요.",
            },
            {"role": "user", "content": "그 약을 복용할 때 특히 주의할 점은 무엇인가요?"},
        ),
    ),
)


@dataclass(frozen=True)
class EndpointResult:
    status: int
    latency_seconds: float
    answer: str | None
    success: bool
    fallback: bool


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_completion(payload: Any) -> tuple[str | None, bool]:
    """Return a nonblank answer only for the public completion contract."""
    if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
        return None, False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None, False
    choice = choices[0]
    message = choice.get("message")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
    ):
        return None, False
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not all(
        _nonnegative_int(usage.get(name))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return None, False
    answer = message["content"].strip()
    return answer, answer == SAFETY_FALLBACK


def _post_json(
    url: Any,
    payload: Any,
    timeout_seconds: float = MAX_LATENCY_SECONDS,
) -> tuple[int, Any | None]:
    try:
        endpoint = url.rstrip("/") + "/v1/chat/completions"
    except Exception:
        return 0, None
    return post_json(
        endpoint,
        payload,
        {"Content-Type": "application/json"},
        timeout_seconds,
    )


def check_endpoint(url: str, messages: tuple[dict[str, str], ...]) -> EndpointResult:
    started = time.perf_counter()
    status, payload = _post_json(url, {"model": MODEL_ID, "messages": list(messages)})
    latency = time.perf_counter() - started
    answer, fallback = validate_completion(payload)
    return EndpointResult(
        status=status,
        latency_seconds=latency,
        answer=answer,
        success=status == 200 and answer is not None and latency <= MAX_LATENCY_SECONDS,
        fallback=fallback if status == 200 and latency <= MAX_LATENCY_SECONDS else False,
    )


def output_path_allowed(output: Path, repository_root: Path) -> bool:
    """Prevent raw paired answers from becoming an ordinary tracked repository file."""
    resolved_output = output.resolve(strict=False)
    resolved_root = repository_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return True
    try:
        checked = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(resolved_output)],
            cwd=resolved_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return checked.returncode == 0


def write_jsonl(output: Path, records: list[dict[str, str | None]]) -> bool:
    """Atomically write UTF-8-safe paired answers or reject the whole artifact."""
    descriptor = -1
    temporary_path: Path | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".paired-", dir=output.parent)
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                )
        os.replace(temporary_path, output)
        temporary_path = None
        os.chmod(output, 0o600)
        return True
    except Exception:
        return False
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def run_check(
    direct_url: str, score_url: str
) -> tuple[list[dict[str, str | None]], dict[str, Any]]:
    records: list[dict[str, str | None]] = []
    direct_results: list[EndpointResult] = []
    score_results: list[EndpointResult] = []
    coverage: Counter[str] = Counter()
    for scenario, domain, messages in FIXED_SCENARIOS:
        direct = check_endpoint(direct_url, messages)
        score = check_endpoint(score_url, messages)
        direct_results.append(direct)
        score_results.append(score)
        coverage[domain] += 1
        records.append(
            {
                "scenario": scenario,
                "domain": domain,
                "direct_answer": direct.answer,
                "score_answer": score.answer,
            }
        )
    return records, {
        "requests": len(FIXED_SCENARIOS),
        "direct": _aggregate(direct_results),
        "score": _aggregate(score_results),
        "coverage": dict(sorted(coverage.items())),
    }


def _aggregate(results: list[EndpointResult]) -> dict[str, Any]:
    total = len(results)
    latencies = [result.latency_seconds for result in results] or [0.0]
    return {
        "success": sum(result.success for result in results),
        "fallback": sum(result.fallback for result in results),
        "failure": sum(not result.success for result in results),
        "statuses": dict(Counter(result.status for result in results)),
        "completion_rate": sum(result.success for result in results) / total if total else 0.0,
        "fallback_rate": sum(result.fallback for result in results) / total if total else 0.0,
        "error_rate": sum(not result.success for result in results) / total if total else 0.0,
        "latency_min": min(latencies),
        "latency_median": statistics.median(latencies),
        "latency_max": max(latencies),
    }


def print_summary(summary: dict[str, Any]) -> bool:
    direct = summary["direct"]
    score = summary["score"]
    try:
        print(
            "paired_requests={requests} direct_success={direct_success} "
            "score_success={score_success} "
            "direct_failure={direct_failure} score_failure={score_failure}".format(
                requests=summary["requests"],
                direct_success=direct["success"],
                score_success=score["success"],
                direct_failure=direct["failure"],
                score_failure=score["failure"],
            )
        )
        print(
            "completion_rate=direct:{:.3f},score:{:.3f} fallback_rate=direct:{:.3f},score:{:.3f} "
            "error_rate=direct:{:.3f},score:{:.3f}".format(
                direct["completion_rate"],
                score["completion_rate"],
                direct["fallback_rate"],
                score["fallback_rate"],
                direct["error_rate"],
                score["error_rate"],
            )
        )
        print(
            "latency_seconds=direct:min={:.3f},median={:.3f},max={:.3f} "
            "score:min={:.3f},median={:.3f},max={:.3f}".format(
                direct["latency_min"],
                direct["latency_median"],
                direct["latency_max"],
                score["latency_min"],
                score["latency_median"],
                score["latency_max"],
            )
        )
        coverage = ",".join(f"{name}:{count}" for name, count in summary["coverage"].items())
        print("domain_coverage=" + coverage)
        return True
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def _url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("endpoint must be an HTTP(S) URL")
    return value.rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed paired completion checks.")
    parser.add_argument("--direct-url", required=True, type=_url)
    parser.add_argument("--score-url", required=True, type=_url)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.direct_url == arguments.score_url:
        parser.error("direct and score endpoints must be distinct")
    repository = Path(__file__).resolve().parents[1]
    if not output_path_allowed(arguments.output, repository):
        parser.error("output must be outside the repository or ignored")
    return arguments


def main() -> int:
    try:
        arguments = parse_args()
        records, summary = run_check(arguments.direct_url, arguments.score_url)
        wrote_output = write_jsonl(arguments.output, records)
    except (Exception, KeyboardInterrupt):
        wrote_output = False
        summary = _empty_summary()
    if not wrote_output:
        print_summary(_empty_summary())
        return 1
    if not print_summary(summary):
        return 1
    return int(summary["direct"]["failure"] != 0 or summary["score"]["failure"] != 0)


def _empty_summary() -> dict[str, Any]:
    return {
        "requests": 0,
        "direct": _aggregate([]),
        "score": _aggregate([]),
        "coverage": {},
    }


if __name__ == "__main__":
    raise SystemExit(main())
