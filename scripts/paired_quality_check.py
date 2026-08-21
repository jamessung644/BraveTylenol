#!/usr/bin/env python3
"""Aggregate-only paired direct versus score-first completion checker."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, Any | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=MAX_LATENCY_SECONDS) as response:  # noqa: S310 - explicit CLI URL.
            status = response.status
            try:
                return status, json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return status, None
    except HTTPError as error:
        return error.code, None
    except Exception:
        return 0, None


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


def write_jsonl(output: Path, records: list[dict[str, str | None]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if descriptor != -1:
            os.close(descriptor)


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


def print_summary(summary: dict[str, Any]) -> None:
    direct = summary["direct"]
    score = summary["score"]
    print(
        "paired_requests={requests} direct_success={direct_success} score_success={score_success} "
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
    arguments = parse_args()
    records, summary = run_check(arguments.direct_url, arguments.score_url)
    try:
        write_jsonl(arguments.output, records)
    except OSError:
        print_summary(
            {
                "requests": 0,
                "direct": _aggregate([]),
                "score": _aggregate([]),
                "coverage": {},
            }
        )
        return 1
    print_summary(summary)
    return int(summary["direct"]["failure"] != 0 or summary["score"]["failure"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
