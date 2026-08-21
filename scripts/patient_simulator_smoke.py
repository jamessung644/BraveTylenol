#!/usr/bin/env python3
"""Bounded, aggregate-only patient-simulator smoke checker."""

from __future__ import annotations

import argparse
import copy
import os
import stat
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from absolute_http import post_json
except ModuleNotFoundError:
    from scripts.absolute_http import post_json

HARNESS_MODEL_ID = "team-chatbot"
SIMULATOR_MODEL_ID = "patient-simulator-ko"
SIMULATOR_URL = "https://patient.hackathon.lunit.io/v1/chat/completions"
REQUIRED_CONVERSATIONS = 5
MAX_ASSISTANT_TURNS = 3
MAX_LATENCY_SECONDS = 165.0
MAX_KEY_FILE_BYTES = 65_536


@dataclass
class ConversationResult:
    assistant_turns: int = 0
    status_failures: int = 0
    shape_failures: int = 0
    simulator_retries: int = 0
    simulator_restarts: int = 0
    repeated_user_stops: int = 0
    harness_latency_failures: int = 0
    harness_latencies: list[float] = field(default_factory=list)
    simulator_statuses: Counter[int] = field(default_factory=Counter)
    harness_statuses: Counter[int] = field(default_factory=Counter)
    duration_seconds: float = 0.0


def read_api_key_file(path: Path) -> str:
    """Read one bounded owner-only regular file descriptor without path reopens."""
    descriptor = -1
    try:
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "geteuid"):
            raise ValueError
        descriptor = os.open(os.fspath(path), os.O_RDONLY | os.O_NOFOLLOW)
        file_status = os.fstat(descriptor)
        invalid_mode = stat.S_IMODE(file_status.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        if (
            not stat.S_ISREG(file_status.st_mode)
            or file_status.st_uid != os.geteuid()
            or invalid_mode
            or not 0 < file_status.st_size <= MAX_KEY_FILE_BYTES
        ):
            raise ValueError
        data = bytearray()
        while len(data) <= MAX_KEY_FILE_BYTES:
            chunk = os.read(descriptor, MAX_KEY_FILE_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_KEY_FILE_BYTES:
            raise ValueError
        key = bytes(data).decode("utf-8").strip()
    except (OSError, UnicodeError, ValueError):
        raise ValueError("credential file is not usable") from None
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                raise ValueError("credential file is not usable") from None
    if not key:
        raise ValueError("credential file is not usable")
    return key


def _post_json(
    url: Any,
    payload: Any,
    headers: Any,
    timeout_seconds: float = MAX_LATENCY_SECONDS,
) -> tuple[int, Any | None]:
    try:
        return post_json(url, payload, headers, timeout_seconds)
    except Exception:
        return 0, None


def _completion_message(
    payload: Any,
    model: str,
    response_role: str,
    history_role: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("model") != model:
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message")
    usage = payload.get("usage")
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(message, dict)
        or message.get("role") != response_role
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        or not isinstance(usage, dict)
        or not all(
            isinstance(usage.get(name), int)
            and not isinstance(usage.get(name), bool)
            and usage[name] >= 0
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
    ):
        return None
    normalized_message = copy.deepcopy(message)
    if history_role is not None:
        normalized_message["role"] = history_role
    return normalized_message


def _simulator_turn(
    simulator_url: str,
    history: list[dict[str, Any]],
    api_key: str,
) -> tuple[int, Any | None, int]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"model": SIMULATOR_MODEL_ID, "messages": history}
    status, response = _post_json(simulator_url, payload, headers)
    if status != 502:
        return status, response, 0
    retry_status, retry_response = _post_json(simulator_url, payload, headers)
    return retry_status, retry_response, 1


def run_conversation(
    *,
    harness_url: str,
    simulator_url: str,
    api_key: str,
    max_turns: int,
    clock: Callable[[], float] = time.perf_counter,
) -> ConversationResult:
    """Run one restart-bounded simulated conversation without printing its contents."""
    started = clock()
    result = ConversationResult()
    history: list[dict[str, Any]] = []
    seen_user_messages: set[str] = set()
    restart_available = True
    while result.assistant_turns < max_turns:
        status, payload, retries = _simulator_turn(simulator_url, history, api_key)
        result.simulator_statuses[status] += 1
        result.simulator_retries += retries
        if status == 404 and restart_available:
            restart_available = False
            result.simulator_restarts += 1
            history = []
            seen_user_messages = set()
            continue
        if status != 200:
            result.status_failures += 1
            break
        user_message = _completion_message(
            payload,
            SIMULATOR_MODEL_ID,
            "assistant",
            history_role="user",
        )
        if user_message is None:
            result.shape_failures += 1
            break
        user_text = user_message["content"]
        if user_text in seen_user_messages:
            result.repeated_user_stops += 1
            break
        seen_user_messages.add(user_text)
        history.append(user_message)
        harness_started = clock()
        harness_status, harness_payload = _post_json(
            harness_url.rstrip("/") + "/v1/chat/completions",
            {"model": HARNESS_MODEL_ID, "messages": history},
            {"Content-Type": "application/json"},
        )
        harness_latency = clock() - harness_started
        result.harness_latencies.append(harness_latency)
        if harness_latency > MAX_LATENCY_SECONDS:
            result.harness_latency_failures += 1
        result.harness_statuses[harness_status] += 1
        if harness_status != 200:
            result.status_failures += 1
            break
        assistant_message = _completion_message(harness_payload, HARNESS_MODEL_ID, "assistant")
        if assistant_message is None:
            result.shape_failures += 1
            break
        history.append(assistant_message)
        result.assistant_turns += 1
    result.duration_seconds = clock() - started
    return result


def run_smoke(
    *,
    harness_url: str,
    api_key: str,
    conversations: int,
    max_turns: int,
) -> list[ConversationResult]:
    return [
        run_conversation(
            harness_url=harness_url,
            simulator_url=SIMULATOR_URL,
            api_key=api_key,
            max_turns=max_turns,
        )
        for _ in range(conversations)
    ]


def _status_counts(results: list[ConversationResult], name: str) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for result in results:
        counter.update(getattr(result, name))
    return dict(sorted(counter.items()))


def _format_status_counts(counts: dict[int, int]) -> str:
    return ",".join(f"{status}:{count}" for status, count in counts.items()) or "none"


def print_summary(results: list[ConversationResult]) -> bool:
    """Emit aggregate-only output or fail silently when the output device is unsafe."""
    try:
        durations = [result.duration_seconds for result in results] or [0.0]
        completed = sum(
            result.status_failures == 0 and result.shape_failures == 0 for result in results
        )
        assistant_turns = sum(result.assistant_turns for result in results)
        repeated_stops = sum(result.repeated_user_stops for result in results)
        status_failures = sum(result.status_failures for result in results)
        shape_failures = sum(result.shape_failures for result in results)
        retries = sum(result.simulator_retries for result in results)
        restarts = sum(result.simulator_restarts for result in results)
        harness_latency_failures = sum(result.harness_latency_failures for result in results)
        simulator_statuses = _format_status_counts(_status_counts(results, "simulator_statuses"))
        harness_statuses = _format_status_counts(_status_counts(results, "harness_statuses"))
        harness_latencies = [latency for result in results for latency in result.harness_latencies]
        minimum_harness_latency = min(harness_latencies) if harness_latencies else 0.0
        median_harness_latency = statistics.median(harness_latencies) if harness_latencies else 0.0
        maximum_harness_latency = max(harness_latencies) if harness_latencies else 0.0
        print(
            f"conversations={len(results)} completed={completed} "
            f"assistant_turns={assistant_turns} repeated_user_stops={repeated_stops}"
        )
        print(
            f"status_failures={status_failures} shape_failures={shape_failures} "
            f"harness_latency_failures={harness_latency_failures} simulator_retries={retries} "
            f"simulator_restarts={restarts}"
        )
        print(
            f"duration_seconds=min={min(durations):.3f},"
            f"median={statistics.median(durations):.3f},max={max(durations):.3f}"
        )
        print("simulator_status_counts=" + simulator_statuses)
        print("harness_status_counts=" + harness_statuses)
        print(
            f"harness_latency_seconds=min={minimum_harness_latency:.3f},"
            f"median={median_harness_latency:.3f},max={maximum_harness_latency:.3f}"
        )
        return True
    except Exception:
        return False


def _url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("harness must be an HTTP(S) URL")
    return value.rstrip("/")


def _conversations(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("conversations must equal 5") from error
    if parsed != REQUIRED_CONVERSATIONS:
        raise argparse.ArgumentTypeError("conversations must equal 5")
    return parsed


def _max_turns(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-turns must be between 1 and 3") from error
    if not 1 <= parsed <= MAX_ASSISTANT_TURNS:
        raise argparse.ArgumentTypeError("max-turns must be between 1 and 3")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded patient simulator smoke checks.")
    parser.add_argument("--harness-url", required=True, type=_url)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--conversations", default=str(REQUIRED_CONVERSATIONS), type=_conversations)
    parser.add_argument("--max-turns", default=str(MAX_ASSISTANT_TURNS), type=_max_turns)
    return parser.parse_args()


def main() -> int:
    try:
        arguments = parse_args()
        api_key = read_api_key_file(arguments.api_key_file)
        results = run_smoke(
            harness_url=arguments.harness_url,
            api_key=api_key,
            conversations=arguments.conversations,
            max_turns=arguments.max_turns,
        )
    except (Exception, KeyboardInterrupt):
        print_summary([])
        return 1
    if not print_summary(results):
        return 1
    return int(
        any(
            result.status_failures or result.shape_failures or result.harness_latency_failures
            for result in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
