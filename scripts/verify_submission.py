#!/usr/bin/env python3
"""Build and exercise the submission exactly through its evaluator-facing API."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPException as HttpClientError
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    """Raised when the submission does not satisfy the smoke-test contract."""


class HttpStatusError(VerificationError):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"HTTP request failed with status {status}")


@dataclass(frozen=True)
class Options:
    image: str
    port: int
    timeout: float
    startup_timeout: float
    harness_mode: str
    skip_build: bool


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    bearer_token: str | None = None,
    timeout: float = 10,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode()
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            raw_body = response.read()
    except HTTPError as error:
        raise HttpStatusError(error.code) from error
    except (URLError, TimeoutError, HttpClientError) as error:
        raise VerificationError(
            f"HTTP request could not complete: {type(error).__name__}"
        ) from error
    try:
        decoded = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError("HTTP response was not valid JSON") from error
    if not isinstance(decoded, dict):
        raise VerificationError("HTTP response must be a JSON object")
    return status, decoded


def docker_run_command(
    *,
    image: str,
    container_name: str,
    port: int,
    harness_mode: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"127.0.0.1:{port}:8000",
        "--env",
        f"HARNESS_MODE={harness_mode}",
        image,
    ]


def validate_models(payload: Mapping[str, Any]) -> None:
    models = payload.get("data")
    if payload.get("object") != "list" or not isinstance(models, list):
        raise VerificationError(
            "/v1/models did not return an OpenAI-compatible model list"
        )
    if not any(
        isinstance(model, dict) and model.get("id") == "team-chatbot"
        for model in models
    ):
        raise VerificationError("/v1/models did not advertise team-chatbot")


def validate_chat_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if (
        payload.get("object") != "chat.completion"
        or not isinstance(choices, list)
        or not choices
    ):
        raise VerificationError("chat response was not an OpenAI-compatible completion")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise VerificationError("chat response choice was invalid")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise VerificationError("chat response did not contain an assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VerificationError("chat response content was empty")
    usage = payload.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return {
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "content_chars": len(content),
        "total_tokens": total_tokens,
    }


def run_command(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except FileNotFoundError as error:
        raise VerificationError(
            f"Required command is unavailable: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise VerificationError(
            f"Command failed with exit code {error.returncode}: {command[0]}"
        ) from error


def wait_until_ready(base_url: str, *, startup_timeout: float) -> None:
    deadline = time.monotonic() + startup_timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, payload = request_json(f"{base_url}/v1/models", timeout=2)
            if status == 200:
                validate_models(payload)
                return
        except VerificationError as error:
            last_error = error
        time.sleep(0.5)
    detail = f" ({last_error})" if last_error else ""
    raise VerificationError(
        f"container did not become ready within {startup_timeout:g}s{detail}"
    )


def chat_payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"model": "team-chatbot", "messages": messages, "stream": False}


def print_summary(label: str, elapsed: float, summary: Mapping[str, Any]) -> None:
    print(
        f"{label}: PASS in {elapsed:.2f}s "
        f"(model={summary['model']}, finish={summary['finish_reason']}, "
        f"chars={summary['content_chars']}, tokens={summary['total_tokens']})"
    )


def verify(options: Options, *, api_key: str) -> None:
    if not api_key:
        raise VerificationError("LUNIT_FM_API_KEY is not set")
    container_name = f"brave-tylenol-verify-{os.getpid()}"
    base_url = f"http://127.0.0.1:{options.port}"

    if not options.skip_build:
        print(f"Building {options.image} ...")
        run_command(["docker", "build", "--tag", options.image, "."])

    print(f"Starting isolated evaluator-style container on port {options.port} ...")
    run_command(
        docker_run_command(
            image=options.image,
            container_name=container_name,
            port=options.port,
            harness_mode=options.harness_mode,
        ),
        capture_output=True,
    )
    try:
        wait_until_ready(base_url, startup_timeout=options.startup_timeout)
        print("models: PASS")

        try:
            request_json(
                f"{base_url}/v1/chat/completions",
                method="POST",
                payload=chat_payload([{"role": "user", "content": "안녕하세요"}]),
                timeout=5,
            )
        except HttpStatusError as error:
            if error.status != 503:
                raise VerificationError(
                    f"unauthenticated chat returned unexpected status {error.status}"
                ) from error
        else:
            raise VerificationError(
                "container unexpectedly accepted chat without evaluator Bearer auth"
            )
        print("bearer forwarding guard: PASS")

        first_messages = [
            {
                "role": "user",
                "content": "고혈압 환자가 집에서 혈압을 잴 때 지켜야 할 기본 원칙을 간단히 설명해 주세요.",
            }
        ]
        started = time.monotonic()
        status, first_payload = request_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            payload=chat_payload(first_messages),
            bearer_token=api_key,
            timeout=options.timeout,
        )
        if status != 200:
            raise VerificationError(f"single-turn chat returned HTTP {status}")
        first_summary = validate_chat_completion(first_payload)
        print_summary("single-turn L2", time.monotonic() - started, first_summary)

        first_answer = first_payload["choices"][0]["message"]["content"]
        second_messages = [
            *first_messages,
            {"role": "assistant", "content": first_answer},
            {
                "role": "user",
                "content": "방금 답변의 핵심만 두 문장으로 다시 말해 주세요.",
            },
        ]
        started = time.monotonic()
        status, second_payload = request_json(
            f"{base_url}/v1/chat/completions",
            method="POST",
            payload=chat_payload(second_messages),
            bearer_token=api_key,
            timeout=options.timeout,
        )
        if status != 200:
            raise VerificationError(f"multi-turn chat returned HTTP {status}")
        second_summary = validate_chat_completion(second_payload)
        print_summary("multi-turn L2", time.monotonic() - started, second_summary)
    except VerificationError:
        try:
            logs = run_command(
                ["docker", "logs", "--tail", "40", container_name], capture_output=True
            )
            if logs.stdout.strip():
                print("Container log tail:", file=sys.stderr)
                print(logs.stdout.rstrip(), file=sys.stderr)
            if logs.stderr.strip():
                print(logs.stderr.rstrip(), file=sys.stderr)
        except VerificationError:
            pass
        raise
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "2", container_name],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> Options:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="brave-tylenol:verify")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=170)
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument(
        "--harness-mode", choices=("passthrough", "rag"), default="passthrough"
    )
    parser.add_argument("--skip-build", action="store_true")
    arguments = parser.parse_args(argv)
    return Options(
        image=arguments.image,
        port=arguments.port,
        timeout=arguments.timeout,
        startup_timeout=arguments.startup_timeout,
        harness_mode=arguments.harness_mode,
        skip_build=arguments.skip_build,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        verify(parse_args(argv), api_key=os.environ.get("LUNIT_FM_API_KEY", ""))
    except VerificationError as error:
        print(f"VERIFICATION FAILED: {error}", file=sys.stderr)
        return 1
    print("Submission verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
