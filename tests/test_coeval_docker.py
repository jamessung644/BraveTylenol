"""Opt-in Docker contract test based on lunit-io/CoEval at 741263cf.

This test builds the submission image, starts it without an API key in the
container environment, and sends the same OpenAI-compatible request shape used
by CoEval. It is intentionally excluded from normal unit-test runs because it
requires Docker, network access, and a real Lunit API key.
"""

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from lunit_hackathon.config import Settings

_RUN_DOCKER_TESTS = os.getenv("RUN_COEVAL_DOCKER") == "1"
_COEVAL_CONCURRENT_LIMITS = (12, 16)
_COEVAL_MAX_TOKENS = 6_144
_COEVAL_REQUEST_TIMEOUT_SECONDS = 180.0
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not _RUN_DOCKER_TESTS,
    reason="set RUN_COEVAL_DOCKER=1 to run the live Docker/CoEval contract test",
)


@dataclass(frozen=True)
class DockerTarget:
    base_url: str
    api_key: str = field(repr=False)


@pytest.fixture(scope="module")
def coeval_docker_target() -> Iterator[DockerTarget]:
    if shutil.which("docker") is None:
        pytest.fail("Docker CLI is required when RUN_COEVAL_DOCKER=1")

    api_key = os.getenv("COEVAL_TEST_API_KEY") or Settings().api_key
    if not api_key:
        pytest.fail("COEVAL_TEST_API_KEY or LUNIT_FM_API_KEY is required")

    suffix = uuid.uuid4().hex[:10]
    image = "brave-tylenol:coeval-contract-test"
    container = f"brave-tylenol-coeval-contract-{suffix}"
    _docker("build", "--tag", image, ".", timeout=900)
    try:
        _docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            container,
            "--publish",
            "127.0.0.1::8000",
            image,
            timeout=60,
        )
        published = _docker("port", container, "8000/tcp", timeout=30).strip()
        port = published.rsplit(":", 1)[-1]
        base_url = f"http://127.0.0.1:{port}"
        _wait_until_healthy(base_url, container)
        yield DockerTarget(base_url=base_url, api_key=api_key)
    finally:
        _remove_container(container)


def test_coeval_docker_preflight(coeval_docker_target: DockerTarget) -> None:
    target = coeval_docker_target
    with httpx.Client(timeout=10.0) as client:
        health = client.get(f"{target.base_url}/health")
        healthz = client.get(f"{target.base_url}/healthz")
        models = client.get(f"{target.base_url}/v1/models")

    assert health.status_code == healthz.status_code == models.status_code == 200
    assert health.json() == healthz.json() == {"status": "ok"}
    assert models.json()["data"][0]["id"] == "team-chatbot"


@pytest.mark.asyncio
async def test_coeval_docker_accepts_multiturn_generation_contract(
    coeval_docker_target: DockerTarget,
) -> None:
    response = await _coeval_request(coeval_docker_target, sample_id=0)

    assert response.status_code == 200
    _assert_valid_completion(response.json())


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_limit", _COEVAL_CONCURRENT_LIMITS)
async def test_coeval_docker_handles_official_concurrency(
    coeval_docker_target: DockerTarget,
    concurrent_limit: int,
) -> None:
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=_COEVAL_REQUEST_TIMEOUT_SECONDS) as client:
        responses = await asyncio.gather(
            *(
                _coeval_request(
                    coeval_docker_target,
                    sample_id=index,
                    client=client,
                )
                for index in range(concurrent_limit)
            )
        )
    elapsed = time.perf_counter() - started

    assert [response.status_code for response in responses] == [
        200
    ] * concurrent_limit
    payloads = [response.json() for response in responses]
    for payload in payloads:
        _assert_valid_completion(payload)
    assert len({payload["id"] for payload in payloads}) == concurrent_limit
    assert all(response.headers.get("x-request-id") for response in responses)
    assert elapsed < _COEVAL_REQUEST_TIMEOUT_SECONDS


async def _coeval_request(
    target: DockerTarget,
    *,
    sample_id: int,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    payload = {
        "model": "team-chatbot",
        "messages": [
            {"role": "system", "content": "You are a careful medical assistant."},
            {
                "role": "user",
                "content": f"I have had a mild fever for two days. Case {sample_id}.",
            },
            {
                "role": "assistant",
                "content": "Do you have warning signs or relevant conditions?",
            },
            {
                "role": "user",
                "content": "No breathing difficulty or confusion. What should I monitor?",
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": _COEVAL_MAX_TOKENS,
        "stream": False,
    }
    if client is not None:
        return await client.post(
            f"{target.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {target.api_key}"},
            json=payload,
        )
    async with httpx.AsyncClient(timeout=_COEVAL_REQUEST_TIMEOUT_SECONDS) as owned_client:
        return await owned_client.post(
            f"{target.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {target.api_key}"},
            json=payload,
        )


def _assert_valid_completion(payload: dict[str, Any]) -> None:
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "team-chatbot"
    assert isinstance(payload["id"], str) and payload["id"].startswith("chatcmpl-")
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["message"]["content"].strip()
    assert choice["finish_reason"] in {"stop", "length"}
    usage = payload["usage"]
    usage_names = ("prompt_tokens", "completion_tokens", "total_tokens")
    assert all(isinstance(usage[name], int) for name in usage_names)
    assert all(usage[name] >= 0 for name in usage_names)
    # Public usage includes every bounded malformed/blank recovery attempt, so
    # it can exceed the per-attempt 1,024-token server cap.
    assert usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )


def _wait_until_healthy(base_url: str, container: str) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    logs = _docker("logs", container, timeout=30, check=False)
    pytest.fail(f"Docker service did not become healthy; logs:\n{logs[-4_000:]}")


def _docker(
    *arguments: str,
    timeout: float,
    check: bool = True,
) -> str:
    result = subprocess.run(
        ["docker", *arguments],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        pytest.fail(
            f"docker {' '.join(arguments[:2])} failed with exit {result.returncode}: "
            f"{result.stderr[-2_000:]}"
        )
    return result.stdout


def _remove_container(container: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "--force", container],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # The container name is unique to this test run; a later Docker cleanup
        # can remove it if the local daemon itself stopped responding.
        pass
