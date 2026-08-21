import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGED_RUNTIME_SOURCES = [
    ROOT / "app.py",
    *sorted((ROOT / "lunit_hackathon").rglob("*.py")),
]
REQUIRED_TRACKED_RELEASE_PATHS = {
    ".dockerignore",
    "Dockerfile",
    "app.py",
    "main.py",
    "requirements.txt",
    "lunit_hackathon/__init__.py",
    "lunit_hackathon/artifacts.py",
    "lunit_hackathon/config.py",
    "lunit_hackathon/errors.py",
    "lunit_hackathon/generation.py",
    "lunit_hackathon/l2_client.py",
    "lunit_hackathon/logging_config.py",
    "lunit_hackathon/mcp_client.py",
    "lunit_hackathon/network_policy.py",
    "lunit_hackathon/orchestrator.py",
    "lunit_hackathon/prompts.py",
    "lunit_hackathon/retrieval.py",
    "lunit_hackathon/schemas.py",
    "lunit_hackathon/submission_credential.py",
    "lunit_hackathon/tool_bindings.py",
    "lunit_hackathon/runtime_artifacts/runtime_bundle_v1.json",
    "lunit_hackathon/runtime_artifacts/runtime_bundle_v1.sha256",
    "canonical_sources/docs/model-instructions/10_MCP_CATALOG.md",
    "canonical_sources/docs/model-instructions/14_LUNIT_RUNTIME_APPLICATION_BLUEPRINT.md",
    "canonical_sources/docs/model-instructions/20_GENERATION_PROMPT.md",
    "canonical_sources/docs/model-instructions/30_RETRIEVAL_PROMPT.md",
    "canonical_sources/docs/model-instructions/45_RUNTIME_RESILIENCE.md",
    "runtime_sources/legal_policy_ko_v1.json",
    "runtime_sources/natural_language_policy_ko_v1.json",
    "runtime_sources/release_config_v1.json",
    "scripts/compile_runtime_artifacts.py",
}


def _git_paths(*arguments: str) -> set[str]:
    result = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    }


def test_docker_runs_the_orchestrator_not_the_legacy_direct_proxy():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY app.py" in dockerfile
    assert "COPY lunit_hackathon" in dockerfile
    assert '"uvicorn", "app:app"' in dockerfile
    assert "COPY main.py /app/main.py" in dockerfile
    assert "SUBMISSION_CREDENTIAL_SOURCE=main" in dockerfile
    assert "COPY . " not in dockerfile
    assert "OTEL_SDK_DISABLED=true" in dockerfile
    assert "ProxyHandler({})" in dockerfile
    assert ".open('http://127.0.0.1:8000/health'" in dockerfile


def test_docker_context_contains_runtime_and_excludes_secrets():
    ignore_rules = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "**" in ignore_rules
    assert "!requirements.txt" in ignore_rules
    assert "!app.py" in ignore_rules
    assert "!main.py" in ignore_rules
    assert "!lunit_hackathon/" in ignore_rules
    assert "!lunit_hackathon/**" in ignore_rules
    assert "lunit_hackathon/**/__pycache__/" in ignore_rules
    assert "lunit_hackathon/**/*.py[co]" in ignore_rules
    assert "!.env" not in ignore_rules


def test_packaged_runtime_has_no_legacy_static_medical_fallback():
    packaged_source = "\n".join(
        path.read_text(encoding="utf-8") for path in PACKAGED_RUNTIME_SOURCES
    )

    assert "KOREAN_BASELINE_RESPONSE" not in packaged_source
    assert "_fallback_completion(" not in packaged_source
    assert "completion_payload(" not in packaged_source
    assert "static-korean-baseline" not in packaged_source


def test_packaged_runtime_has_no_unapproved_final_answer_tool():
    forbidden = "submit" + "_final_answer"

    assert all(
        forbidden not in path.read_text(encoding="utf-8")
        for path in PACKAGED_RUNTIME_SOURCES
    )


def test_release_runtime_and_build_inputs_are_tracked_explicitly():
    tracked = _git_paths("ls-files")

    assert REQUIRED_TRACKED_RELEASE_PATHS <= tracked, (
        "release-critical files are absent from git ls-files: "
        f"{sorted(REQUIRED_TRACKED_RELEASE_PATHS - tracked)}"
    )


def test_packaged_runtime_has_no_untracked_dependency():
    untracked = _git_paths("ls-files", "--others", "--exclude-standard")
    untracked_runtime = {
        path
        for path in untracked
        if path
        in {
            "app.py",
            "main.py",
            "Dockerfile",
            ".dockerignore",
            "requirements.txt",
        }
        or path.startswith("lunit_hackathon/")
    }

    assert not untracked_runtime, (
        "Docker runtime still depends on untracked paths: "
        f"{sorted(untracked_runtime)}"
    )


@pytest.mark.skipif(
    not os.environ.get("CANDIDATE_COMMIT"),
    reason="set CANDIDATE_COMMIT to run the clean git-archive release gate",
)
def test_candidate_commit_archive_import_and_chat_smoke(tmp_path):
    candidate = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            f"{os.environ['CANDIDATE_COMMIT']}^{{commit}}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", candidate],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(tmp_path, filter="data")

    archived_paths = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert REQUIRED_TRACKED_RELEASE_PATHS <= archived_paths

    smoke = """
import asyncio

from httpx import ASGITransport, AsyncClient

import app as application_module
from lunit_hackathon.config import Settings
from lunit_hackathon.schemas import L2Completion, TokenUsage


class FakeL2:
    def __init__(self, settings, *, http_client=None, semaphore=None):
        del settings, http_client, semaphore
        self.last_usage = TokenUsage()

    async def complete(self, **kwargs):
        del kwargs
        return L2Completion(content="archive smoke response")


async def main():
    application_module.L2Client = FakeL2
    settings = Settings(_env_file=None).model_copy(update={"agent_mode": "passthrough"})
    application = application_module.create_app(settings)
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://archive",
    ) as client:
        ready = await client.get("/readyz")
        chat = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "smoke"}]},
        )
    assert ready.status_code == 200
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "archive smoke response"


asyncio.run(main())
"""
    environment = {
        "AGENT_MODE": "passthrough",
        "LUNIT_FM_API_KEY": "lunit_test_archive",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", smoke],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, "candidate archive import/chat smoke failed"
