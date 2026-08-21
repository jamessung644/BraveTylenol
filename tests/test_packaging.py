import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRACKED_RELEASE_PATHS = {
    ".dockerignore",
    "Dockerfile",
    "main.py",
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


def test_docker_runs_the_restored_main_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY main.py /app/main.py" in dockerfile
    assert 'CMD ["python", "main.py", "serve"' in dockerfile
    assert "COPY app.py" not in dockerfile
    assert "COPY lunit_hackathon" not in dockerfile
    assert "COPY . " not in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "http://127.0.0.1:8000/health" in dockerfile


def test_docker_context_is_the_minimal_main_runtime():
    ignore_rules = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ignore_rules == {"**", "!Dockerfile", "!main.py"}
    assert "!.env" not in ignore_rules


def test_release_runtime_and_build_inputs_are_tracked_explicitly():
    tracked = _git_paths("ls-files")

    assert REQUIRED_TRACKED_RELEASE_PATHS <= tracked, (
        "release-critical files are absent from git ls-files: "
        f"{sorted(REQUIRED_TRACKED_RELEASE_PATHS - tracked)}"
    )


def test_packaged_runtime_has_no_untracked_dependency():
    untracked = _git_paths("ls-files", "--others", "--exclude-standard")
    untracked_runtime = untracked & REQUIRED_TRACKED_RELEASE_PATHS

    assert not untracked_runtime, (
        "Docker runtime still depends on untracked paths: "
        f"{sorted(untracked_runtime)}"
    )


@pytest.mark.skipif(
    not os.environ.get("CANDIDATE_COMMIT"),
    reason="set CANDIDATE_COMMIT to run the clean git-archive release gate",
)
def test_candidate_commit_archive_contains_minimal_runtime(tmp_path):
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
