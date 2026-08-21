from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _lines(name: str) -> list[str]:
    return [line.strip() for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def test_dockerfile_is_a_non_root_fastapi_runtime_with_stdlib_healthcheck():
    """Removing the production app entrypoint or non-root boundary must fail this test."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim" in dockerfile
    assert "COPY requirements.txt /app/requirements.txt" in dockerfile
    assert "RUN pip install --no-cache-dir -r /app/requirements.txt" in dockerfile
    assert "COPY app.py /app/app.py" in dockerfile
    assert "COPY lunit_hackathon /app/lunit_hackathon" in dockerfile
    assert "USER 65532:65532" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" in dockerfile
    assert '"uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"' in dockerfile


def test_dockerfile_copies_only_the_declared_production_runtime_files():
    """A broad COPY or legacy server copy would package tests, credentials, or the old runtime."""
    copy_sources = [
        line.split()[1]
        for line in _lines("Dockerfile")
        if line.startswith("COPY ") and not line.startswith("COPY --")
    ]

    assert copy_sources == ["requirements.txt", "app.py", "lunit_hackathon"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "main.py" not in dockerfile
    assert ".env" not in dockerfile
    assert ".git" not in dockerfile
    assert "tests" not in dockerfile
    assert "openai" not in dockerfile.lower()


def test_dockerignore_is_a_strict_runtime_allowlist():
    """Adding an unreviewed build-context path must remain impossible by default."""
    patterns = _lines(".dockerignore")

    assert patterns[0] == "**"
    assert patterns[:6] == [
        "**",
        "!Dockerfile",
        "!requirements.txt",
        "!app.py",
        "!lunit_hackathon/",
        "!lunit_hackathon/**",
    ]
    assert {
        "lunit_hackathon/**/__pycache__/",
        "lunit_hackathon/**/*.py[cod]",
        "lunit_hackathon/**/.env*",
        "lunit_hackathon/**/.git/",
        "lunit_hackathon/**/.pytest_cache/",
        "lunit_hackathon/**/.ruff_cache/",
        "lunit_hackathon/**/.coverage",
        "lunit_hackathon/**/.idea/",
        "lunit_hackathon/**/.vscode/",
        "lunit_hackathon/**/tests/",
        "lunit_hackathon/**/docs/",
        "lunit_hackathon/**/scripts/",
        "lunit_hackathon/**/*.key",
        "lunit_hackathon/**/*.pem",
        "lunit_hackathon/**/*.p12",
        "lunit_hackathon/**/[Oo]pen[Aa][Ii]*",
    } <= set(patterns)


def test_runtime_requirements_exclude_development_and_openai_dependencies():
    """A test/linter or judge SDK in the production layer must be caught."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()

    assert "fastapi" in requirements
    assert "uvicorn" in requirements
    for forbidden in ("pytest", "ruff", "openai", "langchain"):
        assert forbidden not in requirements
