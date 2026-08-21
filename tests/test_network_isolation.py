"""Regression gates for the submission's Lunit-only egress boundary."""

from __future__ import annotations

import ast
import re
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from lunit_hackathon.artifacts import MCP_ENDPOINT, MODEL_ENDPOINT
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import ConfigurationError
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.mcp_client import MCPClient
from lunit_hackathon.network_policy import (
    OFFICIAL_MCP_URL,
    OFFICIAL_MODEL_BASE_URL,
    OFFICIAL_MODEL_CHAT_COMPLETIONS_URL,
    require_official_mcp_endpoint,
    require_official_model_endpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    PROJECT_ROOT / "app.py",
    *(PROJECT_ROOT / "lunit_hackathon").rglob("*.py"),
)
_URL = re.compile(r"https?://[^\s'\"`)]+")
_ALLOWED_RUNTIME_URL_LITERALS = {
    OFFICIAL_MODEL_BASE_URL,
    OFFICIAL_MODEL_CHAT_COMPLETIONS_URL,
    OFFICIAL_MCP_URL,
}
_NETWORK_MODULES = {
    "aiohttp",
    "http.client",
    "httpx",
    "httpx2",
    "requests",
    "socket",
    "urllib.request",
}
_NETWORK_IMPORT_ALLOWLIST = {
    "app.py": {"httpx"},
    "lunit_hackathon/l2_client.py": {"httpx"},
    "lunit_hackathon/mcp_client.py": {"httpx2"},
}


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _network_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            for module in _NETWORK_MODULES:
                if name == module or name.startswith(f"{module}."):
                    found.add(module)
    return found


def _settings(**updates: object) -> Settings:
    return Settings(_env_file=None).model_copy(
        update={
            "lunit_fm_api_key": SecretStr("lunit_synthetic_test_only"),
            **updates,
        }
    )


def test_runtime_python_contains_no_non_lunit_http_url_literals():
    observed: dict[str, set[str]] = {}
    for path in RUNTIME_FILES:
        urls = set(_URL.findall(path.read_text(encoding="utf-8")))
        if urls:
            observed[_relative(path)] = urls

    assert set().union(*observed.values()) <= _ALLOWED_RUNTIME_URL_LITERALS


def test_network_capable_imports_are_confined_to_reviewed_adapters():
    observed = {
        _relative(path): imports
        for path in RUNTIME_FILES
        if (imports := _network_imports(path))
    }

    assert observed == _NETWORK_IMPORT_ALLOWLIST


def test_every_runtime_httpx_session_disables_environment_and_redirects():
    calls: list[tuple[str, ast.Call]] = []
    for path in RUNTIME_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "httpx"
                and node.func.attr == "AsyncClient"
            ):
                calls.append((_relative(path), node))

    assert {path for path, _ in calls} == {
        "app.py",
        "lunit_hackathon/l2_client.py",
    }
    for path, call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        for name in ("trust_env", "follow_redirects"):
            value = keywords.get(name)
            assert isinstance(value, ast.Constant), f"{path}: missing explicit {name}"
            assert value.value is False, f"{path}: {name} must be False"


@pytest.mark.parametrize(
    "url",
    [
        "http://model.hackathon.lunit.io/v1/chat/completions",
        "https://model.hackathon.lunit.io",
        "https://model.hackathon.lunit.io/",
        "https://model.hackathon.lunit.io/v1/chat/completions/",
        "https://model.hackathon.lunit.io/v1/chat/completions?next=evil",
        "https://model.hackathon.lunit.io.evil.test/v1/chat/completions",
        "https://model.hackathon.lunit.io@evil.test/v1/chat/completions",
        "https://example.test/v1/chat/completions",
    ],
)
def test_model_endpoint_allowlist_is_exact(url):
    with pytest.raises(ConfigurationError, match="allowlist"):
        require_official_model_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://mcp.hackathon.lunit.io/mcp",
        "https://mcp.hackathon.lunit.io",
        "https://mcp.hackathon.lunit.io/mcp/",
        "https://mcp.hackathon.lunit.io/mcp?next=evil",
        "https://mcp.hackathon.lunit.io.evil.test/mcp",
        "https://mcp.hackathon.lunit.io@evil.test/mcp",
        "https://example.test/mcp",
    ],
)
def test_mcp_endpoint_allowlist_is_exact(url):
    with pytest.raises(ConfigurationError, match="allowlist"):
        require_official_mcp_endpoint(url)


def test_official_endpoints_are_accepted():
    require_official_model_endpoint(OFFICIAL_MODEL_CHAT_COMPLETIONS_URL)
    require_official_mcp_endpoint(OFFICIAL_MCP_URL)


def test_defaults_compiled_manifest_and_network_policy_are_identical(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_URL", raising=False)
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)
    settings = Settings(_env_file=None)

    assert settings.lunit_fm_api_url == OFFICIAL_MODEL_BASE_URL
    assert settings.chat_completions_url == MODEL_ENDPOINT
    assert settings.mcp_url == MCP_ENDPOINT
    assert MODEL_ENDPOINT == OFFICIAL_MODEL_CHAT_COMPLETIONS_URL
    assert MCP_ENDPOINT == OFFICIAL_MCP_URL


async def test_l2_rejects_external_override_before_http_adapter_runs():
    class ForbiddenHTTPClient:
        async def post(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("an external HTTP request was attempted")

    settings = _settings(lunit_fm_api_url="https://pubmed.example.test")

    with pytest.raises(ConfigurationError, match="allowlist"):
        await L2Client(settings, http_client=ForbiddenHTTPClient()).complete(
            messages=[{"role": "user", "content": "synthetic question"}]
        )


async def test_mcp_rejects_external_override_before_http_adapter_runs():
    factory_called = False

    @asynccontextmanager
    async def forbidden_http_factory(**kwargs):
        nonlocal factory_called
        del kwargs
        factory_called = True
        raise AssertionError("an external HTTP request was attempted")
        yield  # pragma: no cover

    settings = _settings(mcp_url="https://dailymed.example.test/mcp")

    with pytest.raises(ConfigurationError, match="allowlist"):
        async with MCPClient(settings, http_client_factory=forbidden_http_factory).connect():
            pass

    assert factory_called is False


def test_l2_owned_http_session_ignores_ambient_proxy_and_redirect_settings(monkeypatch):
    observed: dict[str, object] = {}
    sentinel = SimpleNamespace()

    def fake_async_client(**kwargs):
        observed.update(kwargs)
        return sentinel

    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.test")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/untrusted-ca.pem")
    monkeypatch.setattr("lunit_hackathon.l2_client.httpx.AsyncClient", fake_async_client)

    client = L2Client(_settings())

    assert client._client() is sentinel
    assert observed == {"follow_redirects": False, "trust_env": False}


async def test_mcp_session_ignores_ambient_proxy_and_does_not_follow_redirects(
    monkeypatch,
):
    observed: dict[str, object] = {}

    @asynccontextmanager
    async def http_client_factory(**kwargs):
        observed.update(kwargs)
        yield object()

    @asynccontextmanager
    async def transport_context():
        yield ("read", "write", "session")

    def transport_factory(url, *, http_client):
        assert url == OFFICIAL_MCP_URL
        assert http_client is not None
        return transport_context()

    @asynccontextmanager
    async def client_factory(transport):
        assert transport is not None
        yield SimpleNamespace()

    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.test")
    monkeypatch.setenv("SSL_CERT_FILE", "/tmp/untrusted-ca.pem")

    client = MCPClient(
        _settings(),
        http_client_factory=http_client_factory,
        transport_factory=transport_factory,
        client_factory=client_factory,
    )
    async with client.connect():
        pass

    assert observed["follow_redirects"] is False
    assert observed["trust_env"] is False
    assert str(observed["headers"]["Authorization"]).startswith("Bearer lunit_")
