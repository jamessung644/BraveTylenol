import pytest
from pydantic import ValidationError

import lunit_hackathon.submission_credential as submission_credential
from lunit_hackathon.config import Settings


def test_settings_loads_dotenv_and_hides_secrets(tmp_path, monkeypatch):
    for name in (
        "LUNIT_FM_API_URL",
        "LUNIT_FM_API_KEY",
        "LUNIT_FM_MODEL",
        "DASHBOARD_USERNAME",
        "DASHBOARD_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "LUNIT_FM_API_URL=https://model.example.test\n"
        "LUNIT_FM_API_KEY=test-api-key\n"
        "LUNIT_FM_MODEL=test-model\n"
        "DASHBOARD_USERNAME=test-user\n"
        "DASHBOARD_PASSWORD=test-password\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=dotenv)

    assert settings.chat_completions_url == "https://model.example.test/v1/chat/completions"
    assert settings.api_key == "test-api-key"
    assert settings.lunit_fm_model == "test-model"
    assert "test-api-key" not in repr(settings)
    assert "test-password" not in repr(settings)


def test_example_placeholders_are_not_treated_as_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("LUNIT_FM_API_KEY=여기에_직접_입력\n", encoding="utf-8")

    assert Settings(_env_file=dotenv).api_key is None


def test_embedded_main_credential_is_explicitly_gated(monkeypatch):
    calls = []

    def fake_import(name):
        calls.append(name)
        return type("FakeMain", (), {"EMBEDDED_LUNIT_API_KEY": "lunit_test_embedded"})

    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    monkeypatch.delenv("SUBMISSION_CREDENTIAL_SOURCE", raising=False)
    monkeypatch.setattr(submission_credential, "import_module", fake_import)

    ordinary = Settings(_env_file=None)
    packaged = Settings(
        _env_file=None,
        SUBMISSION_CREDENTIAL_SOURCE="main",
    )

    assert ordinary.embedded_api_key is None
    assert calls == []
    assert packaged.embedded_api_key == "lunit_test_embedded"
    assert calls == ["main"]
    assert "lunit_test_embedded" not in repr(packaged)


def test_non_string_embedded_main_credential_is_unavailable(monkeypatch):
    monkeypatch.delenv("LUNIT_FM_API_KEY", raising=False)
    monkeypatch.setattr(
        submission_credential,
        "import_module",
        lambda name: type("FakeMain", (), {"EMBEDDED_LUNIT_API_KEY": object()}),
    )
    settings = Settings(
        _env_file=None,
        SUBMISSION_CREDENTIAL_SOURCE="main",
    )

    assert settings.embedded_api_key is None


def test_latency_controls_have_safe_defaults(monkeypatch):
    monkeypatch.delenv("MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("LUNIT_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("L2_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("HARNESS_MODE", raising=False)

    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds == 165
    assert settings.model_attempt_timeout_seconds == 45
    assert settings.max_completion_tokens == 4096
    assert settings.reasoning_effort == "low"
    assert settings.retry_attempts == 0
    assert settings.agent_mode == "hybrid"


def test_release_configuration_rejects_transport_retries(monkeypatch):
    monkeypatch.setenv("L2_RETRY_ATTEMPTS", "1")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_container_defaults_to_bounded_hybrid_mode(monkeypatch):
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("HARNESS_MODE", raising=False)
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)
    monkeypatch.delenv("MAX_MCP_CALLS", raising=False)
    monkeypatch.delenv("MAX_TOOL_CALLS", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_MODEL_CALLS", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_MCP_CALLS", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_RAG_REQUESTS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.agent_mode == "hybrid"
    assert settings.mcp_url == "https://mcp.hackathon.lunit.io/mcp"
    assert settings.max_mcp_calls == 3
    assert settings.max_concurrent_model_calls == 16
    assert settings.max_concurrent_mcp_calls == 16
    assert settings.max_concurrent_rag_requests == 16


def test_legacy_baseline_environment_names_remain_supported(monkeypatch):
    for name in (
        "REQUEST_TIMEOUT_SECONDS",
        "AGENT_MODE",
        "MAX_MCP_CALLS",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UPSTREAM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("HARNESS_MODE", "passthrough")
    monkeypatch.setenv("MAX_TOOL_CALLS", "1")
    monkeypatch.setenv("HARNESS_LOG_LEVEL", "WARNING")

    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds == 90
    assert settings.agent_mode == "passthrough"
    assert settings.max_mcp_calls == 1
    assert settings.log_level == "WARNING"
