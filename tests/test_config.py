import pytest
from pydantic import ValidationError

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


def test_latency_controls_have_safe_defaults(monkeypatch):
    for name in (
        "REQUEST_TIMEOUT_SECONDS",
        "UPSTREAM_TIMEOUT_SECONDS",
        "RETRIEVAL_TIMEOUT_SECONDS",
        "GENERATION_TIMEOUT_SECONDS",
        "VERIFICATION_MINIMUM_SECONDS",
        "MAX_COMPLETION_TOKENS",
        "LUNIT_REASONING_EFFORT",
        "RETRIEVAL_REASONING_EFFORT",
        "GENERATION_REASONING_EFFORT",
        "VERIFICATION_REASONING_EFFORT",
        "L2_RETRY_ATTEMPTS",
        "AGENT_MODE",
        "HARNESS_MODE",
        "LUNIT_MCP_URL",
        "MAX_MCP_CALLS",
        "MAX_TOOL_CALLS",
        "MAX_EVIDENCE_CHARS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds == 165
    assert settings.retrieval_timeout_seconds == 75
    assert settings.generation_timeout_seconds == 60
    assert settings.verification_minimum_seconds == 25
    assert settings.max_completion_tokens == 4096
    assert settings.reasoning_effort == "low"
    assert settings.retrieval_reasoning_effort == "medium"
    assert settings.generation_reasoning_effort == "high"
    assert settings.verification_reasoning_effort == "medium"
    assert settings.retry_attempts == 1
    assert settings.agent_mode == "rag"
    assert settings.mcp_url == "https://mcp.hackathon.lunit.io/mcp"
    assert settings.max_mcp_calls == 6
    assert settings.max_evidence_chars == 24_000


def test_container_defaults_to_score_first_mcp_mode(monkeypatch):
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.mcp_url == "https://mcp.hackathon.lunit.io/mcp"


def test_score_first_settings_support_environment_overrides(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TIMEOUT_SECONDS", "74")
    monkeypatch.setenv("GENERATION_TIMEOUT_SECONDS", "59")
    monkeypatch.setenv("VERIFICATION_MINIMUM_SECONDS", "24")
    monkeypatch.setenv("RETRIEVAL_REASONING_EFFORT", "low")
    monkeypatch.setenv("GENERATION_REASONING_EFFORT", "medium")
    monkeypatch.setenv("VERIFICATION_REASONING_EFFORT", "high")

    settings = Settings(_env_file=None)

    assert settings.retrieval_timeout_seconds == 74
    assert settings.generation_timeout_seconds == 59
    assert settings.verification_minimum_seconds == 24
    assert settings.retrieval_reasoning_effort == "low"
    assert settings.generation_reasoning_effort == "medium"
    assert settings.verification_reasoning_effort == "high"


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("RETRIEVAL_TIMEOUT_SECONDS", "0"),
        ("GENERATION_TIMEOUT_SECONDS", "176"),
        ("VERIFICATION_MINIMUM_SECONDS", "0"),
    ],
)
def test_stage_budget_values_must_be_positive_and_bounded(monkeypatch, environment_name, value):
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


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
