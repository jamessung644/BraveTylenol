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
    monkeypatch.delenv("MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("LUNIT_REASONING_EFFORT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds == 65
    assert settings.max_completion_tokens == 1024
    assert settings.reasoning_effort == "low"


def test_container_defaults_to_one_call_direct_mode(monkeypatch):
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.mcp_url is None


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
