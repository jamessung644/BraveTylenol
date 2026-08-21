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
    monkeypatch.delenv("L2_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_L2_REQUESTS", raising=False)
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("HARNESS_MODE", raising=False)
    monkeypatch.delenv("ENABLE_RAG", raising=False)

    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds == 65
    assert settings.max_completion_tokens == 1024
    assert settings.reasoning_effort == "low"
    assert settings.retry_attempts == 1
    assert settings.max_concurrent_l2_requests == 16
    assert settings.agent_mode == "direct"
    assert settings.enable_rag is False
    assert settings.rag_enabled is False


def test_container_defaults_to_one_call_direct_mode(monkeypatch):
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)
    monkeypatch.delenv("ENABLE_RAG", raising=False)

    settings = Settings(_env_file=None)

    assert settings.mcp_url is None
    assert settings.rag_enabled is False


def test_legacy_rag_environment_does_not_enable_rag_without_opt_in(monkeypatch):
    monkeypatch.delenv("AGENT_MODE", raising=False)
    monkeypatch.delenv("ENABLE_RAG", raising=False)
    monkeypatch.setenv("HARNESS_MODE", "rag")
    monkeypatch.setenv("LUNIT_MCP_URL", "https://mcp.injected-by-pipeline.test")

    settings = Settings(_env_file=None)

    assert settings.agent_mode == "rag"
    assert settings.mcp_url == "https://mcp.injected-by-pipeline.test"
    assert settings.enable_rag is False
    assert settings.rag_enabled is False


def test_rag_requires_mode_endpoint_and_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "true")
    monkeypatch.setenv("AGENT_MODE", "direct")
    monkeypatch.setenv("LUNIT_MCP_URL", "https://mcp.example.test")

    assert Settings(_env_file=None).rag_enabled is False

    monkeypatch.setenv("AGENT_MODE", "rag")
    monkeypatch.delenv("LUNIT_MCP_URL", raising=False)

    assert Settings(_env_file=None).rag_enabled is False

    monkeypatch.setenv("LUNIT_MCP_URL", "https://mcp.example.test")

    settings = Settings(_env_file=None)

    assert settings.enable_rag is True
    assert settings.rag_enabled is True


def test_concurrent_l2_limit_loads_from_environment(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_L2_REQUESTS", "12")

    assert Settings(_env_file=None).max_concurrent_l2_requests == 12


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
