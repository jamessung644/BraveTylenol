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

    assert settings.max_completion_tokens == 3072
    assert settings.reasoning_effort == "low"
    assert settings.agent_mode == "fast"
