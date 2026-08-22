from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from lunit_hackathon.submission_credential import embedded_main_api_key

_PLACEHOLDERS = {"", "여기에_직접_입력", "lunit_replace_me"}


class Settings(BaseSettings):
    """Validated runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    lunit_fm_api_url: str = Field(
        default="https://model.hackathon.lunit.io",
        validation_alias="LUNIT_FM_API_URL",
    )
    lunit_fm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LUNIT_FM_API_KEY",
    )
    submission_credential_source: Literal["none", "main"] = Field(
        default="none",
        validation_alias="SUBMISSION_CREDENTIAL_SOURCE",
    )
    lunit_fm_model: str = Field(
        default="Lunit/L2-preview",
        validation_alias="LUNIT_FM_MODEL",
    )
    dashboard_username: SecretStr | None = Field(
        default=None,
        validation_alias="DASHBOARD_USERNAME",
    )
    dashboard_password: SecretStr | None = Field(
        default=None,
        validation_alias="DASHBOARD_PASSWORD",
    )

    request_timeout_seconds: float = Field(
        default=165.0,
        gt=0,
        le=175,
        validation_alias=AliasChoices(
            "REQUEST_TIMEOUT_SECONDS",
            "UPSTREAM_TIMEOUT_SECONDS",
        ),
    )
    model_attempt_timeout_seconds: float = Field(
        default=145.0,
        gt=0,
        le=145,
        validation_alias="MODEL_ATTEMPT_TIMEOUT_SECONDS",
    )
    retry_attempts: int = Field(
        default=0,
        ge=0,
        le=0,
        validation_alias="L2_RETRY_ATTEMPTS",
    )
    max_completion_tokens: int = Field(
        default=2_048,
        ge=512,
        le=6_144,
        validation_alias="MAX_COMPLETION_TOKENS",
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        validation_alias="LUNIT_REASONING_EFFORT",
    )
    agent_mode: Literal["direct", "hybrid", "rag", "passthrough"] = Field(
        default="hybrid",
        validation_alias=AliasChoices("AGENT_MODE", "HARNESS_MODE"),
    )
    mcp_url: str | None = Field(
        default="https://mcp.hackathon.lunit.io/mcp",
        validation_alias="LUNIT_MCP_URL",
    )
    max_mcp_calls: int = Field(
        default=3,
        ge=0,
        le=12,
        validation_alias=AliasChoices("MAX_MCP_CALLS", "MAX_TOOL_CALLS"),
    )
    max_tool_result_chars: int = Field(
        default=8_000,
        ge=1_000,
        le=100_000,
        validation_alias="MAX_TOOL_RESULT_CHARS",
    )
    max_evidence_chars: int = Field(
        default=12_000,
        ge=2_000,
        le=200_000,
        validation_alias="MAX_EVIDENCE_CHARS",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "HARNESS_LOG_LEVEL"),
    )
    max_concurrent_model_calls: int = Field(
        default=16,
        ge=1,
        le=64,
        validation_alias="MAX_CONCURRENT_MODEL_CALLS",
    )
    max_concurrent_mcp_calls: int = Field(
        default=16,
        ge=1,
        le=64,
        validation_alias="MAX_CONCURRENT_MCP_CALLS",
    )
    max_concurrent_rag_requests: int = Field(
        default=16,
        ge=1,
        le=16,
        validation_alias="MAX_CONCURRENT_RAG_REQUESTS",
    )

    @field_validator(
        "lunit_fm_api_key",
        "dashboard_username",
        "dashboard_password",
        mode="before",
    )
    @classmethod
    def reject_placeholders(cls, value: Any) -> Any:
        if value is None or str(value).strip() in _PLACEHOLDERS:
            return None
        return value

    @property
    def api_key(self) -> str | None:
        return self.lunit_fm_api_key.get_secret_value() if self.lunit_fm_api_key else None

    @property
    def embedded_api_key(self) -> str | None:
        return embedded_main_api_key(self.submission_credential_source == "main")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.lunit_fm_api_url.rstrip('/')}/v1/chat/completions"
