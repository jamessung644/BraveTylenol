from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    retrieval_timeout_seconds: float = Field(
        default=75.0,
        gt=0,
        le=175,
        validation_alias="RETRIEVAL_TIMEOUT_SECONDS",
    )
    generation_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=175,
        validation_alias="GENERATION_TIMEOUT_SECONDS",
    )
    verification_minimum_seconds: float = Field(
        default=25.0,
        gt=0,
        le=175,
        validation_alias="VERIFICATION_MINIMUM_SECONDS",
    )
    retry_attempts: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias="L2_RETRY_ATTEMPTS",
    )
    max_completion_tokens: int = Field(
        default=4_096,
        ge=512,
        le=6_144,
        validation_alias="MAX_COMPLETION_TOKENS",
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="low",
        validation_alias="LUNIT_REASONING_EFFORT",
    )
    retrieval_reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        validation_alias="RETRIEVAL_REASONING_EFFORT",
    )
    generation_reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="high",
        validation_alias="GENERATION_REASONING_EFFORT",
    )
    verification_reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        validation_alias="VERIFICATION_REASONING_EFFORT",
    )
    agent_mode: Literal["direct", "rag", "passthrough"] = Field(
        default="rag",
        validation_alias=AliasChoices("AGENT_MODE", "HARNESS_MODE"),
    )
    mcp_url: str | None = Field(
        default="https://mcp.hackathon.lunit.io/mcp",
        validation_alias="LUNIT_MCP_URL",
    )
    max_mcp_calls: int = Field(
        default=6,
        ge=0,
        le=12,
        validation_alias=AliasChoices("MAX_MCP_CALLS", "MAX_TOOL_CALLS"),
    )
    max_tool_result_chars: int = Field(
        default=12_000,
        ge=1_000,
        le=100_000,
        validation_alias="MAX_TOOL_RESULT_CHARS",
    )
    max_evidence_chars: int = Field(
        default=24_000,
        ge=2_000,
        le=200_000,
        validation_alias="MAX_EVIDENCE_CHARS",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "HARNESS_LOG_LEVEL"),
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
    def chat_completions_url(self) -> str:
        return f"{self.lunit_fm_api_url.rstrip('/')}/v1/chat/completions"
