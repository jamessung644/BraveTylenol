from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", frozen=True)

    api_key: str | None = Field(default=None, validation_alias="LUNIT_FM_API_KEY")
    api_url: str = Field(default="https://model.hackathon.lunit.io", validation_alias="LUNIT_FM_API_URL")
    model_name: str = Field(default="Lunit/L2-preview", validation_alias="LUNIT_FM_MODEL")
    mcp_url: str = Field(default="https://mcp.hackathon.lunit.io/mcp", validation_alias="LUNIT_MCP_URL")
    harness_mode: Literal["rag", "passthrough"] = Field(default="passthrough", validation_alias="HARNESS_MODE")
    max_tool_calls: int = Field(default=4, ge=1, le=12, validation_alias="MAX_TOOL_CALLS")
    upstream_timeout_seconds: float = Field(default=150.0, gt=0, le=175, validation_alias="UPSTREAM_TIMEOUT_SECONDS")
    max_completion_tokens: int = Field(default=3_072, ge=512, le=6_144, validation_alias="MAX_COMPLETION_TOKENS")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="low", validation_alias="LUNIT_REASONING_EFFORT")
    max_tool_result_chars: int = Field(default=12_000, ge=1_000, validation_alias="MAX_TOOL_RESULT_CHARS")
    max_evidence_chars: int = Field(default=32_000, ge=2_000, validation_alias="MAX_EVIDENCE_CHARS")
