from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_url: str = Field("https://model.hackathon.lunit.io", alias="LUNIT_FM_API_URL")
    api_key: str = Field("", alias="LUNIT_FM_API_KEY")
    model: str = Field("Lunit/L2-preview", alias="LUNIT_FM_MODEL")
    request_timeout_seconds: float = Field(
        65.0,
        gt=0,
        le=175,
        alias="REQUEST_TIMEOUT_SECONDS",
    )
    max_completion_tokens: int = Field(
        1024,
        ge=256,
        le=4096,
        alias="MAX_COMPLETION_TOKENS",
    )
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        "low",
        alias="LUNIT_REASONING_EFFORT",
    )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/chat/completions"


@lru_cache
def get_settings() -> Settings:
    return Settings()
