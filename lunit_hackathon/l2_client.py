import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Self

import httpx
from pydantic import ValidationError

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.schemas import ChatMessage, L2Completion, TokenUsage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


class L2Client:
    """Async client for Lunit L2's OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self.last_usage = TokenUsage()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> L2Completion:
        api_key = self._settings.api_key
        if not api_key:
            raise ConfigurationError("LUNIT_FM_API_KEY is required in .env or the environment")

        payload: dict[str, Any] = {
            "model": self._settings.lunit_fm_model,
            "messages": [self._serialize_message(message) for message in messages],
        }
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        response = await self._post(payload, api_key)
        completion = self._parse_completion(response)
        self.last_usage = completion.usage
        return completion

    async def _post(self, payload: Mapping[str, Any], api_key: str) -> httpx.Response:
        client = self._client()
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)

        for attempt in range(self._settings.retry_attempts + 1):
            try:
                response = await client.post(
                    self._settings.chat_completions_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=timeout,
                )
            except httpx.TimeoutException as error:
                raise UpstreamTimeoutError("L2 request timed out") from error
            except httpx.TransportError as error:
                raise UpstreamTransportError("L2 service is unreachable") from error

            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < self._settings.retry_attempts
            ):
                delay = min(0.25 * (2**attempt), 1.0)
                logger.warning(
                    "l2_retry attempt=%d status=%d",
                    attempt + 1,
                    response.status_code,
                )
                await asyncio.sleep(delay)
                continue
            if response.is_error:
                raise UpstreamResponseError(self._sanitized_error(response))
            return response

        raise AssertionError("unreachable")

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    @staticmethod
    def _serialize_message(message: ChatMessage | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(message, ChatMessage):
            return message.model_dump(exclude_none=True)
        return dict(message)

    @staticmethod
    def _parse_completion(response: httpx.Response) -> L2Completion:
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MalformedUpstreamResponseError("L2 returned invalid JSON") from error

        try:
            message = payload["choices"][0]["message"]
            completion = L2Completion(
                content=message.get("content"),
                tool_calls=message.get("tool_calls") or [],
                usage=payload.get("usage") or {},
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValidationError) as error:
            raise MalformedUpstreamResponseError("L2 returned an invalid completion") from error

        if not (completion.content and completion.content.strip()) and not completion.tool_calls:
            raise MalformedUpstreamResponseError("L2 returned no content or tool calls")
        return completion

    @staticmethod
    def _sanitized_error(response: httpx.Response) -> str:
        error_type = "unknown"
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            raw_type = payload["error"].get("type")
            if isinstance(raw_type, str):
                error_type = re.sub(r"[^A-Za-z0-9_.-]", "", raw_type)[:80] or "unknown"
        return f"L2 returned HTTP {response.status_code} ({error_type})"
