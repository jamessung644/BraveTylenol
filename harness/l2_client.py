import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import httpx
from pydantic import ValidationError

from harness.config import Settings
from harness.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from harness.schemas import ChatMessage, L2Completion, TokenUsage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class L2Client:
    """Async client for the L2 OpenAI-compatible chat-completions endpoint."""

    _RETRYABLE_STATUS_CODES: ClassVar[set[int]] = {429, 502, 503, 504}

    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> None:
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
        if not self._settings.api_key:
            raise ConfigurationError("LUNIT_FM_API_KEY is required for L2 requests")

        payload: dict[str, Any] = {
            "model": self._settings.model_name,
            "messages": [self._serialize_message(message) for message in messages],
        }
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        started = time.monotonic()
        response = await self._post_completion(payload)
        try:
            completion = self._parse_completion(response)
        except MalformedUpstreamResponseError as error:
            logger.warning("l2 parse_failure error_type=%s", type(error).__name__)
            raise
        self.last_usage = _sum_usage(self.last_usage, completion.usage)
        logger.info(
            "l2 completion duration_ms=%d prompt_tokens=%d completion_tokens=%d tool_count=%d returned_tool_calls=%d accumulated_tokens=%d",
            round((time.monotonic() - started) * 1_000),
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            len(tools or []),
            len(completion.tool_calls),
            self.last_usage.total_tokens,
        )
        return completion

    async def _post_completion(self, payload: Mapping[str, Any]) -> httpx.Response:
        client = self._get_http_client()
        url = f"{self._settings.api_url.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._settings.api_key}"}
        timeout = httpx.Timeout(self._settings.upstream_timeout_seconds)

        for attempt in range(self._settings.l2_max_attempts):
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=timeout)
            except httpx.TimeoutException as error:
                logger.warning("l2 request_failure error_type=timeout")
                raise UpstreamTimeoutError("The upstream model request timed out") from error
            except httpx.TransportError as error:
                logger.warning("l2 request_failure error_type=transport")
                raise UpstreamTransportError("The upstream model could not be reached") from error

            if response.status_code in self._RETRYABLE_STATUS_CODES and attempt < self._settings.l2_max_attempts - 1:
                logger.info(
                    "l2 retry status=%d attempt=%d max_attempts=%d",
                    response.status_code,
                    attempt + 1,
                    self._settings.l2_max_attempts,
                )
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            if response.is_error:
                message = self._response_error_message(response)
                logger.warning(
                    "l2 request_failure error_type=response detail=%s attempt=%d max_attempts=%d message_count=%d tool_count=%d payload_chars=%d",
                    message,
                    attempt + 1,
                    self._settings.l2_max_attempts,
                    len(payload.get("messages", [])),
                    len(payload.get("tools", [])),
                    len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
                )
                raise UpstreamResponseError(message)
            return response

        raise AssertionError("unreachable")

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    @staticmethod
    def _serialize_message(message: ChatMessage | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(message, ChatMessage):
            return message.model_dump(exclude_none=True)
        return dict(message)

    @staticmethod
    def _response_error_message(response: httpx.Response) -> str:
        error_type = "unknown"
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("type"), str):
                error_type = re.sub(r"[^A-Za-z0-9_.-]", "", error["type"])[:80] or "unknown"
        return f"Upstream model returned HTTP {response.status_code} ({error_type})"

    @staticmethod
    def _parse_completion(response: httpx.Response) -> L2Completion:
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MalformedUpstreamResponseError("Upstream model returned invalid JSON") from error

        try:
            choices = payload["choices"]
            message = choices[0]["message"]
            completion = L2Completion(
                content=message.get("content"),
                tool_calls=message.get("tool_calls") or [],
                usage=payload.get("usage") or {},
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValidationError) as error:
            raise MalformedUpstreamResponseError("Upstream model returned an invalid completion") from error

        if not (completion.content and completion.content.strip()) and not completion.tool_calls:
            raise MalformedUpstreamResponseError("Upstream model returned no content or tool calls")
        return completion


def _sum_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )
