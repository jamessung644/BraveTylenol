import time
import uuid
from typing import Any

import httpx

from app.config import Settings
from app.fast_prompt import GENERATION_SYSTEM_PROMPT


class L2TimeoutError(RuntimeError):
    """The single upstream L2 request exceeded the configured deadline."""


class L2ResponseError(RuntimeError):
    """L2 returned an unusable HTTP or completion response."""


class FastL2Harness:
    """Speed-first driver that makes exactly one L2 call per conversation turn."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def answer(
        self,
        messages: list[dict[str, Any]],
        *,
        requested_max_tokens: int | None = None,
    ) -> dict[str, Any]:
        max_tokens = min(
            requested_max_tokens or self.settings.max_completion_tokens,
            self.settings.max_completion_tokens,
        )
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                *messages,
            ],
            "max_tokens": max_tokens,
            "reasoning_effort": self.settings.reasoning_effort,
            "temperature": 0.0,
        }

        try:
            response = await self._client.post(
                self.settings.chat_completions_url,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise L2TimeoutError("L2 request timed out") from exc
        except httpx.TransportError as exc:
            raise L2ResponseError("L2 service is unreachable") from exc

        if response.is_error:
            raise L2ResponseError(f"L2 returned HTTP {response.status_code}")

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise L2ResponseError("L2 returned an invalid completion") from exc
        if not isinstance(content, str) or not content.strip():
            raise L2ResponseError("L2 returned no final text")

        return {
            "id": body.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": body.get("created") or int(time.time()),
            "model": self.settings.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": body["choices"][0].get("finish_reason") or "stop",
                }
            ],
            "usage": body.get("usage"),
        }
