"""Request-scoped coordination of direct and retrieval-augmented answers."""

import logging
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from harness.errors import MalformedUpstreamResponseError, RetrievalError
from harness.schemas import ChatMessage, TokenUsage

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Runs one chat request without retaining retrieval state between requests."""

    def __init__(self, *, l2: Any, generation: Any | None, mode: Literal["rag", "passthrough"]) -> None:
        self._l2 = l2
        self._generation = generation
        self._mode = mode
        self.last_usage = TokenUsage()

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        if self._mode == "passthrough":
            return await self._direct(messages)

        try:
            if self._generation is None:
                raise RuntimeError("RAG mode requires a generation engine")
            answer = await self._generation.answer(messages)
        except RetrievalError as error:
            logger.warning(
                "retrieval fallback request_id=%s error_type=%s",
                uuid.uuid4(),
                type(error).__name__,
            )
            return await self._direct(messages)

        self.last_usage = getattr(self._l2, "last_usage", TokenUsage())
        return _required_content(answer)

    async def _direct(self, messages: Sequence[ChatMessage]) -> str:
        completion = await self._l2.complete(messages=messages)
        self.last_usage = completion.usage
        return _required_content(completion.content)


def _required_content(content: str | None) -> str:
    if content is None or not content.strip():
        raise MalformedUpstreamResponseError("Upstream model returned no final text")
    return content
