import logging
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from lunit_hackathon.errors import MalformedUpstreamResponseError, RetrievalError
from lunit_hackathon.schemas import ChatMessage, TokenUsage

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Request-scoped coordination that never authors a medical answer in Python."""

    def __init__(
        self,
        *,
        l2_client: Any,
        generation_engine: Any | None,
        mode: Literal["rag", "passthrough"],
    ) -> None:
        self._l2 = l2_client
        self._generation = generation_engine
        self._mode = mode
        self.last_usage = TokenUsage()

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        if self._mode == "passthrough":
            return await self._passthrough(messages)
        if self._generation is None:
            raise RuntimeError("RAG mode requires a generation engine")

        try:
            answer = await self._generation.answer(messages)
        except RetrievalError as error:
            logger.warning(
                "retrieval_fallback request_id=%s error_type=%s",
                uuid.uuid4(),
                type(error).__name__,
            )
            answer = await self._generation.direct_answer(messages)

        self.last_usage = getattr(self._l2, "last_usage", TokenUsage())
        return _required_content(answer)

    async def _passthrough(self, messages: Sequence[ChatMessage]) -> str:
        completion = await self._l2.complete(messages=messages)
        self.last_usage = completion.usage
        return _required_content(completion.content)


def _required_content(content: str | None) -> str:
    if content is None or not content.strip():
        raise MalformedUpstreamResponseError("L2 returned no final text")
    return content
