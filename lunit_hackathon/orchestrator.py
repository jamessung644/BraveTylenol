import logging
from collections.abc import Sequence
from typing import Any, Literal

from lunit_hackathon.errors import MalformedUpstreamResponseError, RetrievalError
from lunit_hackathon.prompts import DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT
from lunit_hackathon.schemas import ChatMessage, TokenUsage

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Request-scoped coordination that never authors a medical answer in Python."""

    def __init__(
        self,
        *,
        l2_client: Any,
        generation_engine: Any | None,
        mode: Literal["direct", "rag", "passthrough"],
    ) -> None:
        self._l2 = l2_client
        self._generation = generation_engine
        self._mode = mode
        self.last_usage = TokenUsage()
        self.last_finish_reason: str | None = None

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        if self._mode == "passthrough":
            return await self._passthrough(messages)
        if self._generation is None:
            raise RuntimeError("Medical generation mode requires a generation engine")

        # The evaluation container may not receive an MCP endpoint. Skip the
        # retrieval-decision round trip in that case and ask L2 for the final,
        # medically prompted answer in exactly one upstream call.
        if self._mode == "direct":
            answer = await self._generation.direct_answer(messages)
            self.last_usage = getattr(self._l2, "last_usage", TokenUsage())
            self.last_finish_reason = getattr(self._l2, "last_finish_reason", None)
            return _required_content(answer)

        try:
            answer = await self._generation.answer(messages)
        except RetrievalError as error:
            logger.warning(
                "retrieval_fallback error_code=%s",
                error.code,
            )
            answer = await self._generation.direct_answer(messages)

        self.last_usage = getattr(self._l2, "last_usage", TokenUsage())
        self.last_finish_reason = getattr(self._l2, "last_finish_reason", None)
        return _required_content(answer)

    async def _passthrough(self, messages: Sequence[ChatMessage]) -> str:
        protected_messages = [
            ChatMessage(role="system", content=DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT),
            *messages,
        ]
        completion = await self._l2.complete(messages=protected_messages)
        self.last_usage = getattr(self._l2, "last_usage", completion.usage)
        self.last_finish_reason = completion.finish_reason
        return _required_content(completion.content)


def _required_content(content: str | None) -> str:
    if content is None or not content.strip():
        raise MalformedUpstreamResponseError("L2 returned no final text")
    return content
