import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Literal

from lunit_hackathon.errors import MalformedUpstreamResponseError
from lunit_hackathon.generation import GenerationEngine, requires_retrieval
from lunit_hackathon.schemas import ChatMessage, TokenUsage

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Request-scoped coordination that never authors a medical answer in Python."""

    def __init__(
        self,
        *,
        l2_client: Any,
        generation_engine: Any | None,
        mode: Literal["direct", "hybrid", "rag", "passthrough"],
        rag_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._l2 = l2_client
        self._generation = generation_engine
        self._mode = mode
        self._rag_semaphore = rag_semaphore
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
            return await self._direct(messages)

        if self._mode == "hybrid":
            if not requires_retrieval(messages):
                return await self._direct(messages)

        if self._rag_semaphore is None:
            return await self._rag(messages)
        # ``Semaphore.acquire`` completes synchronously while a permit is
        # available.  Check saturation before awaiting so admission never
        # depends on an arbitrary event-loop timer and never queues behind a
        # full RAG cohort into the final-answer reserve.
        if self._rag_semaphore.locked():
            # Saturation must not queue into the final-answer reserve. Only a
            # source-dependent request needs the explicit no-evidence final;
            # forced-RAG traffic without source dependency can remain direct.
            if requires_retrieval(messages):
                logger.info("rag_admission_full route=evidence_unavailable")
                return await self._evidence_unavailable(messages)
            logger.info("rag_admission_full route=direct")
            return await self._direct(messages)
        await self._rag_semaphore.acquire()
        try:
            return await self._rag(messages)
        finally:
            self._rag_semaphore.release()

    async def _direct(self, messages: Sequence[ChatMessage]) -> str:
        answer = await self._generation.direct_answer(messages)
        self._update_usage()
        return _required_content(answer)

    async def _rag(self, messages: Sequence[ChatMessage]) -> str:
        # GenerationEngine maps Retrieval success or failure into a fresh,
        # phase-specific no-tool final transcript and validates its plain answer.
        answer = await self._generation.answer(messages)
        self._update_usage()
        return _required_content(answer)

    async def _evidence_unavailable(self, messages: Sequence[ChatMessage]) -> str:
        answer = await self._generation.evidence_unavailable_answer(messages)
        self._update_usage()
        return _required_content(answer)

    def _update_usage(self) -> None:
        self.last_usage = getattr(self._l2, "last_usage", TokenUsage())
        self.last_finish_reason = getattr(self._l2, "last_finish_reason", None)

    async def _passthrough(self, messages: Sequence[ChatMessage]) -> str:
        # Passthrough skips evidence coordination, but it may not bypass the
        # final-answer prompt, validator, or single bounded recovery invariant.
        generation = self._generation or GenerationEngine(
            self._l2,
            retrieval_engine=None,
        )
        answer = await generation.direct_answer(messages)
        self._update_usage()
        return _required_content(answer)


def _required_content(content: str | None) -> str:
    if content is None or not content.strip():
        raise MalformedUpstreamResponseError("L2 returned no final text")
    return content
