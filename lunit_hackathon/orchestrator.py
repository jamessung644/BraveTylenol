import logging
from collections.abc import Sequence
from typing import Any, Literal

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.routing import route_messages, self_contained_query
from lunit_hackathon.schemas import ChatMessage, TokenUsage

logger = logging.getLogger(__name__)

MEDICAL_SAFETY_FALLBACK = (
    "질문을 확인했습니다. 증상이 심하거나 갑자기 악화되면 즉시 119 또는 "
    "응급실의 도움을 받고, 정확한 판단을 위해 의료 전문가와 상담해 주세요."
)
_EXPECTED_L2_ERRORS = (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
    UpstreamResponseError,
    TimeoutError,
)


class ChatOrchestrator:
    """Request-scoped score-first coordination with a bounded safety recovery."""

    def __init__(
        self,
        *,
        l2_client: Any,
        retrieval_engine: Any | None,
        generation_engine: Any | None,
        answer_verifier: Any | None,
        settings: Settings,
        mode: Literal["direct", "rag", "passthrough"],
    ) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine
        self._generation = generation_engine
        self._verifier = answer_verifier
        self._settings = settings
        self._mode = mode
        self.last_usage = TokenUsage()
        self.last_finish_reason = "stop"

    async def answer(
        self,
        messages: Sequence[ChatMessage],
        deadline: RequestDeadline,
    ) -> str:
        route = route_messages(messages)
        if self._mode != "rag" or not route.retrieval_required:
            return await self._direct_or_fallback(messages, deadline)

        if self._retrieval is None:
            raise RuntimeError("RAG mode requires a retrieval engine")
        if self._generation is None:
            raise RuntimeError("Medical generation mode requires a generation engine")

        query = self_contained_query(messages)
        try:
            retrieval = await self._retrieval.retrieve(query, route, deadline)
        except RetrievalError as error:
            _log_expected_failure("retrieval_direct_recovery", error)
            return await self._direct_or_fallback(messages, deadline)

        try:
            candidate = _required_content(
                await self._generation.grounded_answer(messages, retrieval, deadline)
            )
        except _EXPECTED_L2_ERRORS as error:
            _log_expected_failure("grounded_failure", error)
            if deadline.can_spend(10.0):
                return await self._direct_or_fallback(messages, deadline)
            return self._finalize(MEDICAL_SAFETY_FALLBACK)

        if (
            route.verification_required
            and deadline.can_spend(self._settings.verification_minimum_seconds)
        ):
            if self._verifier is None:
                raise RuntimeError("Verification-required route has no verifier")
            try:
                candidate = _required_content(
                    await self._verifier.verify(messages, candidate, retrieval, deadline)
                )
            except _EXPECTED_L2_ERRORS as error:
                _log_expected_failure("verifier_candidate_retained", error)

        return self._finalize(candidate)

    async def _direct_or_fallback(
        self,
        messages: Sequence[ChatMessage],
        deadline: RequestDeadline,
    ) -> str:
        if self._generation is None:
            raise RuntimeError("Medical generation mode requires a generation engine")
        try:
            answer = await self._generation.direct_answer(messages, deadline)
            return self._finalize(_required_content(answer))
        except _EXPECTED_L2_ERRORS as error:
            _log_expected_failure("direct_static_fallback", error)
            return self._finalize(MEDICAL_SAFETY_FALLBACK)

    def _finalize(self, content: str) -> str:
        self.last_usage = _nonnegative_usage(getattr(self._l2, "last_usage", TokenUsage()))
        self.last_finish_reason = "stop"
        return content


def _required_content(content: str | None) -> str:
    if content is None or not content.strip():
        raise MalformedUpstreamResponseError("L2 returned no final text")
    return content.strip()


def _nonnegative_usage(value: Any) -> TokenUsage:
    prompt_tokens = _nonnegative_int(getattr(value, "prompt_tokens", 0))
    completion_tokens = _nonnegative_int(getattr(value, "completion_tokens", 0))
    total_tokens = max(
        _nonnegative_int(getattr(value, "total_tokens", 0)),
        prompt_tokens + completion_tokens,
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _log_expected_failure(event: str, error: BaseException) -> None:
    code = error.code if isinstance(error, RetrievalError) else _error_code(error)
    logger.warning("%s error_code=%s", event, code)


def _error_code(error: BaseException) -> str:
    if isinstance(error, ConfigurationError):
        return "configuration"
    if isinstance(error, UpstreamTimeoutError | TimeoutError):
        return "timeout"
    if isinstance(error, UpstreamTransportError):
        return "transport"
    if isinstance(error, MalformedUpstreamResponseError):
        return "malformed"
    if isinstance(error, UpstreamResponseError):
        return "upstream_response"
    return "unknown"
