import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Protocol

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import SecretStr

from lunit_hackathon.config import Settings
from lunit_hackathon.credentials import resolve_lunit_api_key
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.mcp_client import MCPClient
from lunit_hackathon.orchestrator import MEDICAL_SAFETY_FALLBACK, ChatOrchestrator
from lunit_hackathon.retrieval import RetrievalEngine
from lunit_hackathon.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    TokenUsage,
)
from lunit_hackathon.verification import AnswerVerifier

logger = logging.getLogger(__name__)
_EVALUATOR_MODEL_ID = "team-chatbot"
_EXPECTED_RECOVERY_ERRORS = (
    ConfigurationError,
    RetrievalError,
    MalformedUpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
    UpstreamResponseError,
    TimeoutError,
)


class OrchestratorProtocol(Protocol):
    last_usage: TokenUsage
    last_finish_reason: str

    async def answer(
        self,
        messages: Sequence[ChatMessage],
        deadline: RequestDeadline,
    ) -> str: ...


def create_app(
    settings: Settings | None = None,
    orchestrator: OrchestratorProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.l2_http_client = httpx.AsyncClient()
        try:
            yield
        finally:
            await application.state.l2_http_client.aclose()

    application = FastAPI(
        title="Lunit L2 Medical Chat",
        version="1.0.0",
        lifespan=lifespan,
    )

    def build_orchestrator(request_settings: Settings) -> ChatOrchestrator:
        shared_http = getattr(application.state, "l2_http_client", None)
        l2 = L2Client(request_settings, http_client=shared_http)
        generation = GenerationEngine(l2, request_settings)
        verifier = AnswerVerifier(l2, request_settings)
        retrieval = None
        mode = request_settings.agent_mode
        if mode == "rag" and request_settings.mcp_url:
            retrieval = RetrievalEngine(
                l2,
                MCPClient(request_settings),
                request_settings,
            )
        elif mode == "rag":
            mode = "direct"
        return ChatOrchestrator(
            l2_client=l2,
            retrieval_engine=retrieval,
            generation_engine=generation,
            answer_verifier=verifier,
            settings=request_settings,
            mode=mode,
        )

    @application.middleware("http")
    async def sanitized_request_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.error(
                "request_failed request_id=%s method=%s path=%s error_code=unexpected",
                request_id,
                request.method,
                request.url.path,
            )
            raise
        duration_ms = (perf_counter() - started) * 1_000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @application.get("/health")
    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=_EVALUATOR_MODEL_ID,
                    owned_by="brave-tylenol",
                )
            ]
        )

    @application.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        response_model_exclude_none=True,
    )
    async def create_chat_completion(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")

        deadline = RequestDeadline.start(
            total_seconds=resolved_settings.request_timeout_seconds,
        )
        active_orchestrator: OrchestratorProtocol | None = None
        try:
            request_settings = _request_settings(
                resolved_settings,
                authorization,
                request.max_tokens,
            )
            active_orchestrator = orchestrator or build_orchestrator(request_settings)
            async with asyncio.timeout(deadline.remaining()):
                answer = await active_orchestrator.answer(request.messages, deadline)
            if not isinstance(answer, str) or not answer.strip():
                answer = MEDICAL_SAFETY_FALLBACK
        except _EXPECTED_RECOVERY_ERRORS as error:
            logger.warning("completion_recovery error_code=%s", _error_code(error))
            answer = MEDICAL_SAFETY_FALLBACK

        usage = _nonnegative_usage(
            getattr(active_orchestrator, "last_usage", TokenUsage()),
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=_EVALUATOR_MODEL_ID,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    return application


def _request_settings(
    settings: Settings,
    authorization: str | None,
    requested_max_tokens: int | None,
) -> Settings:
    api_key = resolve_lunit_api_key(authorization, settings.api_key)
    updates: dict[str, Any] = {"lunit_fm_api_key": SecretStr(api_key)}
    if requested_max_tokens is not None:
        updates["max_completion_tokens"] = min(
            requested_max_tokens,
            settings.max_completion_tokens,
        )
    return settings.model_copy(update=updates)


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


def _error_code(error: BaseException) -> str:
    if isinstance(error, ConfigurationError):
        return "configuration"
    if isinstance(error, RetrievalError):
        return error.code
    if isinstance(error, UpstreamTimeoutError | TimeoutError):
        return "timeout"
    if isinstance(error, UpstreamTransportError):
        return "transport"
    if isinstance(error, MalformedUpstreamResponseError):
        return "malformed"
    if isinstance(error, UpstreamResponseError):
        return "upstream_response"
    return "unknown"


app = create_app()
