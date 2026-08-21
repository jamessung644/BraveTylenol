import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Protocol

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import SecretStr

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.mcp_client import MCPClient
from lunit_hackathon.orchestrator import ChatOrchestrator
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

logger = logging.getLogger(__name__)


class OrchestratorProtocol(Protocol):
    last_usage: TokenUsage

    async def answer(self, messages: Sequence[ChatMessage]) -> str: ...


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
        if request_settings.agent_mode == "passthrough":
            return ChatOrchestrator(
                l2_client=l2,
                generation_engine=None,
                mode="passthrough",
            )

        # CoEval sends the API key in the request Bearer header and may not
        # provide an MCP endpoint. Without MCP, preserve the medical system
        # prompt but avoid a failed retrieval probe and a second L2 call.
        if not request_settings.mcp_url:
            return ChatOrchestrator(
                l2_client=l2,
                generation_engine=GenerationEngine(l2, retrieval_engine=None),
                mode="direct",
            )

        retrieval = RetrievalEngine(
            l2,
            MCPClient(request_settings),
            request_settings,
        )
        generation = GenerationEngine(l2, retrieval)
        return ChatOrchestrator(
            l2_client=l2,
            generation_engine=generation,
            mode="rag",
        )

    @application.middleware("http")
    async def sanitized_request_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            logger.error(
                "request_failed request_id=%s method=%s path=%s error_type=%s",
                request_id,
                request.method,
                request.url.path,
                type(error).__name__,
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
                    id="team-chatbot",
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
        # CoEval owns the credential carried by each request. Prefer it over a
        # possibly stale deployment-level fallback so one bad environment value
        # cannot make every otherwise valid evaluation request fail with 401.
        request_api_key = _bearer_token(authorization) or resolved_settings.api_key
        if not request_api_key:
            raise HTTPException(
                status_code=503,
                detail="LUNIT_FM_API_KEY is not configured",
            )

        request_settings = resolved_settings.model_copy(
            update={"lunit_fm_api_key": SecretStr(request_api_key)}
        )
        active_orchestrator = orchestrator or build_orchestrator(request_settings)
        try:
            async with asyncio.timeout(request_settings.request_timeout_seconds):
                answer = await active_orchestrator.answer(request.messages)
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="L2 upstream timed out") from error
        except ConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except UpstreamTimeoutError as error:
            raise HTTPException(status_code=504, detail="L2 upstream timed out") from error
        except (
            UpstreamTransportError,
            UpstreamResponseError,
            MalformedUpstreamResponseError,
        ) as error:
            raise HTTPException(
                status_code=502,
                detail="L2 upstream request failed",
            ) from error

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ],
            usage=getattr(active_orchestrator, "last_usage", TokenUsage()),
        )

    return application


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


app = create_app()
