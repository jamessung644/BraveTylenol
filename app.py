import asyncio
import time
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Protocol

import httpx
from fastapi import FastAPI, Header, HTTPException

from harness.config import Settings
from harness.errors import (
    ConfigurationError,
    MalformedUpstreamResponseError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from harness.generation import GenerationEngine
from harness.l2_client import L2Client
from harness.mcp_client import MCPClient
from harness.orchestrator import ChatOrchestrator
from harness.retrieval import RetrievalEngine
from harness.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)


class ChatOrchestratorProtocol(Protocol):
    async def answer(self, messages: Sequence[ChatMessage]) -> str: ...


def create_app(
    settings: Settings | None = None,
    orchestrator: ChatOrchestratorProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_orchestrator = orchestrator

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.l2_http_client = httpx.AsyncClient()
        try:
            yield
        finally:
            await application.state.l2_http_client.aclose()

    application = FastAPI(lifespan=lifespan)

    def build_orchestrator(request_settings: Settings) -> ChatOrchestrator:
        http_client = getattr(application.state, "l2_http_client", None)
        l2 = L2Client(request_settings, http_client=http_client)
        if request_settings.harness_mode == "passthrough":
            return ChatOrchestrator(l2=l2, generation=None, mode="passthrough")
        retrieval = RetrievalEngine(l2, MCPClient(request_settings), request_settings)
        generation = GenerationEngine(l2, retrieval)
        return ChatOrchestrator(l2=l2, generation=generation, mode=request_settings.harness_mode)

    @application.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": "team-chatbot", "object": "model", "owned_by": "brave-tylenol"}],
        }

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
        bearer_token = None
        if authorization:
            scheme, separator, token = authorization.partition(" ")
            if separator and scheme.lower() == "bearer" and token.strip():
                bearer_token = token.strip()
        request_api_key = resolved_settings.api_key or bearer_token
        if not request_api_key:
            raise HTTPException(status_code=503, detail="LUNIT_FM_API_KEY is not configured")
        request_settings = resolved_settings.model_copy(update={"api_key": request_api_key})
        try:
            active_orchestrator = resolved_orchestrator or build_orchestrator(request_settings)
            async with asyncio.timeout(resolved_settings.upstream_timeout_seconds):
                answer = await active_orchestrator.answer(request.messages)
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail="L2 upstream timed out") from error
        except ConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except UpstreamTimeoutError as error:
            raise HTTPException(status_code=504, detail="L2 upstream timed out") from error
        except (UpstreamTransportError, UpstreamResponseError, MalformedUpstreamResponseError) as error:
            raise HTTPException(status_code=502, detail="L2 upstream request failed") from error

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


app = create_app()
