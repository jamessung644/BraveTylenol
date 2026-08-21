import time
import uuid
from collections.abc import Sequence
from typing import Protocol

from fastapi import FastAPI, HTTPException

from harness.config import Settings
from harness.errors import ConfigurationError
from harness.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TokenUsage,
)


class ChatOrchestratorProtocol(Protocol):
    async def answer(self, messages: Sequence[ChatMessage]) -> str: ...


class _UnavailableOrchestrator:
    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        del messages
        raise ConfigurationError("The chat orchestrator is not configured")


def create_app(
    settings: Settings | None = None,
    orchestrator: ChatOrchestratorProtocol | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_orchestrator = orchestrator or _UnavailableOrchestrator()
    application = FastAPI()

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
    async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        if not resolved_settings.api_key:
            raise HTTPException(status_code=503, detail="LUNIT_FM_API_KEY is not configured")
        try:
            answer = await resolved_orchestrator.answer(request.messages)
        except ConfigurationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

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
            usage=TokenUsage(),
        )

    return application


app = create_app()
