from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException

from app.config import Settings, get_settings
from app.fast_harness import FastL2Harness, L2ResponseError, L2TimeoutError
from app.schemas import ChatCompletionRequest, ModelCard, ModelList


def create_app(
    *,
    settings: Settings | None = None,
    harness: FastL2Harness | None = None,
) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(configured.request_timeout_seconds, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        try:
            yield
        finally:
            await application.state.http_client.aclose()

    application = FastAPI(
        title="Brave Tylenol L2 Fast Baseline",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=configured.model)])

    @application.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str | None = Header(default=None),
    ):
        request_api_key = configured.api_key or _bearer_token(authorization)
        if not request_api_key:
            raise HTTPException(status_code=503, detail="LUNIT_FM_API_KEY is not configured")
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="The last message must have role=user")

        request_settings = configured.model_copy(update={"api_key": request_api_key})
        driver = harness or FastL2Harness(
            request_settings,
            http_client=application.state.http_client,
        )
        try:
            return await driver.answer(
                [message.model_dump(exclude_none=True) for message in request.messages],
                requested_max_tokens=request.max_tokens,
            )
        except L2TimeoutError as exc:
            raise HTTPException(status_code=504, detail="L2 upstream timed out") from exc
        except L2ResponseError as exc:
            raise HTTPException(status_code=502, detail="L2 upstream request failed") from exc

    return application


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    return token.strip() or None


app = create_app()
