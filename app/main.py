from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.fast_harness import FastL2Harness, L2ResponseError, L2TimeoutError
from app.schemas import ChatCompletionRequest, ModelCard, ModelList


def create_app(
    *,
    settings: Settings | None = None,
    harness: FastL2Harness | None = None,
) -> FastAPI:
    configured = settings or get_settings()
    owned_harness = harness is None and bool(configured.api_key)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.harness = harness
        if application.state.harness is None and configured.api_key:
            application.state.harness = FastL2Harness(configured)
        yield
        if owned_harness and application.state.harness is not None:
            await application.state.harness.aclose()

    application = FastAPI(
        title="Brave Tylenol L2 Fast Baseline",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models", response_model=ModelList)
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=configured.model)])

    @application.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if not configured.api_key:
            raise HTTPException(status_code=503, detail="LUNIT_FM_API_KEY is not configured")
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="The last message must have role=user")

        driver: FastL2Harness = application.state.harness
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


app = create_app()
