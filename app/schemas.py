from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "용맹한타이레놀"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class CitableItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0, le=1)


class CitationSelection(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[CitableItem] = Field(default_factory=list)
    note: str = ""


class RetrievalEvidence(BaseModel):
    cite_uid: str
    source_type: str = "unknown"
    title: str = "Untitled source"
    url: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
