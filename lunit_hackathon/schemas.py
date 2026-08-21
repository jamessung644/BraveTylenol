from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)

CallId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[\x21-\x7e]+$"),
]
FunctionName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]
TransportToolName = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$"),
]
CitationUid = Annotated[str, StringConstraints(min_length=1, max_length=256)]
TOOL_CALL_FINISH_REASONS = frozenset({"stop", "tool_calls", "function_call"})


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: FunctionName
    arguments: str = Field(max_length=100_000)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: CallId
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Request-origin metadata is deliberately private: public callers cannot set
    # it through JSON and model_dump() never forwards it to L2.  The API boundary
    # uses this marker only for caller-supplied system/developer context that was
    # downgraded to data, so routing never mistakes it for a dialogue user turn.
    _internal_origin: str | None = PrivateAttr(default=None)

    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def normalize_text_parts(cls, value: Any) -> Any:
        """Accept the text-only OpenAI/CoEval content-part form.

        Images, audio, files, and tool payloads remain outside this public API.
        The original text order is preserved without attempting autocorrection.
        """

        if isinstance(value, str) or value is None:
            return value
        if not isinstance(value, list):
            return value
        text_parts: list[str] = []
        for part in value:
            if not isinstance(part, Mapping) or part.get("type") not in {
                "text",
                "input_text",
            }:
                raise ValueError("Only text content parts are supported")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Text content parts require a string")
            text_parts.append(text)
        return "".join(text_parts)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class L2Completion(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class MCPTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ``name`` is the exact server-returned transport name kept for MCP SDK
    # compatibility.  ``model_function_name`` is an explicit optional alias;
    # when absent, the binding compiler records an explicit identity mapping.
    name: TransportToolName
    model_function_name: FunctionName | None = None
    description: str = Field(default="", max_length=4_096)
    input_schema: dict[str, Any]

    @property
    def transport_tool_name(self) -> str:
        return self.name

    def as_chat_tool(self) -> dict[str, Any]:
        # Lazy import avoids a module cycle while ensuring callers cannot
        # accidentally expose the raw MCP schema to L2.
        from lunit_hackathon.tool_bindings import compile_tool_binding

        return compile_tool_binding(self).as_chat_tool()


class MCPCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(max_length=500_000)
    is_error: bool = False
    cite_uids: list[CitationUid] = Field(default_factory=list, max_length=256)
    citation_contents: dict[CitationUid, Annotated[str, StringConstraints(max_length=500_000)]] = (
        Field(default_factory=dict, max_length=256)
    )


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cite_uid: CitationUid
    source_tool: TransportToolName
    relevance_score: float = Field(ge=0.0, le=1.0)
    content: str = Field(max_length=200_000)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[EvidenceItem] = Field(default_factory=list, max_length=8)
    note: str = Field(default="", max_length=512)
    execution_status: Literal[
        "ok",
        "source_unavailable",
        "timeout",
        "schema_error",
        "budget_exhausted",
    ] = "ok"
    semantic_reason: Literal[
        "completed",
        "not_needed",
        "no_match",
        "low_quality",
        "coverage_gap",
        "conflict",
        "invalid_query",
    ] = "completed"
    routing_status: Literal["appropriate", "misrouted"] = "appropriate"

    @model_validator(mode="after")
    def validate_evidence_cardinality(self) -> Self:
        cite_uids = [item.cite_uid for item in self.items]
        if len(cite_uids) != len(set(cite_uids)):
            raise ValueError("retrieval evidence cite_uid values must be unique")
        if self.status == "sufficient" and not self.items:
            raise ValueError("sufficient retrieval requires evidence")
        if self.status == "no_evidence" and self.items:
            raise ValueError("no_evidence retrieval cannot contain evidence")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
