from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @field_validator("role", mode="before")
    @classmethod
    def normalize_developer_role(cls, value: Any) -> Any:
        # Modern OpenAI clients may emit `developer`; L2 receives the
        # equivalent protected instruction as a standard system message.
        return "system" if value == "developer" else value

    @field_validator("content", mode="before")
    @classmethod
    def normalize_text_content_parts(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        text_parts: list[str] = []
        for part in value:
            if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
                raise ValueError("Only text content parts are supported")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("Text content parts require a string text field")
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
    name: str
    description: str = ""
    input_schema: dict[str, Any]

    def as_chat_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class MCPCallResult(BaseModel):
    content: str
    is_error: bool = False
    cite_uids: list[str] = Field(default_factory=list)
    citation_contents: dict[str, str] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    cite_uid: str
    source_tool: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    content: str


class RetrievalResult(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[EvidenceItem] = Field(default_factory=list)
    note: str = ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = False

    @property
    def requested_max_tokens(self) -> int | None:
        requested = [
            value
            for value in (self.max_tokens, self.max_completion_tokens)
            if value is not None
        ]
        return min(requested) if requested else None


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
