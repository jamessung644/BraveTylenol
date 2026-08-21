import asyncio
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from harness.config import Settings
from harness.errors import RetrievalError
from harness.mcp_client import MCPClientProtocol
from harness.prompts import FINALIZE_RETRIEVAL_TOOL, RETRIEVAL_SYSTEM_PROMPT
from harness.schemas import (
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    ToolCall,
)


class _FinalItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class _Finalization(BaseModel):
    status: str
    items: list[_FinalItem]
    note: str


class RetrievalEngine:
    def __init__(self, l2_client: Any, mcp_client: MCPClientProtocol, settings: Settings) -> None:
        self._l2 = l2_client
        self._mcp = mcp_client
        self._settings = settings

    async def retrieve(self, query: str) -> RetrievalResult:
        candidates: dict[str, tuple[str, str]] = {}
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=RETRIEVAL_SYSTEM_PROMPT),
            ChatMessage(role="user", content=query),
        ]
        calls_used = 0
        async with self._mcp.connect() as connection:
            discovered = await connection.list_tools()
            if any(tool.name == "finalize_retrieval" for tool in discovered):
                raise RetrievalError("MCP tool name collision: finalize_retrieval")
            tool_map = {tool.name: tool for tool in discovered}
            tools = [tool.as_openai_tool() for tool in discovered] + [FINALIZE_RETRIEVAL_TOOL]
            while calls_used < self._settings.max_tool_calls:
                completion: L2Completion = await self._l2.complete(messages=messages, tools=tools, tool_choice="auto")
                messages.append(ChatMessage(role="assistant", content=completion.content, tool_calls=completion.tool_calls))
                if not completion.tool_calls:
                    raise RetrievalError("Retrieval planner returned no tool calls")
                final_call = next((call for call in completion.tool_calls if call.function.name == "finalize_retrieval"), None)
                if final_call is not None:
                    finalization = self._parse_finalization(final_call)
                    return self._resolve(finalization, candidates)

                valid: list[tuple[ToolCall, dict[str, Any]]] = []
                errors: list[tuple[ToolCall, str]] = []
                remaining = self._settings.max_tool_calls - calls_used
                for call in completion.tool_calls:
                    arguments, error = _parse_arguments(call.function.arguments)
                    if error:
                        errors.append((call, error))
                    elif call.function.name not in tool_map:
                        errors.append((call, "unknown tool"))
                    elif len(valid) >= remaining:
                        errors.append((call, "tool-call budget exhausted"))
                    else:
                        valid.append((call, arguments))
                calls_used += len(valid)
                for call, error in errors:
                    messages.append(_tool_error(call, error))
                results = await asyncio.gather(*(self._call(connection, call, arguments) for call, arguments in valid))
                for call, (result, cite_uids) in zip((call for call, _ in valid), results, strict=True):
                    messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=result))
                    for cite_uid in cite_uids:
                        candidates.setdefault(cite_uid, (call.function.name, result))
                if calls_used >= self._settings.max_tool_calls:
                    return self._partial(candidates)
        return self._partial(candidates)

    async def _call(self, connection: Any, call: ToolCall, arguments: dict[str, Any]) -> tuple[str, list[str]]:
        try:
            result = await connection.call_tool(call.function.name, arguments)
            if result.is_error:
                return _error_content("MCP tool returned an error"), []
            return _truncate(result.content, self._settings.max_tool_result_chars), _cite_uids(result.content)
        except RetrievalError:
            return _error_content("MCP tool call failed"), []

    def _parse_finalization(self, call: ToolCall) -> _Finalization:
        arguments, error = _parse_arguments(call.function.arguments)
        if error:
            raise RetrievalError(f"Invalid finalize_retrieval arguments: {error}")
        try:
            finalization = _Finalization.model_validate(arguments)
        except ValidationError as error:
            raise RetrievalError("Invalid finalize_retrieval arguments") from error
        if finalization.status not in {"sufficient", "partial", "no_evidence"}:
            raise RetrievalError("Invalid finalize_retrieval status")
        return finalization

    def _resolve(self, finalization: _Finalization, candidates: Mapping[str, tuple[str, str]]) -> RetrievalResult:
        scores: dict[str, float] = {}
        for item in finalization.items:
            if item.cite_uid in candidates:
                scores[item.cite_uid] = max(scores.get(item.cite_uid, -1), item.relevance_score)
        selected = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        items = self._bounded_items(selected, candidates)
        status = finalization.status if items else "no_evidence"
        return RetrievalResult(status=status, items=items, note=_truncate(finalization.note, self._settings.max_evidence_chars))

    def _partial(self, candidates: Mapping[str, tuple[str, str]]) -> RetrievalResult:
        items = self._bounded_items([(cite_uid, 0.0) for cite_uid in candidates], candidates)
        return RetrievalResult(status="partial" if items else "no_evidence", items=items, note="MCP tool-call budget exhausted before finalization.")

    def _bounded_items(self, selected: list[tuple[str, float]], candidates: Mapping[str, tuple[str, str]]) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        remaining = self._settings.max_evidence_chars
        for cite_uid, score in selected:
            source_tool, content = candidates[cite_uid]
            if remaining <= 0:
                break
            capped = _truncate(content, remaining)
            items.append(EvidenceItem(cite_uid=cite_uid, relevance_score=score, source_tool=source_tool, content=capped))
            remaining -= len(capped)
        return items


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}, "invalid JSON arguments"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


def _tool_error(call: ToolCall, message: str) -> ChatMessage:
    return ChatMessage(role="tool", tool_call_id=call.id, content=_error_content(message))


def _error_content(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))


def _cite_uids(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return []
    found: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "cite_uid" and isinstance(item, str):
                    found.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return found


def _truncate(value: str, maximum: int) -> str:
    suffix = "...[truncated]"
    if len(value) <= maximum:
        return value
    if maximum <= len(suffix):
        return suffix[:maximum]
    return f"{value[: maximum - len(suffix)]}{suffix}"
