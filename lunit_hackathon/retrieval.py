"""Bounded, routed MCP evidence retrieval."""

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import validators
from pydantic import BaseModel, Field, ValidationError

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import RetrievalError
from lunit_hackathon.evidence import (
    authority_rank,
    extract_evidence_metadata,
    rank_deduplicate_and_bound,
)
from lunit_hackathon.mcp_client import MCPClientProtocol, bound_mcp_content, collect_cite_uids
from lunit_hackathon.prompts import FINALIZE_RETRIEVAL_TOOL, RETRIEVAL_PLANNER_SYSTEM_PROMPT
from lunit_hackathon.schemas import (
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    RouteDecision,
    ToolCall,
)

logger = logging.getLogger(__name__)

_MAX_PLANNER_TURNS = 3
_MAX_EXECUTION_WAVES = 2
_MAX_MCP_CALLS = 6

type _ValidCall = tuple[ToolCall, dict[str, Any]]
type _CallError = tuple[ToolCall, str]


class _FinalItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class _Finalization(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[_FinalItem]
    note: str


@dataclass(frozen=True)
class _Candidate:
    source_tool: str
    content: str


@dataclass(frozen=True)
class _CallOutcome:
    content: str
    cite_uids: list[str]
    citation_contents: dict[str, str]


class RetrievalEngine:
    """Lets L2 plan two routed MCP execution waves and finalize real evidence."""

    def __init__(
        self,
        l2_client: Any,
        mcp_client: MCPClientProtocol,
        settings: Settings,
    ) -> None:
        self._l2 = l2_client
        self._mcp = mcp_client
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        route: RouteDecision,
        deadline: RequestDeadline,
    ) -> RetrievalResult:
        timeout = deadline.stage_timeout(
            self._settings.retrieval_timeout_seconds,
            reserve_seconds=self._settings.generation_timeout_seconds,
        )
        if timeout <= 0:
            raise RetrievalError(
                "Retrieval deadline is exhausted",
                code="retrieval_deadline_exhausted",
            )

        try:
            async with asyncio.timeout(timeout):
                return await self._retrieve(query, route)
        except TimeoutError as error:
            raise RetrievalError(
                "Retrieval deadline is exhausted",
                code="retrieval_deadline_exhausted",
            ) from error

    async def _retrieve(self, query: str, route: RouteDecision) -> RetrievalResult:
        candidates: dict[str, _Candidate] = {}
        messages = [
            ChatMessage(role="system", content=RETRIEVAL_PLANNER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=query),
        ]
        calls_used = 0
        execution_waves = 0
        planner_turns = 0
        recovery_used = False
        force_recovery = False
        seen_calls: set[str] = set()

        try:
            async with self._mcp.connect() as connection:
                try:
                    discovered = await connection.list_tools()
                except RetrievalError:
                    raise
                except Exception as error:
                    raise RetrievalError(
                        "MCP tool discovery failed",
                        code="mcp_tool_discovery_failed",
                    ) from error

                if any(tool.name == "finalize_retrieval" for tool in discovered):
                    raise RetrievalError(
                        "MCP tool name collision: finalize_retrieval",
                        code="mcp_tool_name_collision",
                    )
                for discovered_tool in discovered:
                    _validate_schema(discovered_tool.input_schema)

                routed_tools = [
                    discovered_tool
                    for discovered_tool in discovered
                    if discovered_tool.name in route.tool_names
                ]
                if not routed_tools:
                    raise RetrievalError(
                        "No discovered MCP tools are allowed by this route",
                        code="mcp_no_routed_tools",
                    )

                tool_map = {
                    discovered_tool.name: discovered_tool for discovered_tool in routed_tools
                }
                routed_chat_tools = [tool.as_chat_tool() for tool in routed_tools]
                call_budget = min(self._settings.max_mcp_calls, _MAX_MCP_CALLS)
                logger.info(
                    "retrieval_discovery discovered_tools=%d routed_tools=%d call_budget=%d",
                    len(discovered),
                    len(routed_tools),
                    call_budget,
                )

                while planner_turns < _MAX_PLANNER_TURNS:
                    final_only = (
                        execution_waves >= _MAX_EXECUTION_WAVES or calls_used >= call_budget
                    )
                    planner_tools: list[dict[str, Any]]
                    tool_choice: str | dict[str, Any]
                    if force_recovery:
                        preferred = routed_tools[0]
                        planner_tools = [preferred.as_chat_tool()]
                        tool_choice = {
                            "type": "function",
                            "function": {"name": preferred.name},
                        }
                        force_recovery = False
                    elif final_only:
                        planner_tools = [FINALIZE_RETRIEVAL_TOOL]
                        tool_choice = {
                            "type": "function",
                            "function": {"name": "finalize_retrieval"},
                        }
                    else:
                        planner_tools = [*routed_chat_tools, FINALIZE_RETRIEVAL_TOOL]
                        tool_choice = "auto"

                    completion = await self._planner_completion(
                        messages=messages,
                        tools=planner_tools,
                        tool_choice=tool_choice,
                    )
                    planner_turns += 1
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=completion.content,
                            tool_calls=completion.tool_calls,
                        )
                    )

                    finalization, valid_calls, errors = self._validate_calls(
                        completion.tool_calls,
                        tool_map,
                        seen_calls,
                        call_budget - calls_used,
                    )
                    for tool_call, message in errors:
                        messages.append(_tool_error(tool_call, message))

                    if finalization is not None:
                        resolved = self._resolve(finalization, candidates)
                        _log_completion(resolved, calls_used, planner_turns)
                        return resolved

                    if not valid_calls:
                        if (
                            not recovery_used
                            and not final_only
                            and calls_used < call_budget
                            and execution_waves < _MAX_EXECUTION_WAVES
                        ):
                            recovery_used = True
                            force_recovery = True
                            continue
                        continue

                    calls_used += len(valid_calls)
                    execution_waves += 1
                    outcomes = await asyncio.gather(
                        *(
                            self._call(connection, tool_call, arguments)
                            for tool_call, arguments in valid_calls
                        ),
                        return_exceptions=True,
                    )
                    usable_in_wave = False
                    for (tool_call, _), outcome in zip(valid_calls, outcomes, strict=True):
                        if isinstance(outcome, asyncio.CancelledError):
                            raise outcome
                        if isinstance(outcome, BaseException):
                            messages.append(_tool_error(tool_call, "MCP tool call failed"))
                            continue
                        messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=tool_call.id,
                                content=outcome.content,
                            )
                        )
                        for cite_uid in outcome.cite_uids:
                            candidates.setdefault(
                                cite_uid,
                                _Candidate(
                                    source_tool=tool_call.function.name,
                                    content=outcome.citation_contents.get(
                                        cite_uid, outcome.content
                                    ),
                                ),
                            )
                            usable_in_wave = True

                    if not usable_in_wave and not candidates:
                        raise RetrievalError(
                            "All MCP calls in the retrieval wave failed",
                            code="mcp_all_calls_failed",
                        )
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError(
                "MCP connection failed",
                code="mcp_connection_failed",
            ) from error

        partial = self._partial(
            candidates,
            note="Retrieval planner turn limit exhausted before finalization.",
        )
        _log_completion(partial, calls_used, planner_turns)
        return partial

    async def _planner_completion(self, **kwargs: Any) -> L2Completion:
        try:
            return await self._l2.complete(**kwargs)
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError(
                "Retrieval planner failed",
                code="retrieval_planner_failed",
            ) from error

    def _validate_calls(
        self,
        calls: list[ToolCall],
        tool_map: Mapping[str, Any],
        seen_calls: set[str],
        remaining_calls: int,
    ) -> tuple[_Finalization | None, list[_ValidCall], list[_CallError]]:
        finalization: _Finalization | None = None
        valid: list[_ValidCall] = []
        errors: list[_CallError] = []

        for tool_call in calls:
            if tool_call.function.name == "finalize_retrieval":
                if finalization is None:
                    finalization = self._parse_finalization(tool_call)
                else:
                    errors.append((tool_call, "duplicate retrieval finalization"))
                continue

            arguments, error = _parse_arguments(tool_call.function.arguments)
            if error is not None:
                errors.append((tool_call, error))
            elif tool_call.function.name not in tool_map:
                errors.append((tool_call, "unknown or unrouted tool"))
            elif not _arguments_match_schema(
                arguments,
                tool_map[tool_call.function.name].input_schema,
            ):
                errors.append((tool_call, "arguments do not match tool schema"))
            elif _call_signature(tool_call.function.name, arguments) in seen_calls:
                errors.append((tool_call, "duplicate tool call"))
            elif len(valid) >= remaining_calls:
                errors.append((tool_call, "tool-call budget exhausted"))
            else:
                seen_calls.add(_call_signature(tool_call.function.name, arguments))
                valid.append((tool_call, arguments))

        return finalization, valid, errors

    async def _call(
        self,
        connection: Any,
        tool_call: ToolCall,
        arguments: dict[str, Any],
    ) -> _CallOutcome:
        result = await connection.call_tool(tool_call.function.name, arguments)
        if result.is_error:
            raise RetrievalError("MCP tool returned an error", code="mcp_tool_call_failed")
        cite_uids = [
            cite_uid
            for cite_uid in (result.cite_uids or collect_cite_uids(result.content))
            if cite_uid.strip()
        ]
        content = bound_mcp_content(
            result.content,
            self._settings.max_tool_result_chars,
            cite_uids,
        )
        citation_contents = {
            cite_uid: bound_mcp_content(
                fragment,
                self._settings.max_tool_result_chars,
                [cite_uid],
            )
            for cite_uid, fragment in result.citation_contents.items()
            if cite_uid in cite_uids
        }
        return _CallOutcome(content, cite_uids, citation_contents)

    @staticmethod
    def _parse_finalization(tool_call: ToolCall) -> _Finalization:
        arguments, error = _parse_arguments(tool_call.function.arguments)
        if error:
            raise RetrievalError(
                "Invalid finalize_retrieval arguments",
                code="retrieval_finalize_invalid",
            )
        try:
            return _Finalization.model_validate(arguments)
        except ValidationError as error:
            raise RetrievalError(
                "Invalid finalize_retrieval arguments",
                code="retrieval_finalize_invalid",
            ) from error

    def _resolve(
        self,
        finalization: _Finalization,
        candidates: Mapping[str, _Candidate],
    ) -> RetrievalResult:
        scores: dict[str, float] = {}
        for item in finalization.items:
            if item.cite_uid in candidates:
                scores[item.cite_uid] = max(scores.get(item.cite_uid, -1.0), item.relevance_score)
        items = self._bounded_items(scores.items(), candidates)
        return RetrievalResult(
            status=finalization.status if items else "no_evidence",
            items=items,
            note=_truncate(finalization.note, self._settings.max_evidence_chars),
        )

    def _partial(
        self,
        candidates: Mapping[str, _Candidate],
        *,
        note: str,
    ) -> RetrievalResult:
        items = self._bounded_items(
            ((cite_uid, 0.0) for cite_uid in candidates),
            candidates,
        )
        return RetrievalResult(
            status="partial" if items else "no_evidence",
            items=items,
            note=note,
        )

    def _bounded_items(
        self,
        selected: Any,
        candidates: Mapping[str, _Candidate],
    ) -> list[EvidenceItem]:
        items = []
        for cite_uid, score in selected:
            candidate = candidates[cite_uid]
            metadata = extract_evidence_metadata(candidate.content)
            items.append(
                EvidenceItem(
                    cite_uid=cite_uid,
                    source_tool=candidate.source_tool,
                    relevance_score=score,
                    authority_rank=authority_rank(candidate.source_tool),
                    content=candidate.content,
                    **metadata,
                )
            )
        return rank_deduplicate_and_bound(items, self._settings.max_evidence_chars)


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}, "invalid JSON arguments"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


def _validate_schema(schema: dict[str, Any]) -> None:
    try:
        validator = validators.validator_for(schema)
        validator.check_schema(schema)
        _validate_schema_references(schema)
        validator(schema)
    except Exception as error:
        raise RetrievalError("MCP tool schema is invalid", code="mcp_schema_invalid") from error


def _arguments_match_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    try:
        validator = validators.validator_for(schema)(schema)
        return not any(validator.iter_errors(arguments))
    except Exception as error:
        raise RetrievalError("MCP tool schema is invalid", code="mcp_schema_invalid") from error


def _validate_schema_references(schema: Mapping[str, Any]) -> None:
    """Reject unsupported external and unresolved local JSON Schema references."""

    def validate_reference(reference: Any) -> None:
        if not isinstance(reference, str) or not reference.startswith("#"):
            raise ValueError("external JSON Schema references are not supported")
        if reference == "#":
            return
        if not reference.startswith("#/"):
            raise ValueError("JSON Schema reference is not a supported JSON Pointer")

        target: Any = schema
        for raw_token in reference[2:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if isinstance(target, Mapping) and token in target:
                target = target[token]
                continue
            if isinstance(target, list) and token.isdecimal() and int(token) < len(target):
                target = target[int(token)]
                continue
            raise ValueError("JSON Schema reference does not resolve")

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                if keyword in value:
                    validate_reference(value[keyword])
            for nested_value in value.values():
                visit(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                visit(nested_value)

    visit(schema)


def _call_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"arguments": arguments, "name": name},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _tool_error(tool_call: ToolCall, message: str) -> ChatMessage:
    return ChatMessage(
        role="tool",
        tool_call_id=tool_call.id,
        content=json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":")),
    )


def _truncate(value: str, maximum: int) -> str:
    suffix = "...[truncated]"
    if len(value) <= maximum:
        return value
    if maximum <= len(suffix):
        return suffix[:maximum]
    return f"{value[: maximum - len(suffix)]}{suffix}"


def _log_completion(result: RetrievalResult, calls_used: int, planner_turns: int) -> None:
    logger.info(
        "retrieval_complete status=%s evidence_items=%d tool_calls=%d planner_turns=%d",
        result.status,
        len(result.items),
        calls_used,
        planner_turns,
    )
