import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal

from jsonschema import SchemaError, validators
from pydantic import BaseModel, Field, ValidationError

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import RetrievalError
from lunit_hackathon.mcp_client import (
    MCPClientProtocol,
    bound_mcp_content,
    collect_cite_uids,
)
from lunit_hackathon.prompts import (
    FINALIZE_RETRIEVAL_TOOL,
    RETRIEVAL_PLANNER_SYSTEM_PROMPT,
)
from lunit_hackathon.schemas import (
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    ToolCall,
)

logger = logging.getLogger(__name__)


class _FinalItem(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class _Finalization(BaseModel):
    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[_FinalItem]
    note: str


class RetrievalEngine:
    """Lets L2 plan bounded MCP calls and returns normalized evidence."""

    def __init__(
        self,
        l2_client: Any,
        mcp_client: MCPClientProtocol,
        settings: Settings,
    ) -> None:
        self._l2 = l2_client
        self._mcp = mcp_client
        self._settings = settings

    async def retrieve(self, query: str) -> RetrievalResult:
        candidates: dict[str, tuple[str, str]] = {}
        messages = [
            ChatMessage(role="system", content=RETRIEVAL_PLANNER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=query),
        ]
        calls_used = 0
        planner_turns = 0
        seen_calls: set[str] = set()

        async with self._mcp.connect() as connection:
            discovered = await connection.list_tools()
            if any(tool.name == "finalize_retrieval" for tool in discovered):
                raise RetrievalError(
                    "MCP tool name collision: finalize_retrieval",
                    code="mcp_tool_name_collision",
                )
            tool_map = {tool.name: tool for tool in discovered}
            for tool in discovered:
                _validate_schema(tool.input_schema)
            selected_tools = _select_tools(query, discovered)
            tools = [tool.as_chat_tool() for tool in selected_tools] + [FINALIZE_RETRIEVAL_TOOL]
            selected_names = {tool.name for tool in selected_tools}
            tool_map = {name: tool for name, tool in tool_map.items() if name in selected_names}
            call_budget = _effective_call_budget(
                self._settings.max_mcp_calls,
                selected_tools,
                discovered,
            )
            logger.info(
                "retrieval_discovery available_tools=%d selected_tools=%d call_budget=%d",
                len(discovered),
                len(selected_tools),
                call_budget,
            )
            max_planner_turns = max(2, call_budget + 2)

            while planner_turns < max_planner_turns:
                budget_exhausted = calls_used >= call_budget
                planner_tools = [FINALIZE_RETRIEVAL_TOOL] if budget_exhausted else tools
                tool_choice: str | dict[str, Any] = "auto"
                if budget_exhausted:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": "finalize_retrieval"},
                    }
                completion: L2Completion = await self._l2.complete(
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
                if not completion.tool_calls:
                    raise RetrievalError(
                        "Retrieval planner returned no tool calls",
                        code="retrieval_planner_no_tool_calls",
                    )

                finalization: _Finalization | None = None
                valid: list[tuple[ToolCall, dict[str, Any]]] = []
                errors: list[tuple[ToolCall, str]] = []
                remaining = call_budget - calls_used

                for call in completion.tool_calls:
                    if call.function.name == "finalize_retrieval":
                        finalization = self._parse_finalization(call)
                        continue
                    arguments, error = _parse_arguments(call.function.arguments)
                    if error is None and call.function.name in tool_map:
                        arguments = _apply_query_constraints(
                            query,
                            call.function.name,
                            arguments,
                        )
                    if error:
                        errors.append((call, error))
                    elif call.function.name not in tool_map:
                        errors.append((call, "unknown tool"))
                    elif not _arguments_match_schema(
                        arguments,
                        tool_map[call.function.name].input_schema,
                    ):
                        errors.append((call, "arguments do not match tool schema"))
                    elif _call_signature(call.function.name, arguments) in seen_calls:
                        errors.append(
                            (
                                call,
                                "duplicate tool call; use the previous result or "
                                "finalize retrieval",
                            )
                        )
                    elif len(valid) >= remaining:
                        errors.append((call, "tool-call budget exhausted"))
                    else:
                        seen_calls.add(_call_signature(call.function.name, arguments))
                        valid.append((call, arguments))

                calls_used += len(valid)
                for call, error in errors:
                    messages.append(_tool_error(call, error))

                results = await asyncio.gather(
                    *(self._call(connection, call, arguments) for call, arguments in valid)
                )
                for (call, _), (result, cite_uids, citation_contents) in zip(
                    valid,
                    results,
                    strict=True,
                ):
                    logger.info(
                        "retrieval_tool_result name=%s citation_count=%d",
                        call.function.name,
                        len(cite_uids),
                    )
                    messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=result))
                    for cite_uid in cite_uids:
                        candidates.setdefault(
                            cite_uid,
                            (call.function.name, citation_contents.get(cite_uid, result)),
                        )

                if finalization is not None:
                    resolved = self._resolve(finalization, candidates)
                    _log_completion(resolved, calls_used, planner_turns)
                    return resolved

        partial = self._partial(
            candidates,
            note="Retrieval planner turn limit exhausted before finalization.",
        )
        _log_completion(partial, calls_used, planner_turns)
        return partial

    async def _call(
        self,
        connection: Any,
        call: ToolCall,
        arguments: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, str]]:
        try:
            result = await connection.call_tool(call.function.name, arguments)
            if result.is_error:
                return _error_content("MCP tool returned an error"), [], {}
            cite_uids = result.cite_uids or collect_cite_uids(result.content)
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
            return content, cite_uids, citation_contents
        except RetrievalError:
            return _error_content("MCP tool call failed"), [], {}

    @staticmethod
    def _parse_finalization(call: ToolCall) -> _Finalization:
        arguments, error = _parse_arguments(call.function.arguments)
        if error:
            raise RetrievalError(
                f"Invalid finalize_retrieval arguments: {error}",
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
        candidates: Mapping[str, tuple[str, str]],
    ) -> RetrievalResult:
        scores: dict[str, float] = {}
        for item in finalization.items:
            if item.cite_uid in candidates:
                scores[item.cite_uid] = max(
                    scores.get(item.cite_uid, -1.0),
                    item.relevance_score,
                )
        selected = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        items = self._bounded_items(selected, candidates)
        status = finalization.status if items else "no_evidence"
        return RetrievalResult(
            status=status,
            items=items,
            note=_truncate(finalization.note, self._settings.max_evidence_chars),
        )

    def _partial(
        self,
        candidates: Mapping[str, tuple[str, str]],
        *,
        note: str = "MCP tool-call budget exhausted before finalization.",
    ) -> RetrievalResult:
        items = self._bounded_items(
            [(cite_uid, 0.0) for cite_uid in candidates],
            candidates,
        )
        return RetrievalResult(
            status="partial" if items else "no_evidence",
            items=items,
            note=note,
        )

    def _bounded_items(
        self,
        selected: list[tuple[str, float]],
        candidates: Mapping[str, tuple[str, str]],
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        remaining = self._settings.max_evidence_chars
        for cite_uid, score in selected:
            if remaining <= 0:
                break
            source_tool, content = candidates[cite_uid]
            capped = bound_mcp_content(content, remaining, [cite_uid])
            if len(capped) > remaining:
                break
            items.append(
                EvidenceItem(
                    cite_uid=cite_uid,
                    source_tool=source_tool,
                    relevance_score=score,
                    content=capped,
                )
            )
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


def _validate_schema(schema: dict[str, Any]) -> None:
    try:
        validators.validator_for(schema).check_schema(schema)
    except SchemaError as error:
        raise RetrievalError(
            "MCP tool schema is invalid",
            code="mcp_schema_invalid",
        ) from error


def _arguments_match_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    validator = validators.validator_for(schema)(schema)
    return not any(validator.iter_errors(arguments))


def _call_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {"arguments": arguments, "name": name},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _apply_query_constraints(
    query: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    constrained = dict(arguments)
    if tool_name in {"kcd_get_name", "kcd_search_codes"}:
        normalized = query.casefold().replace(" ", "")
        if "kcd-8" in normalized or "kcd8" in normalized:
            constrained["revision"] = "KCD-8"
        elif "kcd-9" in normalized or "kcd9" in normalized:
            constrained["revision"] = "KCD-9"
    if tool_name in {
        "openapi_hira_get_drug_price",
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_mfds_get_drug_indication",
    }:
        num_rows = constrained.get("num_rows", 3)
        if not isinstance(num_rows, int) or isinstance(num_rows, bool):
            num_rows = 3
        constrained["num_rows"] = max(1, min(num_rows, 3))
    return constrained


def _select_tools(query: str, discovered: list[Any]) -> list[Any]:
    normalized = query.casefold()
    selected_names: set[str] = set()

    def includes_any(markers: tuple[str, ...]) -> bool:
        return any(marker in normalized for marker in markers)

    if includes_any(("kcd", "질병분류", "질병 코드", "상병 코드")):
        selected_names.update(
            {"kcd_get_name", "kcd_search_codes", "openapi_hira_disease_check_code"}
        )
    drug_question = includes_any(
        (
            "의약품",
            "약물",
            "약가",
            "성분",
            "효능",
            "부작용",
            "복용",
            "투여",
            "허가",
            "drug",
            "타이레놀",
        )
    )
    if drug_question:
        drug_tools: set[str] = set()
        mfds_requested = includes_any(("식약처", "mfds"))
        if mfds_requested or includes_any(
            (
                "효능",
                "효과",
                "적응증",
                "주의",
                "금기",
                "허가사항",
                "용법",
                "용량",
            )
        ):
            drug_tools.add("openapi_mfds_get_drug_indication")
        if includes_any(("허가 여부", "허가 상태", "승인", "취하")):
            drug_tools.add("openapi_mfds_check_drug_permission")
        if includes_any(("동일성분", "대체약", "대체 약")):
            drug_tools.add("openapi_mfds_find_drugs_by_ingredient")
        if includes_any(("약가", "급여", "상한금액")):
            drug_tools.add("openapi_hira_get_drug_price")
        if not mfds_requested and includes_any(("부작용", "상호작용", "경고")):
            drug_tools.add("adr_retrieve_drug_info")
        if not drug_tools:
            drug_tools.update({"adr_retrieve_drug_info", "openapi_mfds_get_drug_indication"})
        selected_names.update(drug_tools)
    if includes_any(("보험", "급여", "심평원", "hira", "수가", "요양")):
        selected_names.update(
            {
                "hira_updates_search",
                "openapi_hira_disease_check_code",
                "openapi_hira_get_drug_price",
            }
        )
    if includes_any(("법", "법령", "조문", "의료법")):
        selected_names.update(
            {"openapi_law_get_article", "openapi_law_list_articles", "openapi_law_search"}
        )
    if includes_any(("지침", "권고", "가이드라인", "guideline", "문서", "페이지")):
        selected_names.update(tool.name for tool in discovered if tool.name.startswith("index_"))
    if includes_any(("논문", "연구", "pubmed", "pmc")):
        selected_names.update(
            {"rag_get_all_data_sources", "rag_get_data_source_detail", "rag_vector_query"}
        )

    selected = [tool for tool in discovered if tool.name in selected_names]
    return selected or discovered


def _effective_call_budget(
    configured: int,
    selected: list[Any],
    discovered: list[Any],
) -> int:
    if len(selected) == len(discovered):
        return configured
    return min(configured, len(selected), 2)


def _log_completion(
    result: RetrievalResult,
    calls_used: int,
    planner_turns: int,
) -> None:
    logger.info(
        "retrieval_complete status=%s evidence_items=%d tool_calls=%d planner_turns=%d",
        result.status,
        len(result.items),
        calls_used,
        planner_turns,
    )


def _tool_error(call: ToolCall, message: str) -> ChatMessage:
    return ChatMessage(
        role="tool",
        tool_call_id=call.id,
        content=_error_content(message),
    )


def _error_content(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))


def _truncate(value: str, maximum: int) -> str:
    suffix = "...[truncated]"
    if len(value) <= maximum:
        return value
    if maximum <= len(suffix):
        return suffix[:maximum]
    return f"{value[: maximum - len(suffix)]}{suffix}"
