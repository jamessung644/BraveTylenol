import asyncio
import copy
import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lunit_hackathon.artifacts import (
    FINALIZE_RETRIEVAL_TOOL,
    MCP_BINDING_SEEDS,
    MCP_TOOL_ALIASES,
    RETRIEVAL_SYSTEM_PROMPT,
    RUNTIME_ARTIFACTS,
)
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import (
    RetrievalError,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamTransportError,
)
from lunit_hackathon.mcp_client import (
    MCPClientProtocol,
    bound_mcp_content,
    collect_cite_uids,
)
from lunit_hackathon.schemas import (
    TOOL_CALL_FINISH_REASONS,
    ChatMessage,
    EvidenceItem,
    L2Completion,
    RetrievalResult,
    ToolCall,
)
from lunit_hackathon.tool_bindings import (
    ToolBinding,
    ToolBindingError,
    compile_tool_bindings,
)

logger = logging.getLogger(__name__)

_SOURCE_ROUTING_BLOCK = re.compile(
    r"(<source_routing>\n)(.*?)(\n</source_routing>)",
    re.DOTALL,
)
_PROMPT_CALL_BUDGET = re.compile(r"(검색·열람 MCP 호출은 최대 )\d+(회다\.)")
_INDEX_TOOLS = frozenset(
    {
        "index_get_document_structure",
        "index_get_page_content",
        "index_get_relevant_nodes",
        "index_keyword_search",
        "index_list_documents",
    }
)
_RESEARCH_TOOLS = frozenset(
    {
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
        "rag_vector_query",
    }
)
_FAERS_TOOLS = frozenset(
    {
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
        "rag_sql_query",
    }
)
_KCD_TOOLS = frozenset({"kcd_get_name", "kcd_search_codes"})
_LAW_TOOLS = frozenset(
    {
        "openapi_law_get_article",
        "openapi_law_list_articles",
        "openapi_law_search",
    }
)
_MFDS_TOOLS = frozenset(
    {
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_mfds_get_drug_indication",
    }
)
_HIRA_TOOLS = frozenset(
    {
        "hira_updates_search",
        "openapi_hira_disease_check_code",
        "openapi_hira_get_drug_price",
    }
)
_ADR_TOOLS = frozenset({"adr_retrieve_drug_info"})
_DISCOVERY_ONLY_TOOLS = frozenset(
    {
        "index_get_document_structure",
        "index_get_relevant_nodes",
        "index_keyword_search",
        "index_list_documents",
        "kcd_search_codes",
        "openapi_law_list_articles",
        "openapi_law_search",
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
    }
)


@dataclass(frozen=True)
class _RetrievalIntent:
    """Small, query-derived policy inputs shared by routing and budgeting."""

    official_drug_label: bool = False
    mfds_ingredient_chain: bool = False
    mfds_followup_alias: str | None = None


def is_official_drug_label_request(query: str) -> bool:
    """Return whether ``query`` asks for primary regulatory drug labeling.

    Keep this source-intent predicate shared with the generation admission gate:
    otherwise an English label request can be forced into retrieval after the MCP
    registry has classified it as an unrelated, source-unspecified question.
    """

    normalized = unicodedata.normalize("NFKC", query).casefold()
    compact = re.sub(r"\s+", "", normalized)
    korean_label = any(
        marker in compact
        for marker in (
            "공식라벨",
            "공식의약품라벨",
            "의약품공식라벨",
            "처방정보",
        )
    )
    english_label = re.search(
        r"\b(?:official\s+(?:(?:drug|medication|medicine|prescription)\s+)?labels?"
        r"|prescribing\s+information|package\s+inserts?|drug\s+label(?:ing)?"
        r"|fda[-\s]?approved\s+(?:indications?|labels?|uses?))\b",
        normalized,
    )
    return korean_label or english_label is not None


def _artifact_max_mcp_calls() -> int:
    try:
        value = RUNTIME_ARTIFACTS.bundle["prompts"]["retrieval_dashboard_v1"]["max_retrieval_calls"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Retrieval artifact call budget is missing") from error
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 12:
        raise RuntimeError("Retrieval artifact call budget is invalid")
    return value


_ARTIFACT_MAX_MCP_CALLS = _artifact_max_mcp_calls()


class _FinalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cite_uid: str = Field(min_length=1, max_length=256)
    relevance_score: float = Field(ge=0.0, le=1.0)


class _Finalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "partial", "no_evidence"]
    items: list[_FinalItem] = Field(max_length=8)
    note: str = Field(max_length=512)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        cite_uids = [item.cite_uid for item in self.items]
        if len(cite_uids) != len(set(cite_uids)):
            raise ValueError("finalizer cite_uid values must be unique")
        if self.status == "sufficient" and not self.items:
            raise ValueError("sufficient requires at least one evidence item")
        if self.status == "no_evidence" and self.items:
            raise ValueError("no_evidence requires an empty evidence list")
        return self


@dataclass(frozen=True)
class _ObservedEvidence:
    source_tool: str
    content: str
    content_sha256: str


class ToolBindingRegistry:
    """Cache independently verified live bindings for the process lifetime."""

    def __init__(self) -> None:
        self._bindings: tuple[ToolBinding, ...] | None = None
        self._lock = asyncio.Lock()

    async def resolve(self, connection: Any) -> tuple[ToolBinding, ...]:
        if self._bindings is not None:
            return self._bindings
        async with self._lock:
            if self._bindings is not None:
                return self._bindings
            discovered = await connection.list_tools()
            seed_by_alias = {
                str(seed["logical_alias"]): seed for seed in MCP_BINDING_SEEDS
            }
            verified: list[ToolBinding] = []
            for alias in MCP_TOOL_ALIASES:
                try:
                    compiled = compile_tool_bindings(
                        discovered,
                        approved_aliases=(alias,),
                        binding_seeds=(seed_by_alias[alias],),
                        require_exact=True,
                    )
                except ToolBindingError as error:
                    # One unavailable or non-lossless schema must quarantine only
                    # that approved alias. A query whose route has another verified
                    # tool can still retrieve evidence; _select_bindings fails closed
                    # when none of its intended aliases survived verification.
                    logger.warning(
                        "mcp_binding_quarantined alias=%s reason_code=%s",
                        alias,
                        error.code,
                    )
                    continue
                if len(compiled) != 1 or compiled[0].model_function_name != alias:
                    logger.warning(
                        "mcp_binding_quarantined alias=%s reason_code=registry_mismatch",
                        alias,
                    )
                    continue
                verified.append(compiled[0])

            transport_owners: dict[str, list[str]] = {}
            for binding in verified:
                transport_owners.setdefault(binding.transport_tool_name, []).append(
                    binding.model_function_name
                )
            ambiguous_aliases = {
                alias
                for aliases in transport_owners.values()
                if len(aliases) > 1
                for alias in aliases
            }
            for alias in sorted(ambiguous_aliases):
                logger.warning(
                    "mcp_binding_quarantined alias=%s reason_code=duplicate_transport",
                    alias,
                )
            bindings = tuple(
                binding
                for binding in verified
                if binding.model_function_name not in ambiguous_aliases
            )
            self._bindings = bindings
            return bindings


class RetrievalEngine:
    """Lets L2 plan bounded MCP calls and returns normalized evidence."""

    def __init__(
        self,
        l2_client: Any,
        mcp_client: MCPClientProtocol,
        settings: Settings,
        binding_registry: ToolBindingRegistry | None = None,
    ) -> None:
        self._l2 = l2_client
        self._mcp = mcp_client
        self._settings = settings
        self._binding_registry = binding_registry or ToolBindingRegistry()

    async def retrieve(self, query: str) -> RetrievalResult:
        # Retrieval is optional. It may consume at most one third of the request
        # budget so the first routing turn and the final L2-authored answer each
        # retain their own time slice. A slow MCP dependency therefore degrades to
        # no-evidence instead of causing a user-visible end-to-end timeout.
        retrieval_timeout = min(
            50.0,
            self._settings.request_timeout_seconds * 0.31,
        )
        try:
            async with asyncio.timeout(retrieval_timeout):
                return await self._retrieve(query)
        except TimeoutError as error:
            raise RetrievalError(
                "Retrieval exceeded its time budget",
                code="retrieval_timeout",
            ) from error
        except UpstreamTimeoutError as error:
            raise RetrievalError(
                "Retrieval model step exceeded its time budget",
                code="retrieval_model_timeout",
            ) from error
        except (UpstreamTransportError, UpstreamResponseError) as error:
            raise RetrievalError(
                "Retrieval model step was unavailable",
                code="retrieval_model_unavailable",
            ) from error

    async def _retrieve(self, query: str) -> RetrievalResult:
        # All citation state stays inside this retrieval request.  Nothing in
        # this ledger is shared across users or subsequent calls.
        candidates: dict[str, _ObservedEvidence] = {}
        calls_used = 0
        planner_turns = 0
        seen_calls: set[str] = set()
        seen_discovery_aliases: set[str] = set()
        seen_call_ids: set[str] = set()
        source_unavailable = False
        schema_error_observed = False
        required_next_alias: str | None = None

        async with self._mcp.connect() as connection:
            discovered_bindings = await self._binding_registry.resolve(connection)
            intent = _classify_retrieval_intent(query)
            selected_bindings = _select_bindings(
                query,
                discovered_bindings,
                intent=intent,
            )
            selected_aliases = {
                binding.model_function_name for binding in selected_bindings
            }
            research_only = bool(selected_aliases) and selected_aliases <= _RESEARCH_TOOLS
            if research_only and selected_aliases != _RESEARCH_TOOLS:
                raise RetrievalError(
                    "The verified research discovery chain is incomplete",
                    code="retrieval_route_unavailable",
                )
            call_budget = _effective_call_budget(
                self._settings.max_mcp_calls,
                selected_bindings,
                intent=intent,
            )
            # A zero-call runtime exposes no remote function.  Otherwise the
            # same query-specific registry remains attached to every planner
            # turn, including the forced-finalization turn.  This keeps the
            # prompt registry exact-equal to Model tools[] without permitting
            # an extra MCP call after the budget is exhausted.
            active_bindings = selected_bindings if call_budget else ()
            tools = [binding.as_chat_tool() for binding in active_bindings] + [
                copy.deepcopy(FINALIZE_RETRIEVAL_TOOL)
            ]
            binding_map = {binding.model_function_name: binding for binding in active_bindings}
            if research_only and call_budget:
                # A research citation is valid only after choosing a source and
                # inspecting its detail; do not permit a planner to bypass those
                # graph prerequisites by jumping straight to vector search.
                required_next_alias = "rag_get_all_data_sources"
            messages = [
                ChatMessage(
                    role="system",
                    content=_active_retrieval_prompt(active_bindings, call_budget),
                ),
                ChatMessage(role="user", content=query),
            ]
            logger.info(
                "retrieval_discovery available_tools=%d selected_tools=%d call_budget=%d",
                len(discovered_bindings),
                len(active_bindings),
                call_budget,
            )
            # One planner turn per permitted call plus one finalization turn.
            # Invalid calls consume that turn instead of extending latency.
            max_planner_turns = max(1, call_budget + 1)

            while planner_turns < max_planner_turns:
                budget_exhausted = calls_used >= call_budget
                tool_choice: str | dict[str, Any] = "auto"
                if budget_exhausted:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": "finalize_retrieval"},
                    }
                elif required_next_alias is not None:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": required_next_alias},
                    }
                completion: L2Completion = await self._l2.complete(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=1_536,
                    attempt_timeout_seconds=25.0,
                )
                planner_turns += 1
                if (
                    completion.tool_calls
                    and completion.finish_reason not in TOOL_CALL_FINISH_REASONS
                ):
                    raise RetrievalError(
                        "Retrieval planner returned an invalid tool-call finish reason",
                        code="retrieval_planner_finish_reason_invalid",
                    )
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=completion.content,
                        tool_calls=completion.tool_calls,
                    )
                )
                if completion.tool_calls and completion.content and completion.content.strip():
                    raise RetrievalError(
                        "Retrieval planner mixed text with a tool call",
                        code="retrieval_planner_mixed_content",
                    )
                if not completion.tool_calls:
                    raise RetrievalError(
                        "Retrieval planner returned no tool calls",
                        code="retrieval_planner_no_tool_calls",
                    )
                if len(completion.tool_calls) != 1:
                    raise RetrievalError(
                        "Retrieval planner must emit exactly one tool call per turn",
                        code="retrieval_planner_tool_call_count_invalid",
                    )
                if (
                    required_next_alias is not None
                    and completion.tool_calls[0].function.name != required_next_alias
                ):
                    raise RetrievalError(
                        "Retrieval planner did not advance to the required content step",
                        code="retrieval_chain_incomplete",
                    )
                required_next_alias = None

                call_ids = [call.id for call in completion.tool_calls]
                if len(call_ids) != len(set(call_ids)) or any(
                    call_id in seen_call_ids for call_id in call_ids
                ):
                    raise RetrievalError(
                        "Retrieval planner reused a tool-call ID",
                        code="retrieval_tool_call_id_invalid",
                    )
                seen_call_ids.update(call_ids)

                finalizer_calls = [
                    call
                    for call in completion.tool_calls
                    if call.function.name == "finalize_retrieval"
                ]
                if len(finalizer_calls) > 1 or (
                    finalizer_calls and len(completion.tool_calls) != 1
                ):
                    raise RetrievalError(
                        "finalize_retrieval must be the only call in its planner turn",
                        code="retrieval_finalize_invalid",
                    )
                if finalizer_calls:
                    finalization = self._parse_finalization(finalizer_calls[0])
                    resolved = self._resolve(
                        finalization,
                        candidates,
                        execution_status=(
                            "schema_error"
                            if schema_error_observed
                            else "source_unavailable"
                            if source_unavailable
                            else "ok"
                        ),
                    )
                    _log_completion(resolved, calls_used, planner_turns)
                    return resolved

                valid: list[tuple[ToolCall, ToolBinding, dict[str, Any]]] = []
                errors: list[tuple[ToolCall, str]] = []
                remaining = call_budget - calls_used

                for call in completion.tool_calls:
                    arguments, error = _parse_arguments(call.function.arguments)
                    binding = binding_map.get(call.function.name)
                    if error:
                        errors.append((call, error))
                    elif binding is None:
                        errors.append((call, "unknown tool"))
                    else:
                        try:
                            arguments = binding.project_arguments(arguments)
                            arguments = _apply_query_constraints(
                                query,
                                binding.model_function_name,
                                arguments,
                            )
                            binding.validate_transport_arguments(arguments)
                        except ToolBindingError:
                            errors.append((call, "arguments do not match tool schema"))
                            continue
                    if error or binding is None:
                        continue
                    discovery_step = _is_discovery_step(
                        binding.model_function_name,
                        intent=intent,
                    )
                    signature = _call_signature(binding.transport_tool_name, arguments)
                    if (
                        discovery_step
                        and binding.model_function_name in seen_discovery_aliases
                    ):
                        errors.append(
                            (
                                call,
                                "discovery step already used; advance to a citable "
                                "content step or finalize retrieval",
                            )
                        )
                    elif signature in seen_calls:
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
                        seen_calls.add(signature)
                        if discovery_step:
                            seen_discovery_aliases.add(binding.model_function_name)
                        valid.append((call, binding, arguments))

                calls_used += len(valid)
                schema_error_observed = schema_error_observed or bool(errors)
                for call, error in errors:
                    messages.append(_tool_error(call, error))

                for call, binding, arguments in valid:
                    result, cite_uids, citation_contents, call_failed = await self._call(
                        connection,
                        call,
                        binding,
                        arguments,
                        discovery_only=_is_discovery_step(
                            binding.model_function_name,
                            intent=intent,
                        ),
                    )
                    source_unavailable = source_unavailable or call_failed
                    logger.info(
                        "retrieval_tool_result name=%s citation_count=%d",
                        call.function.name,
                        len(cite_uids),
                    )
                    messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=result))
                    self._record_observations(
                        candidates,
                        binding=binding,
                        result=result,
                        cite_uids=cite_uids,
                        citation_contents=citation_contents,
                    )
                    if not call_failed and calls_used < call_budget:
                        next_alias = _required_next_hop(
                            binding.model_function_name,
                            available_aliases=set(binding_map),
                            intent=intent,
                        )
                        if next_alias is not None and (
                            not candidates or intent.mfds_ingredient_chain
                        ):
                            required_next_alias = next_alias

        raise RetrievalError(
            "Retrieval planner exhausted its turns without finalize_retrieval",
            code="retrieval_finalize_missing",
        )

    async def _call(
        self,
        connection: Any,
        call: ToolCall,
        binding: ToolBinding,
        arguments: dict[str, Any],
        *,
        discovery_only: bool = False,
    ) -> tuple[str, list[str], dict[str, str], bool]:
        try:
            result = await connection.call_tool(binding.transport_tool_name, arguments)
            if result.is_error:
                return _error_content("MCP tool returned an error"), [], {}, True
            cite_uids = (
                []
                if discovery_only
                else result.cite_uids or collect_cite_uids(result.content)
            )
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
            return content, cite_uids, citation_contents, False
        except RetrievalError:
            return _error_content("MCP tool call failed"), [], {}, True

    @staticmethod
    def _record_observations(
        candidates: dict[str, _ObservedEvidence],
        *,
        binding: ToolBinding,
        result: str,
        cite_uids: list[str],
        citation_contents: dict[str, str],
    ) -> None:
        for cite_uid in cite_uids:
            content = citation_contents.get(cite_uid, result)
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            observed = _ObservedEvidence(
                source_tool=binding.transport_tool_name,
                content=content,
                content_sha256=content_sha256,
            )
            previous = candidates.get(cite_uid)
            if previous is not None and (
                previous.content_sha256 != content_sha256
                or previous.source_tool != binding.transport_tool_name
            ):
                raise RetrievalError(
                    "One cite_uid resolved to conflicting MCP content",
                    code="retrieval_citation_conflict",
                )
            candidates.setdefault(cite_uid, observed)

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
        candidates: Mapping[str, _ObservedEvidence],
        *,
        execution_status: Literal[
            "ok", "source_unavailable", "timeout", "schema_error", "budget_exhausted"
        ] = "ok",
    ) -> RetrievalResult:
        unobserved = [
            item.cite_uid for item in finalization.items if item.cite_uid not in candidates
        ]
        if unobserved:
            raise RetrievalError(
                "finalize_retrieval selected an unobserved cite_uid",
                code="retrieval_finalize_unobserved_uid",
            )
        selected = sorted(
            ((item.cite_uid, item.relevance_score) for item in finalization.items),
            key=lambda item: item[1],
            reverse=True,
        )
        items = self._bounded_items(selected, candidates)
        return RetrievalResult(
            status=finalization.status,
            items=items,
            note=finalization.note,
            execution_status=execution_status,
            semantic_reason=(
                "no_match"
                if finalization.status == "no_evidence"
                else "coverage_gap"
                if finalization.status == "partial"
                else "completed"
            ),
        )

    def _bounded_items(
        self,
        selected: list[tuple[str, float]],
        candidates: Mapping[str, _ObservedEvidence],
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        remaining = self._settings.max_evidence_chars
        for cite_uid, score in selected:
            if remaining <= 0 or len(items) >= 8:
                break
            observation = candidates[cite_uid]
            capped = bound_mcp_content(observation.content, remaining, [cite_uid])
            if len(capped) > remaining:
                break
            items.append(
                EvidenceItem(
                    cite_uid=cite_uid,
                    source_tool=observation.source_tool,
                    relevance_score=score,
                    content=capped,
                )
            )
            remaining -= len(capped)
        return items


def _classify_retrieval_intent(query: str) -> _RetrievalIntent:
    """Classify source intent without depending on product or benchmark names."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    compact = re.sub(r"\s+", "", normalized)
    official_drug_label = is_official_drug_label_request(query)

    ingredient_focus = any(
        marker in compact
        for marker in ("성분", "성분명", "주성분", "유효성분", "원료성분")
    ) or bool(re.search(r"\b(?:active\s+)?ingredient(?:s)?\b", normalized))
    mfds_authority = any(
        marker in compact
        for marker in (
            "식약처",
            "식약쳐",
            "식약청",
            "mfds",
            "품목허가",
            "의약품허가",
            "허가사항",
            "허가정보",
            "적응증",
            "용법",
            "금기",
        )
    )
    mfds_ingredient_chain = ingredient_focus and mfds_authority
    mfds_followup_alias: str | None = None
    if mfds_ingredient_chain:
        label_detail = any(
            marker in compact
            for marker in (
                "적응증",
                "용법",
                "용량",
                "금기",
                "주의사항",
                "허가사항",
                "라벨",
            )
        ) or bool(
            re.search(
                r"\b(?:indication|dos(?:e|age|ing)|contraindication|warning|label)\b",
                normalized,
            )
        )
        mfds_followup_alias = (
            "openapi_mfds_get_drug_indication"
            if label_detail
            else "openapi_mfds_check_drug_permission"
        )

    return _RetrievalIntent(
        official_drug_label=official_drug_label,
        mfds_ingredient_chain=mfds_ingredient_chain,
        mfds_followup_alias=mfds_followup_alias,
    )


def _select_bindings(
    query: str,
    bindings: tuple[ToolBinding, ...],
    *,
    intent: _RetrievalIntent | None = None,
) -> tuple[ToolBinding, ...]:
    """Expose the smallest useful strict registry for this source domain."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    compact = re.sub(r"\s+", "", normalized)
    intent = intent or _classify_retrieval_intent(query)
    selected: set[str] = set()

    # The generation gate recognizes official label / prescribing-information
    # questions as authoritative drug lookups.  Keep the retrieval route aligned:
    # ADR is the direct one-hop label source, while an explicit Korean MFDS intent
    # below may additionally select the domestic regulatory route.
    if intent.official_drug_label:
        selected.update(_ADR_TOOLS)

    if any(marker in compact for marker in ("kcd", "질병분류코드", "상병코드")) or re.search(
        r"(?<![a-z0-9])[a-z]\d{2}(?:\.\d+)?(?![a-z0-9])",
        normalized,
    ):
        selected.update({"kcd_get_name", "kcd_search_codes"})
    explicit_mfds = any(
        marker in compact
        for marker in (
            "식약처",
            "식약쳐",
            "식약청",
            "mfds",
            "품목허가",
            "의약품허가",
            "허가사항",
            "허가정보",
        )
    )
    mfds_property = any(
        marker in compact for marker in ("허가", "적응증", "용법", "금기", "성분")
    )
    if explicit_mfds or (mfds_property and not intent.official_drug_label):
        selected.update(_MFDS_TOOLS)
    if any(marker in compact for marker in ("심평원", "심평언", "hira", "급여", "약가", "수가")):
        selected.update(_HIRA_TOOLS)
    infection_law = any(
        marker in compact
        for marker in (
            "감염병예방법",
            "감염병법",
            "감염병의예방및관리에관한법률",
        )
    ) or (
        "감염병" in compact and any(marker in compact for marker in ("법", "법령", "조문", "시행"))
    )
    if infection_law or any(
        marker in compact for marker in ("법령", "법률", "조문", "의료법", "약사법")
    ):
        selected.update(_LAW_TOOLS)
    if any(
        marker in compact
        for marker in (
            "부작용",
            "이상반응",
            "상호작용",
            "경고",
            "adr",
            "dailymed",
            "druglabel",
        )
    ):
        selected.update(_ADR_TOOLS)
    if any(
        marker in compact
        for marker in (
            "가이드라인",
            "guideline",
            "진료지침",
            "임상지침",
            "진단기준",
            "치료목표",
            "권고",
        )
    ):
        selected.update(_INDEX_TOOLS)
    if any(marker in compact for marker in ("faers", "이상사례신호", "자발신고")):
        selected.update(_FAERS_TOOLS)
    explicit_research_corpus = any(
        marker in compact
        for marker in (
            "pubmed",
            "논문",
            "메타분석",
            "체계적문헌고찰",
            "rct",
        )
    )
    source_requested = any(marker in compact for marker in ("출처", "source", "evidence"))
    if explicit_research_corpus or (source_requested and not selected):
        selected.update(_RESEARCH_TOOLS)

    if not selected:
        # A source-unspecified medical question may require an official label,
        # a guideline page, or research evidence.  Expose each complete
        # discovery-to-evidence chain so the planner never sees a search tool
        # without the corresponding structure/page or source-discovery tool.
        selected.update(_ADR_TOOLS | _INDEX_TOOLS | _RESEARCH_TOOLS)

    filtered = tuple(binding for binding in bindings if binding.model_function_name in selected)
    if not filtered:
        raise RetrievalError(
            "No approved MCP tool matches the retrieval domain",
            code="retrieval_route_unavailable",
        )
    return filtered


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError, ValueError):
        return {}, "invalid JSON arguments"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


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


def _effective_call_budget(
    configured: int,
    selected: tuple[ToolBinding, ...],
    *,
    intent: _RetrievalIntent | None = None,
) -> int:
    if not selected:
        return 0
    names = {binding.model_function_name for binding in selected}
    if names & (_LAW_TOOLS | _INDEX_TOOLS | frozenset({"rag_sql_query"})):
        domain_ceiling = 3
    elif intent is not None and intent.mfds_ingredient_chain and names & _MFDS_TOOLS:
        # A generic ingredient is not yet a concrete domestic product.  Permit
        # one bounded product discovery call and one product/label verification.
        domain_ceiling = 2
    elif names & _RESEARCH_TOOLS:
        # Research retrieval is a discovery -> source detail -> vector evidence
        # graph.  The first two results are intentionally non-citable, so a full
        # configured budget of three is needed before evidence can be selected.
        domain_ceiling = 3
    elif names & _KCD_TOOLS:
        domain_ceiling = 2
    else:
        # Direct structured MFDS/HIRA/ADR calls should produce a citable result
        # in one hop.  Extra calls add latency without completing a document
        # discovery chain.
        domain_ceiling = 1
    return min(configured, _ARTIFACT_MAX_MCP_CALLS, domain_ceiling)


def _is_discovery_step(alias: str, *, intent: _RetrievalIntent) -> bool:
    return alias in _DISCOVERY_ONLY_TOOLS or (
        intent.mfds_ingredient_chain
        and alias == "openapi_mfds_find_drugs_by_ingredient"
    )


def _required_next_hop(
    completed_alias: str,
    *,
    available_aliases: set[str],
    intent: _RetrievalIntent,
) -> str | None:
    """Return a safe graph successor that can yield citable content."""

    next_alias: str | None = None
    if completed_alias == "openapi_law_search":
        next_alias = "openapi_law_list_articles"
    elif completed_alias == "openapi_law_list_articles":
        next_alias = "openapi_law_get_article"
    elif completed_alias in _INDEX_TOOLS - {"index_get_page_content"}:
        next_alias = "index_get_page_content"
    elif completed_alias == "kcd_search_codes":
        next_alias = "kcd_get_name"
    elif completed_alias == "rag_get_all_data_sources":
        next_alias = "rag_get_data_source_detail"
    elif completed_alias == "rag_get_data_source_detail":
        if "rag_sql_query" in available_aliases:
            next_alias = "rag_sql_query"
        elif "rag_vector_query" in available_aliases:
            next_alias = "rag_vector_query"
    elif (
        completed_alias == "openapi_mfds_find_drugs_by_ingredient"
        and intent.mfds_ingredient_chain
    ):
        next_alias = intent.mfds_followup_alias

    return next_alias if next_alias in available_aliases else None


def _active_retrieval_prompt(
    bindings: tuple[ToolBinding, ...],
    call_budget: int,
) -> str:
    """Render one deterministic prompt whose MCP aliases equal Model tools[]."""

    active = {binding.model_function_name for binding in bindings}
    match = _SOURCE_ROUTING_BLOCK.search(RETRIEVAL_SYSTEM_PROMPT)
    if match is None:
        raise RetrievalError(
            "Retrieval prompt is missing its source-routing block",
            code="retrieval_prompt_registry_mismatch",
        )

    if active:
        route_lines: list[str] = []
        described: set[str] = set()
        for line in match.group(2).splitlines():
            referenced = {alias for alias in MCP_TOOL_ALIASES if alias in line}
            if not referenced or referenced <= active:
                route_lines.append(line)
                described.update(referenced)
        for alias in sorted(active - described):
            route_lines.append(
                f"- {alias}: 등록된 strict schema에 맞는 원격 단계에만 사용한다. "
                "필요한 후속 열람 도구가 등록되지 않았다면 discovery·목록 결과를 "
                "최종 근거로 과장하지 말고 partial 또는 no_evidence로 종료한다."
            )
        routing = "\n".join(route_lines).strip()
    else:
        routing = (
            "- 이번 request에는 원격 의료 MCP 도구가 등록되지 않았다. "
            "근거를 발명하지 말고 no_evidence로 종료한다."
        )

    prompt = _SOURCE_ROUTING_BLOCK.sub(
        lambda route_match: f"{route_match.group(1)}{routing}{route_match.group(3)}",
        RETRIEVAL_SYSTEM_PROMPT,
        count=1,
    )
    prompt, replacements = _PROMPT_CALL_BUDGET.subn(
        lambda budget_match: f"{budget_match.group(1)}{call_budget}{budget_match.group(2)}",
        prompt,
        count=1,
    )
    observed = {alias for alias in MCP_TOOL_ALIASES if alias in prompt}
    if replacements != 1 or observed != active:
        raise RetrievalError(
            "Active Retrieval prompt differs from the registered MCP tool subset",
            code="retrieval_prompt_registry_mismatch",
        )
    return prompt


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
