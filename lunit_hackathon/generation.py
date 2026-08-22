import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Literal
from urllib.parse import unquote

from lunit_hackathon.artifacts import MCP_TOOL_ALIASES
from lunit_hackathon.errors import (
    MalformedUpstreamResponseError,
    RetrievalError,
    UpstreamTimeoutError,
)
from lunit_hackathon.prompts import (
    RETRIEVE_RELEVANT_CONTENT_TOOL,
    clean_recovery_final_system_prompt,
    direct_final_system_prompt,
    emergency_generation_prompt,
    generation_system_prompt,
    mcp_failure_final_system_prompt,
    post_retrieval_final_system_prompt,
    safe_completion_final_system_prompt,
)
from lunit_hackathon.retrieval import is_official_drug_label_request
from lunit_hackathon.schemas import (
    TOOL_CALL_FINISH_REASONS,
    ChatMessage,
    L2Completion,
    RetrievalResult,
    ToolCall,
)

logger = logging.getLogger(__name__)

_RETRIEVAL_TOOL_NAME = "retrieve_relevant_content"
_MAX_QUERY_CHARS = 2_048
_INITIAL_TOOL_TIMEOUT_SECONDS = 25.0
_FORCED_TOOL_RETRY_TIMEOUT_SECONDS = 10.0
_FINAL_GENERATION_TIMEOUT_SECONDS = 145.0
_RECOVERY_TIMEOUT_SECONDS = 145.0
_EMERGENCY_TIMEOUT_SECONDS = 145.0
_SAFE_COMPLETION_TIMEOUT_SECONDS = 30.0
_FINAL_MAX_TOKENS = 2_048
_EMERGENCY_MAX_TOKENS = 2_048
_SAFE_COMPLETION_MAX_TOKENS = 256
# A clean retry must have enough room to replace any regular final that reached
# its output limit. Keeping this tied to the regular cap prevents the retry from
# being structurally more likely to truncate than the draft it replaces.
_RECOVERY_MAX_TOKENS = _FINAL_MAX_TOKENS
_FINAL_PHASES = ("direct", "post_retrieval", "mcp_failure", "emergency")
FinalPhase = Literal["direct", "post_retrieval", "mcp_failure", "emergency"]
SafeCompletionPhase = Literal["normal", "emergency"]
_PROTOCOL_FUNCTION_NAMES = frozenset(
    (_RETRIEVAL_TOOL_NAME, "finalize_retrieval", *MCP_TOOL_ALIASES)
)


class GenerationEngine:
    """Run isolated decision/final phases while leaving every answer to L2."""

    def __init__(self, l2_client: Any, retrieval_engine: Any | None) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        # Delaying an obvious emergency for evidence lookup is unsafe. L2 still authors
        # the answer, but the application tool is intentionally absent for this turn.
        if _is_emergency_turn(messages):
            return await self._emergency_answer(messages)

        return await self._normal_answer(messages)

    async def _normal_answer(
        self,
        messages: Sequence[ChatMessage],
    ) -> str:
        # This transcript is exclusively for the internal evidence decision. No
        # assistant content from it is ever returned as a user answer or copied into
        # a final-answer transcript.
        decision_conversation = _medical_conversation(
            messages,
            system_prompt=generation_system_prompt(),
        )
        tool_choice: str | dict[str, Any] = "auto"
        retrieval_required = requires_retrieval(messages)
        if retrieval_required:
            tool_choice = {
                "type": "function",
                "function": {"name": _RETRIEVAL_TOOL_NAME},
            }
        first: L2Completion = await self._l2.complete(
            messages=decision_conversation,
            tools=[RETRIEVE_RELEVANT_CONTENT_TOOL],
            tool_choice=tool_choice,
            max_tokens=1_536,
            attempt_timeout_seconds=_INITIAL_TOOL_TIMEOUT_SECONDS,
        )
        if retrieval_required and not first.tool_calls:
            decision_conversation.append(_assistant_message(first))
            decision_conversation.append(
                {
                    "role": "system",
                    "content": (
                        "The request requires current or primary evidence, but the required "
                        "retrieval function was not called. Do not answer from memory. Call "
                        "retrieve_relevant_content exactly once now with a self-contained query."
                    ),
                }
            )
            first = await self._l2.complete(
                messages=decision_conversation,
                tools=[RETRIEVE_RELEVANT_CONTENT_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": _RETRIEVAL_TOOL_NAME},
                },
                max_tokens=1_536,
                attempt_timeout_seconds=_FORCED_TOOL_RETRY_TIMEOUT_SECONDS,
            )
        if retrieval_required and not first.tool_calls:
            return await self._complete_final_answer(
                messages,
                phase="mcp_failure",
                failure_reason="required_evidence_unavailable",
            )
        if not first.tool_calls:
            return await self._complete_final_answer(messages, phase="direct")

        if first.content and first.content.strip():
            call, query, protocol_error = (
                None,
                "",
                "retrieval call cannot be mixed with user-visible text",
            )
        else:
            call, query, protocol_error = _one_valid_retrieval(
                first,
                messages,
            )
        if call is None:
            logger.warning("retrieval_decision_invalid reason=%s", protocol_error)
            return await self._complete_final_answer(
                messages,
                phase="mcp_failure",
                failure_reason="invalid_evidence_request",
            )

        if self._retrieval is None:
            retrieval_result = _retrieval_failure("retrieval_not_configured")
        else:
            try:
                retrieval_result = await self._retrieval.retrieve(query)
            except RetrievalError as error:
                logger.warning("retrieval_failed error_code=%s", error.code)
                retrieval_result = _retrieval_failure(error.code)

        if retrieval_result.execution_status != "ok":
            return await self._complete_final_answer(
                messages,
                phase="mcp_failure",
                retrieval=retrieval_result,
                failure_reason=retrieval_result.execution_status,
            )
        return await self._complete_final_answer(
            messages,
            phase="post_retrieval",
            retrieval=retrieval_result,
        )

    async def direct_answer(
        self,
        messages: Sequence[ChatMessage],
        *,
        emergency: bool | None = None,
    ) -> str:
        """Ask L2 for a final answer without claiming or attempting retrieval."""

        if emergency is True or (emergency is None and _is_emergency_turn(messages)):
            return await self._emergency_answer(messages)
        return await self._complete_final_answer(messages, phase="direct")

    async def evidence_unavailable_answer(
        self,
        messages: Sequence[ChatMessage],
        *,
        emergency: bool | None = None,
        failure_reason: str = "required_evidence_unavailable",
    ) -> str:
        """Generate a guarded final after an admitted evidence route is unavailable."""

        if emergency is True or (emergency is None and _is_emergency_turn(messages)):
            return await self._emergency_answer(messages)
        return await self._complete_final_answer(
            messages,
            phase="mcp_failure",
            failure_reason=failure_reason,
        )

    async def _emergency_answer(self, messages: Sequence[ChatMessage]) -> str:
        return await self._complete_final_answer(messages, phase="emergency")

    async def _complete_final_answer(
        self,
        messages: Sequence[ChatMessage],
        *,
        phase: FinalPhase,
        retrieval: RetrievalResult | None = None,
        failure_reason: str | None = None,
    ) -> str:
        """Generate and validate a final answer, with one clean bounded retry.

        Both attempts are rebuilt from the frozen inbound messages and trusted phase
        context. The evidence-decision transcript and an invalid draft are never
        copied into either final transcript.
        """

        if phase not in _FINAL_PHASES:
            raise ValueError(f"unsupported final phase: {phase}")
        final_context = _final_phase_context(
            phase,
            retrieval=retrieval,
            failure_reason=failure_reason,
        )
        initial_conversation = _medical_conversation(
            messages,
            system_prompt=_final_system_prompt(phase),
            final_phase_context=final_context,
        )
        timeout = (
            _EMERGENCY_TIMEOUT_SECONDS
            if phase == "emergency"
            else _FINAL_GENERATION_TIMEOUT_SECONDS
        )
        final_max_tokens = (
            _EMERGENCY_MAX_TOKENS if phase == "emergency" else _FINAL_MAX_TOKENS
        )
        try:
            completion = await self._l2.complete(
                messages=initial_conversation,
                attempt_timeout_seconds=timeout,
                max_tokens=final_max_tokens,
                allow_blank_recovery=False,
                allow_empty_completion=True,
            )
        except (MalformedUpstreamResponseError, UpstreamTimeoutError) as error:
            completion = None
            violations = [
                "initial_timeout"
                if isinstance(error, UpstreamTimeoutError)
                else "malformed_completion"
            ]
        else:
            violations = _final_violations(
                completion,
                retrieval=retrieval,
                phase=phase,
            )
            if not violations:
                return completion.content or ""  # nonempty is established above
        citation_omission_only = set(violations) == {"missing_allowed_citation"}

        logger.warning(
            "l2_final_invalid phase=%s validator_codes=%s",
            phase,
            ",".join(violations),
        )
        recovery_conversation = _medical_conversation(
            messages,
            system_prompt=clean_recovery_final_system_prompt(),
            final_phase_context=final_context,
        )
        recovery_error: MalformedUpstreamResponseError | UpstreamTimeoutError | None = None
        try:
            recovered = await self._l2.complete(
                messages=recovery_conversation,
                attempt_timeout_seconds=_RECOVERY_TIMEOUT_SECONDS,
                max_tokens=_RECOVERY_MAX_TOKENS,
                allow_blank_recovery=False,
                allow_empty_completion=False,
            )
        except (MalformedUpstreamResponseError, UpstreamTimeoutError) as error:
            recovery_error = error
            remaining = [
                "recovery_timeout"
                if isinstance(error, UpstreamTimeoutError)
                else "recovery_malformed"
            ]
        else:
            remaining = _final_violations(
                recovered,
                retrieval=retrieval,
                phase=phase,
            )
        if not remaining:
            return recovered.content or ""  # nonempty is established above

        if recovery_error is None:
            if citation_omission_only and set(remaining) == {"missing_allowed_citation"}:
                logger.warning(
                    "l2_final_citation_omitted_after_recovery phase=%s",
                    phase,
                )
                return recovered.content or ""
            logger.warning(
                "l2_final_recovery_invalid phase=%s validator_codes=%s",
                phase,
                ",".join(remaining),
            )
        else:
            logger.warning(
                "l2_final_recovery_failed phase=%s validator_codes=%s",
                phase,
                ",".join(remaining),
            )

        if not _safe_completion_permitted(self._l2):
            if recovery_error is not None:
                raise recovery_error
            raise MalformedUpstreamResponseError(
                "L2 final answer remained invalid after bounded recovery"
            )
        return await self._safe_completion(emergency=phase == "emergency")

    async def _safe_completion(self, *, emergency: bool) -> str:
        """Ask L2 once for a fixed-input safety notice; never author it in Python."""

        safe_phase: SafeCompletionPhase = "emergency" if emergency else "normal"
        completion = await self._l2.complete(
            messages=_safe_completion_conversation(safe_phase),
            attempt_timeout_seconds=_SAFE_COMPLETION_TIMEOUT_SECONDS,
            max_tokens=_SAFE_COMPLETION_MAX_TOKENS,
            allow_blank_recovery=False,
            allow_empty_completion=False,
        )
        violations = _safe_completion_violations(completion, phase=safe_phase)
        if violations:
            logger.warning(
                "l2_safe_completion_invalid phase=%s validator_codes=%s",
                safe_phase,
                ",".join(violations),
            )
            raise MalformedUpstreamResponseError("L2 safe completion was invalid")
        return completion.content or ""  # nonempty is established above


def _medical_conversation(
    messages: Sequence[ChatMessage],
    *,
    system_prompt: str | None = None,
    final_phase_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not messages or messages[-1].role != "user" or messages[-1].content is None:
        raise MalformedUpstreamResponseError("Generation requires a latest user message")
    conversation = [
        {
            "role": "system",
            "content": system_prompt or generation_system_prompt(),
        }
    ]
    conversation.extend(
        message.model_dump(exclude_none=True) for message in messages[:-1]
    )
    envelope = {
        "schema_version": "generation-input-v1",
        "trust_level": "untrusted_user_payload",
        "latest_user_message": {
            "turn_index": len(messages) - 1,
            "content": messages[-1].content,
        },
        "application_context": {
            "schema_version": "conversation-context-v4",
            "trust_level": "untrusted_data",
            "history_mode": "raw_history_only",
            "normalization_status": "degraded_raw_only",
            "state_integrity_status": "degraded_raw_only",
            "state_incomplete": True,
            "do_not_infer_absence": True,
            "critical_unknowns": [
                {
                    "reason_code": "derived_state_rebuild_failed",
                    "required_safety_handling": "avoid_reassurance",
                }
            ],
        },
    }
    if final_phase_context is not None:
        envelope["final_phase_context"] = dict(final_phase_context)
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # Decode-after-encode equality is the authority boundary for user text.
    if json.loads(encoded)["latest_user_message"]["content"] != messages[-1].content:
        raise MalformedUpstreamResponseError("Generation input encoding changed user text")
    conversation.append({"role": "user", "content": encoded})
    return conversation


def _safe_completion_conversation(
    phase: SafeCompletionPhase,
) -> list[dict[str, Any]]:
    """Build a fixed transcript containing no user medical content or prior draft."""

    indicator = json.dumps(
        {
            "schema_version": "generation-safe-completion-v1",
            "phase": phase,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        {"role": "system", "content": safe_completion_final_system_prompt()},
        {"role": "user", "content": indicator},
    ]


def _final_system_prompt(phase: FinalPhase) -> str:
    renderers = {
        "direct": direct_final_system_prompt,
        "post_retrieval": post_retrieval_final_system_prompt,
        "mcp_failure": mcp_failure_final_system_prompt,
        "emergency": emergency_generation_prompt,
    }
    return renderers[phase]()


def _final_phase_context(
    phase: FinalPhase,
    *,
    retrieval: RetrievalResult | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "schema_version": "generation-final-context-v1",
        "phase": phase,
        "evidence_status": "not_requested",
    }
    if phase == "post_retrieval":
        if retrieval is None or retrieval.execution_status != "ok":
            raise ValueError("post-retrieval final requires completed evidence retrieval")
        context["evidence_status"] = (
            "available" if _citable_evidence_items(retrieval) else "none"
        )
        context["evidence"] = json.loads(_evidence_json(retrieval))
    elif phase == "mcp_failure":
        context["evidence_status"] = "unavailable"
        context["reason_code"] = _safe_failure_reason(failure_reason)
    return context


def _safe_failure_reason(reason: str | None) -> str:
    if reason is None:
        return "evidence_unavailable"
    normalized = "".join(
        character for character in reason.casefold() if character.isalnum() or character == "_"
    )
    return normalized[:80] or "evidence_unavailable"


def _assistant_message(completion: L2Completion) -> dict[str, Any]:
    return ChatMessage(
        role="assistant",
        content=completion.content,
        tool_calls=completion.tool_calls or None,
    ).model_dump(exclude_none=True)


def _one_valid_retrieval(
    completion: L2Completion,
    messages: Sequence[ChatMessage],
) -> tuple[ToolCall | None, str, str]:
    if completion.finish_reason not in TOOL_CALL_FINISH_REASONS:
        return None, "", "retrieval call has an invalid finish reason"
    calls = completion.tool_calls
    if len(calls) != 1:
        return None, "", "exactly one retrieval call is allowed"
    call = calls[0]
    if call.function.name != _RETRIEVAL_TOOL_NAME:
        return None, "", "unexpected tool"
    if _valid_query(call) is None:
        return None, "", "invalid retrieval request"
    query, source_error = _authoritative_retrieval_query(messages)
    if query is None:
        return None, "", source_error
    return call, query, ""


def _valid_query(call: ToolCall) -> str | None:
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
        arguments = json.loads(
            call.function.arguments,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(arguments, Mapping) or set(arguments) != {"query"}:
        return None
    query = arguments["query"]
    if not isinstance(query, str):
        return None
    query = query.strip()
    if not query or len(query) > _MAX_QUERY_CHARS:
        return None
    if any(ord(character) < 32 and character not in "\t\n" for character in query):
        return None
    return query


_ANAPHORA_MARKERS = (
    "그약",
    "이약",
    "그검사",
    "이검사",
    "그수치",
    "이수치",
    "그결과",
    "이결과",
    "그증상",
    "이증상",
    "그거",
    "그것",
    "그건",
    "그게",
    "그걸",
    "이거",
    "이것",
    "이걸",
    "그내용",
    "그근거",
    "그출처",
    "해당약",
    "앞서말한약",
    "아까검사",
    "아까수치",
    "앞의약",
)
_CONTEXT_STOPWORDS = {
    "제가",
    "저는",
    "나는",
    "지금",
    "오늘",
    "어제",
    "전에",
    "전부터",
    "입니다",
    "이에요",
    "있어요",
    "있습니다",
    "먹고",
    "먹는",
    "복용중",
    "복용하고",
    "중이에요",
    "중입니다",
    "그리고",
    "그런데",
    "그러면",
    "그럼",
    "같이",
    "더",
    "관련",
    "대해",
    "대한",
    "먹어도",
    "복용",
    "검사",
    "결과",
    "수치",
    "증상",
    "질문",
    "알려",
    "찾아",
    "확인",
    "자세히",
    "되나요",
    "근거",
    "출처",
    "최신",
    "현재",
}


def _query_context_error(
    query: str,
    messages: Sequence[ChatMessage],
) -> str | None:
    """Reject guessed or unresolved multi-turn retrieval targets.

    L2 performs the rewrite in its existing Generation decision turn.  This
    validator does not infer a clinical entity; it only verifies that a query
    which follows an anaphoric user turn has removed the anaphora and retained a
    concrete token that the user actually supplied earlier.
    """

    compact_query = _compact_text(query)
    if _contains_sensitive_identifier(query):
        return "retrieval query contains an unnecessary personal identifier"
    if any(
        marker in compact_query
        for marker in ("전체history", "fullhistory", "applicationcontext", "toolcalls")
    ) or re.search(r"(?:^|\s)(?:user|assistant|system)\s*[:=]", query, re.IGNORECASE):
        return "retrieval query contains conversation or protocol history"
    if any(marker in compact_query for marker in _ANAPHORA_MARKERS):
        return "retrieval query contains unresolved anaphora"

    user_turns = _dialogue_user_turns(messages)
    if not user_turns:
        return "retrieval query has no user source turn"
    preservation_error = _material_query_preservation_error(query, user_turns[-1])
    if preservation_error is not None:
        return preservation_error
    if len(user_turns) < 2 or not (
        _contains_anaphora(user_turns[-1])
        or _looks_like_elliptical_material_followup(user_turns[-1])
    ):
        return None

    prior_context = next(
        (
            turn
            for turn in reversed(user_turns[:-1])
            if not _is_acknowledgement_only(turn) and _context_anchor_tokens(turn)
        ),
        "",
    )
    anchors = _context_anchor_tokens(prior_context)
    if not anchors:
        return "the user's referenced target is not established"
    if not all(anchor in compact_query for anchor in anchors):
        return "retrieval query dropped the user's established target"

    required_context = _material_context_tokens(f"{prior_context}\n{user_turns[-1]}")
    if required_context and not all(token in compact_query for token in required_context):
        return "retrieval query dropped material time or polarity context"
    return None


def _authoritative_query_error(query: str) -> str | None:
    """Validate user-authored text before forwarding it to the retrieval planner."""

    if not query.strip() or len(query) > _MAX_QUERY_CHARS:
        return "authoritative retrieval query is blank or exceeds its size bound"
    if any(ord(character) < 32 and character not in "\t\n" for character in query):
        return "authoritative retrieval query contains a forbidden control character"
    compact_query = _compact_text(query)
    if _contains_sensitive_identifier(query):
        return "retrieval query contains an unnecessary personal identifier"
    if any(
        marker in compact_query
        for marker in ("전체history", "fullhistory", "applicationcontext", "toolcalls")
    ) or re.search(r"(?:^|\s)(?:user|assistant|system)\s*[:=]", query, re.IGNORECASE):
        return "retrieval query contains conversation or protocol history"
    return None


def _authoritative_retrieval_query(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, str]:
    """Project only user-authored facts into a bounded retrieval query.

    L2 still decides whether to call the application tool. Its free-form rewrite is
    deliberately not forwarded. A self-contained turn is used byte-for-byte; for a
    genuinely elliptical follow-up, the nearest prior user turn with a concrete
    target is prepended and explicit deictic words are removed deterministically.
    This produces a context-complete query without inventing aliases, entities, or
    clinical facts in Python.
    """

    user_turns = _dialogue_user_turns(messages)
    if not user_turns:
        return None, "retrieval query has no user source turn"
    latest_user = user_turns[-1]
    latest_error = _authoritative_query_error(latest_user)
    if latest_error is not None:
        return None, latest_error
    needs_context = _needs_prior_user_context(latest_user)
    if needs_context and _typed_reference_count(latest_user) > 1:
        return None, "multiple referenced targets cannot be resolved safely"
    if not needs_context:
        return latest_user, ""
    if len(user_turns) < 2:
        return None, "the user's referenced target is not established"

    reference_kind = _reference_kind(latest_user)
    prior_context = next(
        (
            turn
            for turn in reversed(user_turns[:-1])
            if _has_established_retrieval_target(turn, reference_kind)
            and not (
                reference_kind == "drug"
                and _has_singular_drug_reference(latest_user)
                and _has_multiple_drug_targets(turn)
            )
        ),
        "",
    )
    if not prior_context:
        return None, "the user's referenced target is not established"
    prior_error = _authoritative_query_error(prior_context)
    if prior_error is not None:
        return None, prior_error
    projected_latest = _remove_explicit_deictics(latest_user)
    if not re.search(r"[0-9a-z가-힣]", projected_latest, re.IGNORECASE):
        return None, "the user's follow-up contains no retrievable question"
    projected = f"{prior_context.rstrip()}\n{projected_latest}"
    if len(projected) > _MAX_QUERY_CHARS:
        return None, "authoritative retrieval query exceeds its size bound"
    if _contains_anaphora(projected_latest):
        return None, "the user's referenced target could not be resolved"
    return projected, ""


_GENERIC_QUERY_TOKENS = {
    "about",
    "answer",
    "breastfeeding",
    "contraindication",
    "contraindications",
    "coverage",
    "current",
    "dose",
    "dosing",
    "during",
    "evidence",
    "guideline",
    "interaction",
    "interactions",
    "label",
    "latest",
    "official",
    "pregnancy",
    "reimbursement",
    "source",
    "sources",
    "use",
    "using",
    "what",
    "그",
    "그건",
    "그거",
    "그것",
    "그럼",
    "근거",
    "금기",
    "급여",
    "먹어도",
    "부작용",
    "복용",
    "상호작용",
    "사용",
    "소아",
    "수유",
    "수유중",
    "약",
    "약가",
    "어린이",
    "영아",
    "용량",
    "용법",
    "효능",
    "임신",
    "임신중",
    "적응증",
    "진료지침",
    "출처",
    "투여",
    "허가사항",
    "확인",
}

_GENERIC_QUERY_TOKEN_PREFIXES = (
    "괜찮",
    "근거",
    "금기",
    "급여",
    "먹어",
    "문제",
    "뭐",
    "부작용",
    "복용",
    "보여",
    "상호작용",
    "사용",
    "수유",
    "알려",
    "약가",
    "어떻",
    "용량",
    "용법",
    "효능",
    "임신",
    "있",
    "없",
    "적응증",
    "출처",
    "투여",
    "허가사항",
    "확인",
    "자세히",
)


def _concrete_target_tokens(value: str) -> set[str]:
    """Return user-written subject tokens, excluding query operators/modifiers."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    targets: set[str] = set()
    for raw in re.findall(r"[a-z가-힣][0-9a-z가-힣-]*", normalized):
        token = raw
        for suffix in (
            "이라고",
            "이라면",
            "에서는",
            "에게는",
            "으로",
            "에서",
            "에게",
            "에는",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "와",
            "과",
            "도",
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                break
        if (
            len(token) < 2
            or token in _GENERIC_QUERY_TOKENS
            or token.startswith(_GENERIC_QUERY_TOKEN_PREFIXES)
        ):
            continue
        if token in _CONTEXT_STOPWORDS or token in {"알려줘", "보여줘", "궁금해", "괜찮아"}:
            continue
        targets.add(token)
    return targets


def _reference_kind(value: str) -> str:
    compact = _compact_text(value)
    if any(marker in compact for marker in ("그약", "이약")) or re.search(
        r"\b(?:it|that|this)\s+(?:drug|medicine|medication)\b",
        value,
        re.IGNORECASE,
    ):
        return "drug"
    if any(marker in compact for marker in ("그검사", "이검사", "아까검사")):
        return "test"
    if any(marker in compact for marker in ("그수치", "이수치", "아까수치")):
        return "value"
    if any(marker in compact for marker in ("그결과", "이결과")):
        return "result"
    if any(marker in compact for marker in ("그증상", "이증상")):
        return "symptom"
    if any(
        marker in compact
        for marker in (
            "임신",
            "수유",
            "복용",
            "투여",
            "용량",
            "용법",
            "효능",
            "금기",
            "상호작용",
            "부작용",
            "허가사항",
            "급여",
            "약가",
        )
    ) or re.search(
        r"\b(?:pregnancy|breastfeeding|dose|dosing|interaction|side\s+effect|"
        r"contraindication|official\s+label|reimbursement)\b",
        value,
        re.IGNORECASE,
    ):
        return "drug"
    return "generic"


def _has_established_retrieval_target(value: str, reference_kind: str) -> bool:
    """Conservatively identify a user-established clinical/source subject.

    A short bare name is allowed so unseen medicines and diagnoses generalize,
    while greetings, conversational filler, and arbitrary prose cannot become an
    antecedent merely because they contain a long token.
    """

    compact = _compact_text(value)
    if _contains_anaphora(value) or not compact or re.fullmatch(
        r"(?:안녕(?:하세요)?(?:도와(?:줘|주세요))?|도와(?:줘|주세요)|"
        r"무엇을도와드릴까요|반가워요|"
        r"고마워요?|감사합니다|알겠어요?|iunderstand|hello|hi|thisisfictional|"
        r"오늘은좀피곤해요)",
        compact,
    ):
        return False
    if re.fullmatch(
        r"(?:서울|부산|대구|인천|광주|대전|울산|세종|제주|경기|강원|충북|충남|전북|전남|경북|경남)",
        compact,
    ):
        return False
    targets = _concrete_target_tokens(value)
    if not targets:
        return False
    normalized = unicodedata.normalize("NFKC", value).casefold()
    bare_target = (
        len(targets) == 1
        and len(normalized.strip()) <= 32
        and bool(
            re.fullmatch(
                r"[a-z가-힣][0-9a-z가-힣-]*",
                normalized.strip(),
            )
        )
        and not re.search(r"(?:해요|어요|아요|나요|습니다|입니다|다)$", normalized.strip())
    )
    if reference_kind == "drug":
        return bare_target or bool(
            re.search(
                r"복용|투여|처방|먹(?:고|는|었|습니다|어요)|의약품|약물|성분|제품|"
                r"허가사항|적응증|금기|상호작용|부작용|용량",
                normalized,
            )
            or re.search(
                r"\b(?:drug|medicine|medication|prescribed|prescription|dose|dosing|"
                r"ingredient|product|contraindication|interaction|side\s+effect)\b",
                normalized,
            )
            or re.search(
                r"\b(?:take|takes|taking)\s+(?!a\s+break\b)(?:the\s+)?"
                r"[a-z][a-z0-9-]{2,}\b",
                normalized,
            )
        )
    if reference_kind == "test":
        return bool(re.search(r"검사|검진|test|screening|assay", normalized))
    if reference_kind == "value":
        return bool(
            re.search(r"수치|결과|혈압|혈당|심박|체온|value|level|reading|result", normalized)
            or re.search(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|mmhg|mmol|%|bpm)", normalized)
        )
    if reference_kind == "result":
        return bool(re.search(r"검사|결과|판독|test|result|report", normalized))
    if reference_kind == "symptom":
        return bool(
            re.search(r"증상|통증|아프|불편|발열|기침|symptom|pain|fever|cough", normalized)
        )

    medical_or_source_context = (
        "복용",
        "먹",
        "투여",
        "처방",
        "약물",
        "의약품",
        "검사",
        "수치",
        "진단",
        "질환",
        "질병",
        "증상",
        "통증",
        "혈압",
        "혈당",
        "심박",
        "목표",
        "허가",
        "적응증",
        "급여",
        "약가",
        "상호작용",
        "부작용",
        "금기",
        "용량",
        "용법",
        "효능",
        "임신",
        "수유",
        "소아",
        "법령",
        "법률",
        "의료법",
        "약사법",
        "조문",
        "kcd",
        "drug",
        "medicine",
        "medication",
        "taking",
        "dose",
        "diagnosis",
        "test",
        "result",
        "guideline",
    )
    if any(marker in compact for marker in medical_or_source_context):
        return True
    if re.search(r"(?<![a-z0-9])[a-z]\d{2}(?:\.\d+)?(?![a-z0-9])", value, re.I):
        return True
    # A single short surface form may itself be an unseen medicine/product/code.
    # Multi-token prose without a clinical relation is intentionally rejected.
    return bare_target


def _remove_explicit_deictics(value: str) -> str:
    """Remove only explicit references after their exact user context is prepended."""

    projected = unicodedata.normalize("NFKC", value)
    projected = re.sub(
        r"(?:그걸|이걸|(?:그|이|해당)\s*(?:약|검사|수치|결과|증상|거|것|건|게|내용|근거|출처)|"
        r"앞서\s*말한\s*(?:약|검사|수치|결과|증상))"
        r"(?:은|는|이|가|을|를|와|과)?",
        "",
        projected,
    )
    projected = re.sub(r"^\s*(?:그럼|그러면)\s*", "", projected)
    projected = re.sub(
        r"\b(?:what\s+about|it|this|that\s+(?:drug|medicine|test|result)|"
        r"this\s+(?:drug|medicine))\b",
        "",
        projected,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", projected).strip(" \t\r\n,;:")


def _material_query_preservation_error(query: str, latest_user: str) -> str | None:
    query_compact = _compact_text(query)
    latest_compact = _compact_text(latest_user)
    semantic_groups = (
        ("식약처", "mfds", "허가"),
        ("심평원", "심평", "hira", "급여", "약가", "수가"),
        ("법령", "법률", "의료법", "약사법", "감염병예방법", "조문"),
        ("kcd", "질병분류", "상병코드"),
    )
    for group in semantic_groups:
        if any(marker in latest_compact for marker in group) and not any(
            marker in query_compact for marker in group
        ):
            return "retrieval query dropped the requested authority or evidence domain"

    material_groups = (
        ("중단", "끊었", "복용하지", "먹지않"),
        ("없", "아니", "않"),
        ("현재", "지금"),
        ("과거", "예전", "병력"),
        ("임신",),
        ("수유",),
        ("소아", "어린이", "영아", "신생아"),
    )
    for group in material_groups:
        if any(marker in latest_compact for marker in group) and not any(
            marker in query_compact for marker in group
        ):
            return "retrieval query dropped material status, polarity, time, or population"

    numeric_facts = re.findall(
        r"\d+(?:\.\d+)?(?:mg|mcg|g|kg|ml|l|mmol|%|정|알|분|시간|일|주|개월|달|년)",
        latest_compact,
    )
    if any(value not in query_compact for value in numeric_facts):
        return "retrieval query dropped a material number and unit"
    return None


def _compact_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", normalized)


def _contains_sensitive_identifier(value: str) -> bool:
    return bool(
        re.search(r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)", value)
        or re.search(r"(?<!\d)(?:0\d{1,2}[ -]?\d{3,4}[ -]?\d{4})(?!\d)", value)
        or re.search(r"(?<!\w)\+?82[ -]?\d{1,2}[ -]?\d{3,4}[ -]?\d{4}(?!\d)", value)
        or re.search(r"(?<!\d)\d{6}[ -]?\d{7}(?!\d)", value)
        or re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", value, re.IGNORECASE)
        or re.search(
            r"(?:환자|등록|병록|차트|medical\s*record)\s*(?:번호|id|no\.?)?"
            r"\s*[:=#-]?\s*[a-z0-9-]{5,}",
            value,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:여권\s*번호|passport\s*(?:number|no\.?))\s*[:=#-]?\s*[a-z0-9-]{5,}",
            value,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:생년월일|date\s*of\s*birth|dob)\s*[:=]?\s*\d{4}[./ -]\d{1,2}[./ -]\d{1,2}",
            value,
            re.IGNORECASE,
        )
        or re.search(r"(?:주소|address)\s*[:=]\s*\S.{3,}", value, re.IGNORECASE)
        or re.search(
            r"[가-힣]{2,}(?:도|시)\s+[가-힣]{1,}(?:시|군|구)\s+"
            r"[가-힣0-9-]+(?:로|길)\s*\d+",
            value,
        )
        or re.search(
            r"(?:성명|환자\s*이름|patient\s*name)\s*[:=]\s*[가-힣a-z][가-힣a-z .'-]{1,60}",
            value,
            re.IGNORECASE,
        )
    )


def _contains_anaphora(value: str) -> bool:
    compact = _compact_text(value)
    if any(marker in compact for marker in _ANAPHORA_MARKERS):
        return True
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return bool(
        re.search(
            r"\b(?:it|is\s+this\s+safe|that\s+(?:drug|medicine|test|result)|"
            r"this\s+(?:drug|medicine))\b",
            normalized,
        )
    )


def _looks_like_elliptical_material_followup(value: str) -> bool:
    compact = _compact_text(value)
    if len(compact) <= 60 and re.fullmatch(
        r"(?:계속|아직)?(?:이거|이것|이걸|그거|그것|그걸)?(?:같이)?"
        r"(?:먹어도|복용해도|투여해도)(?:돼|되|괜찮|문제없).+",
        compact,
    ):
        return True
    korean_markers = (
        "먹어도돼",
        "먹어도되",
        "복용해도",
        "투여해도",
        "같이먹어도",
        "임신",
        "수유",
        "소아는",
        "용량",
        "용법",
        "효능",
        "금기",
        "상호작용",
        "부작용",
        "허가사항",
        "급여",
        "약가",
        "근거",
        "출처",
    )
    if (
        len(compact) <= 60
        and any(marker in compact for marker in korean_markers)
        and not _concrete_target_tokens(value)
    ):
        return True
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return len(normalized) <= 100 and not _concrete_target_tokens(value) and bool(
        re.search(
            r"\b(?:what\s+about|during\s+(?:pregnancy|breastfeeding)|"
            r"interactions?|side\s+effects?|contraindications?|dosing|coverage|"
            r"reimbursement|official\s+label)\b",
            normalized,
        )
    )


def _dialogue_user_turns(messages: Sequence[ChatMessage]) -> list[str]:
    return [
        message.content or ""
        for message in messages
        if message.role == "user"
        and getattr(message, "_internal_origin", None) != "caller_context"
    ]


def _context_anchor_tokens(value: str) -> set[str]:
    anchors: set[str] = set()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    for raw in re.findall(r"[0-9a-z가-힣]+", normalized):
        token = raw
        for suffix in (
            "으로",
            "에서",
            "에게",
            "한테",
            "에는",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "와",
            "과",
            "도",
        ):
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                token = token[: -len(suffix)]
                break
        if len(token) >= 2 and token not in _CONTEXT_STOPWORDS:
            anchors.add(token)
    return anchors


def _material_context_tokens(value: str) -> set[str]:
    compact = _compact_text(value)
    tokens: set[str] = set()
    for match in re.finditer(r"\d+(?:분|시간|일|주|개월|달|년)", compact):
        tokens.add(match.group(0))
    for marker in (
        "오늘",
        "어제",
        "내일",
        "현재",
        "지금",
        "중단",
        "끊었",
        "복용안",
        "먹지않",
        "없",
        "아니",
        "끝났",
        "해소",
    ):
        if marker in compact:
            tokens.add(marker)
    return tokens


def _evidence_json(result: RetrievalResult) -> str:
    evidence_status = {
        "sufficient": "partial",
        "partial": "partial",
        "no_evidence": "none",
    }[result.status]
    semantic_reason = result.semantic_reason
    source_normalization_status = "unchanged"
    source_normalization_reason_codes: list[str] = []
    if result.status == "sufficient":
        # Dashboard-v1 supplies selection, not claim entailment or verified source
        # normalization metadata. Until an approved catalog adapter proves those
        # fields, sufficient must be monotonically downgraded rather than guessed.
        semantic_reason = "coverage_gap"
        source_normalization_status = "monotonic_downgrade"
        source_normalization_reason_codes = ["catalog_unverified"]
    elif result.status == "partial" and semantic_reason == "completed":
        semantic_reason = "coverage_gap"
    elif result.status == "no_evidence" and semantic_reason == "completed":
        semantic_reason = "no_match"

    items = []
    citable_items = _citable_evidence_items(result)
    if not citable_items:
        evidence_status = "none"
        if semantic_reason == "completed":
            semantic_reason = "no_match"
    # Scrub every selected ledger UID, including identifiers from shells that were
    # dropped as non-citable but may still be cross-referenced by a useful item.
    selected_cite_uids = [item.cite_uid for item in result.items]
    for citation_id, item in enumerate(citable_items, start=1):
        source_type, source_role = _source_classification(item.source_tool)
        limitations = ["source_metadata_unavailable"]
        if _is_truncated_evidence(item.content):
            limitations.append("content_truncated")
        model_facing_content = item.content
        for selected_cite_uid in selected_cite_uids:
            model_facing_content = _model_facing_evidence_content(
                model_facing_content,
                cite_uid=selected_cite_uid,
            )
        projected_item: dict[str, Any] = {
            "citation_id": citation_id,
            "relevance_score": item.relevance_score,
            "limitations": limitations,
            # ``cite_uid`` belongs only to the request-local server ledger. The
            # final model receives the stable numeric citation_id instead.
            "content": model_facing_content,
        }
        # Unknown/null catalog metadata carries no model-facing information. Keep
        # verified classifications when present because they can materially change
        # how the final L2 applies the evidence.
        if source_type is not None:
            projected_item["source_type"] = source_type
        if source_role != "unknown":
            projected_item["source_role"] = source_role
        items.append(projected_item)
    payload = {
        "schema_version": "retrieval-evidence-v4",
        "selection_contract": "dashboard_v1",
        "claim_mapping_status": "not_available",
        "trust_level": "untrusted_evidence",
        "evidence_status": evidence_status,
        "semantic_reason": semantic_reason,
        "execution_status": result.execution_status,
        "routing_status": result.routing_status,
        "source_normalization_status": source_normalization_status,
        "items": items,
    }
    if source_normalization_reason_codes:
        payload["source_normalization_reason_codes"] = (
            source_normalization_reason_codes
        )
    retrieval_note = _model_facing_retrieval_note(
        result.note,
        cite_uids=selected_cite_uids,
    )
    if retrieval_note:
        payload["retrieval_note"] = retrieval_note
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _citable_evidence_items(result: RetrievalResult) -> list[Any]:
    """Return only items whose model-facing payload contains an actual fact.

    An MCP result may be structurally successful while carrying an empty object,
    an identifier-only object, or text removed by the trust-boundary sanitizer.
    Such an item cannot support a user-visible claim and must not acquire a numeric
    citation label merely because the transport returned an item shell.
    """

    selected_cite_uids = [item.cite_uid for item in result.items]
    citable = []
    for item in result.items:
        content = item.content
        for cite_uid in selected_cite_uids:
            content = _model_facing_evidence_content(content, cite_uid=cite_uid)
        if _contains_citable_content(content):
            citable.append(item)
    return citable


def _contains_citable_content(content: str) -> bool:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, ValueError, RecursionError):
        value = content

    redactions = {
        "[internal-id]",
        "[internal-source]",
        "[nested-evidence-redacted]",
    }

    def contains(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(contains(child) for child in value.values())
        if isinstance(value, list):
            return any(contains(child) for child in value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped in redactions:
                return False
            if stripped.startswith(("{", "[")):
                try:
                    nested = json.loads(stripped)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pass
                else:
                    return contains(nested)
            meaningful = re.sub(r"[\W_]", "", stripped, flags=re.UNICODE)
            return bool(meaningful)
        return value is not None

    return contains(value)


def _model_facing_evidence_content(content: str, *, cite_uid: str) -> str:
    """Remove request-local citation identifiers from model-facing evidence.

    MCP evidence is commonly a JSON object whose smallest citable fragment still
    contains its ``cite_uid`` field.  That identifier remains in ``EvidenceItem``
    for server-side validation, but it is neither medical evidence nor a public
    citation label.  Structured removal preserves neighboring facts such as KCD
    code, revision, and name.  A bounded recursive pass also handles JSON encoded
    inside a string; malformed plain text only has the exact opaque UID redacted.
    """

    internal_protocol_keys = {
        "functioncall",
        "functioncalls",
        "toolcall",
        "toolcalls",
    }

    def clean_text(value: str) -> str:
        sanitized = value.replace(cite_uid, "[internal-id]") if cite_uid else value
        sanitized = re.sub(
            r"(?i)(?<![a-z0-9])cite[_\s-]?uids?(?:\s*[:=]\s*|\s+)[^\s,;]+",
            "[internal-id]",
            sanitized,
        )
        for name in sorted(_PROTOCOL_FUNCTION_NAMES, key=len, reverse=True):
            sanitized = re.sub(
                rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])",
                "[internal-source]",
                sanitized,
                flags=re.IGNORECASE,
            )
        return re.sub(
            r"(?i)(?:tool|function)[_\s-]?calls?",
            "internal-protocol",
            sanitized,
        )

    def clean(value: Any, depth: int = 0) -> Any:
        if depth >= 12:
            if isinstance(value, str):
                return clean_text(value)
            if isinstance(value, (dict, list)):
                return "[nested-evidence-redacted]"
            return value
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, child in value.items():
                normalized_key = re.sub(
                    r"[^a-z0-9]",
                    "",
                    unicodedata.normalize("NFKC", str(key)).casefold(),
                )
                if normalized_key in {"citeuid", "citeuids"} | internal_protocol_keys:
                    continue
                sanitized_key = clean_text(str(key))
                if sanitized_key in sanitized:
                    continue
                sanitized[sanitized_key] = clean(child, depth + 1)
            return sanitized
        if isinstance(value, list):
            return [clean(child, depth + 1) for child in value]
        if isinstance(value, str):
            stripped = value.lstrip()
            if stripped.startswith(("{", "[")):
                try:
                    nested = json.loads(value)
                except (json.JSONDecodeError, ValueError, RecursionError):
                    pass
                else:
                    return json.dumps(
                        clean(nested, depth + 1),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
            return clean_text(value)
        return value

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return clean_text(content)
    return json.dumps(
        clean(parsed),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _model_facing_retrieval_note(note: str, *, cite_uids: Sequence[str]) -> str:
    """Keep limitation prose while removing model/tool ledger surface forms."""

    wrapped = json.dumps({"note": note}, ensure_ascii=False)
    cleaned = _model_facing_evidence_content(wrapped, cite_uid="")
    for cite_uid in cite_uids:
        cleaned = _model_facing_evidence_content(cleaned, cite_uid=cite_uid)
    try:
        value = json.loads(cleaned).get("note", "")
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return ""
    return value if isinstance(value, str) else ""


def _retrieval_failure(code: str) -> RetrievalResult:
    safe_code = "".join(character for character in code if character.isalnum() or character in "_-")
    normalized = safe_code.casefold()
    if "timeout" in normalized:
        execution_status = "timeout"
    elif any(
        marker in normalized
        for marker in (
            "schema",
            "protocol",
            "invalid",
            "binding",
            "finalizer",
            "registry",
            "mixed",
        )
    ):
        execution_status = "schema_error"
    elif "budget" in normalized or "max_calls" in normalized:
        execution_status = "budget_exhausted"
    else:
        execution_status = "source_unavailable"
    return RetrievalResult(
        status="no_evidence",
        note=f"retrieval execution unavailable ({safe_code[:80] or 'failed'})",
        execution_status=execution_status,
        semantic_reason="no_match",
        routing_status="appropriate",
    )


def _final_violations(
    completion: L2Completion,
    *,
    retrieval: RetrievalResult | None = None,
    phase: FinalPhase | None = None,
) -> list[str]:
    violations: list[str] = []
    if completion.finish_reason not in (None, "stop"):
        violations.append("invalid_finish_reason")
    if completion.tool_calls:
        violations.append("assistant_tool_calls")
    content = completion.content
    if content is None or not content.strip():
        violations.append("blank_content")
        return violations
    if _looks_like_tool_protocol(content):
        violations.append("tool_protocol_text")
    if retrieval is not None and retrieval.execution_status == "ok":
        violations.extend(_citation_violations(content, retrieval))
    if _phase_has_no_authoritative_evidence(phase, retrieval) and (
        _contains_unsupported_authoritative_claim(content)
    ):
        violations.append("unsupported_authoritative_claim")
    if phase == "emergency" and _contains_unsupported_emergency_claim(content):
        violations.append("unsupported_emergency_claim")
    return violations


def _safe_completion_permitted(l2_client: Any) -> bool:
    """Fail closed when a client cannot prove that request deadline remains."""

    remaining = getattr(l2_client, "remaining_request_seconds", None)
    if not callable(remaining):
        return False
    try:
        seconds = float(remaining())
    except (TypeError, ValueError):
        return False
    return seconds > 0.0


def _safe_completion_violations(
    completion: L2Completion,
    *,
    phase: SafeCompletionPhase = "normal",
) -> list[str]:
    """Validate the minimal L2-authored notice without interpreting medicine."""

    violations: list[str] = []
    if completion.finish_reason not in (None, "stop"):
        violations.append("invalid_finish_reason")
    if completion.tool_calls:
        violations.append("assistant_tool_calls")
    content = completion.content
    if content is None or not content.strip():
        violations.append("blank_content")
        return violations

    normalized = unicodedata.normalize("NFKC", unquote(content)).casefold()
    if _looks_like_tool_protocol(content):
        violations.append("tool_protocol_text")
    if re.search(r"\[\s*\d+\s*\]", normalized) or "cite_uid" in normalized:
        violations.append("citation_token")
    control_tokens = (
        "generation-input-v1",
        "generation-safe-completion-v1",
        "application_context",
        "final_phase_context",
        "runtime_context",
        "safe_completion_final",
        '"schema_version"',
        '"phase"',
    )
    if any(token in normalized for token in control_tokens) or re.search(
        r"</?(?:system|assistant|user|runtime_context|phase)\b",
        normalized,
    ):
        violations.append("control_token")
    if (
        normalized.lstrip().startswith(("{", "[", "```"))
        or re.search(r"(?:^|\n)\s*[-*#•]\s+", normalized)
    ):
        violations.append("non_plain_format")
    if re.search(r"[가-힣]", content) is None:
        violations.append("non_korean_content")
    sentence_count = len(
        [
            sentence
            for sentence in re.findall(r"[^.!?。！？\r\n]+[.!?。！？]?", content)
            if sentence.strip()
        ]
    )
    if not 1 <= sentence_count <= 2:
        violations.append("invalid_sentence_count")

    diagnosis_disclaimer = re.search(
        r"(?:의학적\s*)?진단.{0,18}(?:대신|대체).{0,10}(?:않|아니|없|못)|"
        r"(?:의학적\s*)?진단.{0,14}(?:아니|효력.{0,6}없)",
        normalized,
    )
    if diagnosis_disclaimer is None:
        violations.append("missing_diagnosis_disclaimer")
    if phase == "emergency":
        emergency_destination = re.search(
            r"(?:119|응급\s*(?:서비스|번호|실|의료)|구급|현지\s*응급)",
            normalized,
        )
        emergency_action = re.search(
            r"(?:연락|전화|호출|가(?:세|야|도록)|방문|도움)",
            normalized,
        )
        emergency_urgency = re.search(
            r"(?:즉시|바로|지금|당장|지체\s*없이)",
            normalized,
        )
        delayed_emergency_action = re.search(
            r"(?:나중|내일|며칠\s*후|다음\s*주|시간이\s*되면)",
            normalized,
        )
        negated_emergency_action = re.search(
            r"(?:119|응급\s*(?:서비스|번호|실|의료)|구급|현지\s*응급)"
            r".{0,24}(?:연락|전화|호출|가(?:세|야|도록)|방문|도움)"
            r".{0,12}(?:하지\s*마|하지\s*않|받지\s*마|말(?:라|아)|"
            r"금지|불필요|피하|필요\s*(?:가\s*)?없)",
            normalized,
        )
        if emergency_destination is None or emergency_action is None:
            violations.append("missing_emergency_action")
        elif negated_emergency_action is not None:
            violations.append("negated_emergency_action")
        if emergency_urgency is None or delayed_emergency_action is not None:
            violations.append("missing_immediate_action")
    else:
        incomplete_notice = re.search(
            r"(?:답변|안내|응답).{0,20}(?:완료|제공|검증|확인).{0,12}"
            r"(?:못|어렵|불가)|"
            r"(?:완료|제공|검증|확인).{0,12}(?:못|어렵|불가)",
            normalized,
        )
        clinician_referral = re.search(
            r"(?:의료진|의사|병원|의료기관|전문가).{0,20}"
            r"(?:평가|진료|상담|확인|방문|찾)",
            normalized,
        )
        negated_clinician_referral = re.search(
            r"(?:의료진|의사|병원|의료기관|전문가).{0,24}"
            r"(?:평가|진료|상담|확인|방문|찾).{0,12}"
            r"(?:하지\s*마|하지\s*않|받지\s*마|말(?:라|아)|"
            r"금지|불필요|피하|필요\s*(?:가\s*)?없)",
            normalized,
        )
        if incomplete_notice is None:
            violations.append("missing_incomplete_notice")
        if clinician_referral is None:
            violations.append("missing_clinician_referral")
        elif negated_clinician_referral is not None:
            violations.append("negated_clinician_referral")
    return violations


def _final_content(
    completion: L2Completion,
    *,
    retrieval: RetrievalResult | None = None,
) -> str:
    violations = _final_violations(completion, retrieval=retrieval)
    if violations:
        raise MalformedUpstreamResponseError("L2 returned an invalid final answer")
    return completion.content or ""


def _phase_has_no_authoritative_evidence(
    phase: FinalPhase | None,
    retrieval: RetrievalResult | None,
) -> bool:
    if phase == "mcp_failure":
        return True
    return (
        phase == "post_retrieval"
        and retrieval is not None
        and not _citable_evidence_items(retrieval)
    )


def _contains_unsupported_authoritative_claim(content: str) -> bool:
    """Detect high-risk official claims in an evidence-unavailable final.

    This is deliberately a structural guard, not a medical answer engine. It
    catches authoritative attribution combined with asserted label, legal,
    quantitative, or monitoring details. Limitations and verification referrals
    remain valid L2-authored answers. The rejected draft is never reused; normal
    clean recovery or sanitized failure semantics apply.
    """

    normalized = unicodedata.normalize("NFKC", unquote(content)).casefold()
    # Contrastive conjunctions start a new clause so "not verified, but X is 5 mg"
    # cannot inherit the abstention marker from the first half.
    clauses = re.split(
        r"(?:[.!?。！？;\n]+|\b(?:but|however|yet)\b|(?:하지만|그러나|그렇지만|반면))",
        normalized,
    )
    authority = re.compile(
        r"(?:공식|현행|최신|허가\s*사항|제품\s*설명서|라벨|식약처|심평원|"
        r"질병\s*분류|상병\s*코드|\bkcd\b|\bmfds\b|\bhira\b|"
        r"법(?:률|령)?|조문|시행(?:일|령|규칙)?|규정|고시|지침|가이드라인|"
        r"권고(?:안)?|official|current\s+guideline|prescribing\s+information|label)"
    )
    quantitative = re.compile(
        r"(?:\d+(?:\.\d+)?\s*(?:~|[-–—]|에서|부터|이상|이하|초과|미만)?\s*"
        r"(?:mg|mcg|μg|ug|g|kg|ml|l|mmhg|%|회|초|분|시간|일|주|개월|달|년)"
        r"(?![a-z0-9])|"
        r"\b(?:mg|mcg|μg|ug|mmhg)\s*/\s*(?:kg|day|일)\b)"
    )
    legal_detail = re.compile(
        r"(?:제\s*\d+(?:의\s*\d+)?\s*조|article\s+\d+|"
        r"(?:징역|벌금|과태료|보존|시행)\s*\d+)"
    )
    official_code = re.compile(r"(?<![a-z0-9])[a-z]\d{2}(?:\.\d+)?(?![a-z0-9])")
    schedule = re.compile(
        r"(?:(?:매|마다|간격|주기|투여\s*(?:전|후)|시작\s*(?:전|후)|"
        r"이후|이내).{0,24}(?:검사|재검|추적|모니터링|관찰)|"
        r"(?:검사|재검|추적|모니터링).{0,24}(?:매|마다|간격|주기|"
        r"투여\s*(?:전|후)|시작\s*(?:전|후)|이후|이내)|"
        r"(?:every|after|before|within).{0,32}(?:test|monitor|follow[- ]?up)|"
        r"(?:test|monitor|follow[- ]?up).{0,32}(?:every|after|before|within))"
    )
    assertion = re.compile(
        r"(?:이다|입니다|한다|합니다|된다|됩니다|정한다|정해져|규정|"
        r"요구|권고|금기|금지|허용|적용|해당|대상|필수|해야|하여야|"
        r"피해야|피하|이다(?:고|며)?|이고|이며|"
        r"\b(?:is|are|must|should|required|recommended|contraindicat\w*|"
        r"avoid\w*|prohibit\w*)\b)"
    )
    positive_assertion = re.compile(
        r"(?:이다|입니다|이고|이며|한다|합니다|된다|됩니다|정해져|"
        r"금기|금지|허용|필수|해야|하여야|피해야|피하|"
        r"\b(?:is|are|must|should|required|recommended|contraindicat\w*|"
        r"avoid(?:ed|s|ing)?|prohibit\w*)\b)"
    )
    limitation = re.compile(
        r"(?:확인(?:하지|할\s*수)\s*(?:못|없)|확인\s*(?:불가|필요|해야)|"
        r"근거.{0,16}(?:없|부족|확인)|알\s*수\s*없|단정할\s*수\s*없|"
        r"말씀드릴\s*수\s*없|제시할\s*수\s*없|검증되지\s*않|"
        r"불확실|달라질\s*수|문의|참조|확인해\s*(?:주|보)|"
        r"(?:cannot|could\s+not|unable\s+to)\s+(?:verify|confirm|state)|unverified|"
        r"uncertain|check\s+with|consult)"
    )

    for clause in clauses:
        compact = clause.strip()
        if not compact:
            continue
        has_authority = bool(authority.search(compact))
        has_detail = bool(
            quantitative.search(compact)
            or legal_detail.search(compact)
            or official_code.search(compact)
            or schedule.search(compact)
        )
        asserted_authority = has_authority and bool(assertion.search(compact))
        if not (has_detail or asserted_authority):
            continue
        limitation_match = limitation.search(compact)
        if limitation_match:
            # A limitation only qualifies the proposition it grammatically
            # negates ("500 mg인지 확인할 수 없다").  It cannot launder an
            # already-positive assertion followed by a disclaimer ("500 mg이고
            # 근거 확인 필요").
            preceding = compact[: limitation_match.start()]
            positive_before_limitation = positive_assertion.search(preceding)
            if not positive_before_limitation:
                continue
        # Exact clinical quantities, schedules, legal details, and official codes are
        # inherently source-dependent in this phase even if the draft omits the
        # attribution and presents remembered material as generic knowledge.
        if (
            quantitative.search(compact)
            or schedule.search(compact)
            or legal_detail.search(compact)
            or official_code.search(compact)
            or (has_authority and has_detail)
            or asserted_authority
        ):
            return True
    return False


def _contains_unsupported_emergency_claim(content: str) -> bool:
    """Reject unverifiable or unsafe instructions in the no-evidence emergency phase.

    Emergency generation intentionally performs no retrieval.  This guard therefore
    rejects claims that a current authoritative source was consulted, new oral-drug
    instructions with concrete dosing, and a small class of unsafe bleeding
    procedures.  It does not author or repair medical text; the caller discards the
    draft and gives L2 one fresh clean-recovery attempt.
    """

    normalized = unicodedata.normalize("NFKC", unquote(content)).casefold()
    clauses = re.split(
        r"(?:[.!?。！？;\n]+|\b(?:but|however|yet)\b|(?:하지만|그러나|그렇지만|반면))",
        normalized,
    )
    authority = re.compile(
        r"(?:공식|현행|최신|정부|보건당국|학회|학술지|저널|논문|법(?:률|령)?|조문|"
        r"가이드라인|지침|식약처|질병관리청|심평원|"
        r"official|current|latest|government|health\s+authorit|societ|academy|"
        r"journal|law|statute|guideline)"
    )
    source_attestation = re.compile(
        r"(?:확인(?:했|한|됐|된)|조회(?:했|한|됐|된)|검색(?:했|한|됐|된)|"
        r"찾아보(?:니|았)|검토(?:했|한)|발표(?:했|한)|게시(?:했|한)|"
        r"따르면|의하면|자료를?\s*보면|(?:법령|법률|조문)\s*상|"
        r"명시(?:했|한)|권고(?:했|한|한다|합니다)|"
        r"checked|verified|looked\s+up|searched|reviewed|published|states?|"
        r"according\s+to|recommends?)"
    )
    limitation = re.compile(
        r"(?:확인(?:하지|할\s*수)\s*(?:못|없)|확인\s*(?:불가|필요)|"
        r"조회(?:하지|할\s*수)\s*(?:못|없)|검색(?:하지|할\s*수)\s*(?:못|없)|"
        r"검증되지\s*않|출처.{0,16}(?:없|부족|미확인)|"
        r"could\s+not\s+(?:check|verify)|cannot\s+(?:check|verify|confirm)|"
        r"not\s+(?:checked|verified)|unverified)"
    )
    url = re.compile(r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:go\.kr|gov|org|edu)(?:/|\b))")
    oral_action = re.compile(
        r"(?:복용(?:하|해|하세요|하십시오|해야)|경구로\s*(?:먹|복용|투여)|"
        r"(?:약|알약|정제|캡슐|시럽)(?:을|를)?\s*(?:먹|삼키|복용)|"
        r"\b(?:take|start|swallow|chew)\b.{0,36}"
        r"\b(?:medicine|medication|drug|pill|tablet|capsule|dose)\b|"
        r"\b(?:medicine|medication|drug|pill|tablet|capsule)\b.{0,36}"
        r"\b(?:take|start|swallow|chew)\b)"
    )
    concrete_dose = re.compile(
        r"(?:\d+(?:\.\d+)?\s*(?:~|[-–—]|에서|부터|이상|이하)?\s*"
        r"(?:mg|mcg|μg|ug|g|ml|정|알|회)(?![a-z0-9])|"
        r"(?:한|두|세|네)\s*(?:정|알)|"
        r"(?:매|마다|간격).{0,16}(?:분|시간|일)|"
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|pills?|capsules?)\b)"
    )
    medication_marker = re.compile(
        r"(?:약|복용|경구|알약|정제|캡슐|시럽|medicine|medication|drug|pill|tablet|capsule)"
    )
    oral_dose_action = re.compile(
        r"(?:먹|삼키|씹|복용|드세요|드십시오|혀\s*밑.{0,12}넣|"
        r"\b(?:take|swallow|chew)\b)"
    )
    dispatcher_or_existing_plan = re.compile(
        r"(?:(?:119|응급|구급).{0,20}(?:상담원|상황실|대원).{0,24}"
        r"(?:지시|안내|말|권)|"
        r"(?:dispatcher|emergency\s+operator|paramedic).{0,24}"
        r"(?:instruct|direct|tell|told|advise)|"
        r"(?:이미|기존|평소|사전|미리).{0,20}"
        r"(?:처방받|처방된|응급\s*계획|구조\s*계획|rescue\s+plan|action\s+plan)"
        r".{0,28}(?:대로|따라|지시|as\s+directed)|"
        r"(?:take|use).{0,24}(?:prescribed\s+)?rescue\s+(?:medicine|medication)"
        r".{0,24}as\s+directed.{0,24}(?:existing\s+)?rescue\s+plan)"
    )
    negative_instruction = re.compile(
        r"(?:하지\s*마|사용하지\s*마|금지|피하|임의로.{0,12}(?:말|않)|"
        r"do\s+not|don't|never|avoid)"
    )
    targeted_negative_oral_instruction = re.compile(
        r"(?:(?:do\s+not|don't|never)\s+(?:immediately\s+|now\s+)?"
        r"(?:take|swallow|chew)|"
        r"(?:복용|먹|삼키|씹|드시|드)[가-힣\s]{0,8}(?:않|말|마)|"
        r"(?:복용|섭취)(?:을|를)?\s*피하)"
    )
    exception_cancellation = re.compile(
        r"(?:ignore|disregard|regardless\s+of|irrespective\s+of|"
        r"무시|무관|상관\s*없이|관계\s*없이)"
    )
    tourniquet_action = re.compile(
        r"(?:지혈대|tourniquet).{0,28}(?:사용|적용|감|묶|조이|apply|tie|tighten)"
    )
    elevation_action = re.compile(
        r"(?:(?:팔|다리|사지|상처\s*부위).{0,24}(?:심장보다\s*)?"
        r"(?:높이|높게|올리)|\belevat(?:e|ing)\b.{0,24}(?:limb|arm|leg|wound))"
    )
    induced_vomiting_action = re.compile(
        r"(?:구토(?:를)?\s*(?:유도|시키)|"
        r"(?:억지로\s*)?토(?:하게|하도록)\s*(?:하|만들)|"
        r"(?:induc(?:e|ing)|cause)\s+(?:them\s+|the\s+(?:person|patient|child)\s+)?"
        r"(?:vomit(?:ing)?|emesis)|"
        r"make\s+(?:them|him|her|the\s+(?:person|patient|child))\s+"
        r"(?:vomit|throw\s+up))"
    )
    milk_or_water_action = re.compile(
        r"(?:(?:물|우유)(?:을|를)?\s*(?:마시|마셔|먹이|먹으|드세|섭취)|"
        r"(?:마시|마셔|먹이|먹으|드세|섭취).{0,12}(?:물|우유)|"
        r"\b(?:drink|sip|give|administer)\b.{0,24}\b(?:water|milk)\b)"
    )
    decontamination_context = re.compile(
        r"(?:독|유독|독성|부식|화학|약품|세제|세정제|락스|표백제|"
        r"산성|염기|중화|희석|씻어\s*내|제거|삼킨|먹은|마신|섭취|"
        r"poison|toxi|corrosive|caustic|chemical|cleaner|detergent|bleach|"
        r"acid|alkali|decontaminat|dilut|flush|neutraliz|swallow|ingest)"
    )
    neutralization_action = re.compile(
        r"(?:중화(?:하|시키)|"
        r"(?:식초|베이킹\s*소다|산|염기|알칼리).{0,24}"
        r"(?:중화|산성|염기성)(?:을|를)?\s*(?:없애|잡)|"
        r"\bneutraliz(?:e|es|ed|ing)\b|"
        r"\bcounteract\b.{0,24}\b(?:chemical|acid|alkali|caustic|corrosive)\b)"
    )
    poison_or_dispatcher_instruction = re.compile(
        r"(?:(?:중독\s*(?:관리|상담)?\s*센터|독극물\s*센터|119|"
        r"응급\s*(?:상담원|상황실|요원)|구급\s*(?:상담원|대원))"
        r".{0,40}(?:지시|안내|말|시키|권고)|"
        r"(?:지시|안내|말|시키|권고).{0,40}"
        r"(?:중독\s*(?:관리|상담)?\s*센터|독극물\s*센터|119|"
        r"응급\s*(?:상담원|상황실|요원)|구급\s*(?:상담원|대원))|"
        r"(?:poison\s+(?:control|center|centre|hotline)|dispatcher|"
        r"emergency\s+(?:operator|dispatcher)|paramedic)"
        r".{0,48}(?:instruct|direct|tell|told|advise|say|says)|"
        r"(?:instruct|direct|tell|told|advise|say|says).{0,48}"
        r"(?:poison\s+(?:control|center|centre|hotline)|dispatcher|"
        r"emergency\s+(?:operator|dispatcher)|paramedic))"
    )
    targeted_negative_vomiting = re.compile(
        r"(?:(?:구토|토(?:하게|하도록)).{0,24}"
        r"(?:하지\s*마|시키지\s*마|유도하지\s*마)|"
        r"(?:하지\s*마|시키지\s*마|유도하지\s*마).{0,20}(?:구토|토)|"
        r"(?:do\s+not|don't|never|avoid|should\s+not|must\s+not)"
        r".{0,32}(?:induc(?:e|ing)\s+(?:vomit(?:ing)?|emesis)|"
        r"make.{0,16}(?:vomit|throw\s+up)))"
    )
    targeted_negative_drinking = re.compile(
        r"(?:(?:물|우유).{0,24}(?:마시지\s*마|먹이지\s*마|주지\s*마)|"
        r"(?:do\s+not|don't|never|avoid|should\s+not|must\s+not)"
        r".{0,32}(?:drink|sip|give|administer).{0,20}(?:water|milk))"
    )
    targeted_negative_neutralization = re.compile(
        r"(?:(?:중화하|중화시키).{0,16}(?:지\s*마|지\s*않)|"
        r"(?:do\s+not|don't|never|avoid|should\s+not|must\s+not)"
        r".{0,32}(?:neutraliz|counteract))"
    )
    has_decontamination_context = bool(decontamination_context.search(normalized))

    for clause in clauses:
        compact = clause.strip()
        if not compact:
            continue
        if not limitation.search(compact) and (
            url.search(compact)
            or (authority.search(compact) and source_attestation.search(compact))
        ):
            return True

        has_exception = bool(dispatcher_or_existing_plan.search(compact)) and not bool(
            exception_cancellation.search(compact)
        )
        dose_match = concrete_dose.search(compact)
        oral_match = oral_action.search(compact) or (
            oral_dose_action.search(compact) if dose_match else None
        )
        if oral_match and dose_match:
            # A concrete new oral dose is never made safe by attributing it to a
            # dispatcher or an existing plan.  Only an actual negative instruction
            # about taking the medicine is conservative.
            if not targeted_negative_oral_instruction.search(compact):
                return True
        elif oral_match and medication_marker.search(compact) and not has_exception:
            if not negative_instruction.search(
                compact[
                    max(0, oral_match.start() - 16) : min(
                        len(compact), oral_match.end() + 20
                    )
                ]
            ):
                return True

        decontamination_exception = bool(
            poison_or_dispatcher_instruction.search(compact)
        ) and not bool(exception_cancellation.search(compact))
        vomiting_match = induced_vomiting_action.search(compact)
        if (
            vomiting_match
            and not targeted_negative_vomiting.search(compact)
            and not decontamination_exception
        ):
            return True
        drinking_match = milk_or_water_action.search(compact)
        if drinking_match:
            drink_text = drinking_match.group(0)
            is_milk_advice = "우유" in drink_text or "milk" in drink_text
            if (
                (is_milk_advice or has_decontamination_context)
                and not targeted_negative_drinking.search(compact)
                and not decontamination_exception
            ):
                return True
        if (
            neutralization_action.search(compact)
            and not targeted_negative_neutralization.search(compact)
            and not decontamination_exception
        ):
            return True
        if has_exception:
            continue
        for unsafe_action in (tourniquet_action, elevation_action):
            unsafe_match = unsafe_action.search(compact)
            if unsafe_match and not negative_instruction.search(
                compact[
                    max(0, unsafe_match.start() - 16) : min(
                        len(compact), unsafe_match.end() + 20
                    )
                ]
            ):
                return True
    return False


def requires_retrieval(messages: Sequence[ChatMessage]) -> bool:
    """Route only source-dependent turns to the slower MCP trajectory.

    The latest user turn is authoritative; one preceding user turn is included
    only to preserve short coreference. Broad words such as ``최신`` or ``출처``
    do not force RAG unless they form an actual evidence request or identify a
    source-dependent Korean coding, approval, reimbursement, or legal domain.
    """

    user_turns = _dialogue_user_turns(messages)
    if not user_turns:
        return False
    latest = user_turns[-1]
    if _is_acknowledgement_only(latest):
        return False

    scoped_turns = [latest]
    if len(user_turns) >= 2 and _needs_prior_user_context(latest):
        scoped_turns.insert(0, user_turns[-2])
    text = unicodedata.normalize("NFKC", "\n".join(scoped_turns)).casefold()
    compact = re.sub(r"\s+", "", text)
    if _explicitly_excludes_source_lookup(compact):
        return False

    if is_official_drug_label_request(text):
        return True

    english_medication_action = re.search(
        r"\b(?:administer|dose|dosing|drug|medication|medicine|prescribe|prescribing|"
        r"take|taking|use|using)\b",
        text,
    )
    english_label_or_safety = re.search(
        r"\b(?:contraindicat(?:e|ed|ion|ions)|interaction|official\s+(?:drug\s+)?label|"
        r"prescribing\s+information|boxed\s+warning)\b",
        text,
    )
    if english_medication_action and english_label_or_safety:
        return True

    current_legal_detail = any(
        marker in compact
        for marker in ("현행", "조문", "시행", "개정", "효력", "현재법")
    )
    legal_subject = any(
        marker in compact
        for marker in ("법률", "법령", "시행령", "시행규칙", "의료법", "약사법")
    ) or bool(
        re.search(
            r"(?:감염병|보건|의료|약사|보험|예방|관리)[가-힣]{0,16}법(?:률)?",
            text,
        )
    )
    if current_legal_detail and legal_subject:
        return True

    authoritative_domains = (
        "cite_uid",
        "kcd",
        "질병분류코드",
        "상병코드",
        "식약처",
        "식약쳐",
        "식약청",
        "mfds",
        "심평원",
        "심평언",
        "hira",
        "건강보험급여",
        "보험급여",
        "비급여",
        "약가",
        "허가사항",
        "허가정보",
        "공식라벨",
        "officiallabel",
        "품목허가",
        "의약품허가",
        "법령",
        "의료법",
        "약사법",
        "감염병예방법",
        "법률조문",
    )
    if any(marker in compact for marker in authoritative_domains) and not (
        _authority_only_quoted_or_mentioned(text)
    ):
        return True
    if "수가" in compact and any(
        marker in compact
        for marker in ("의료수가", "진료수가", "수가기준", "심평", "hira", "급여", "보험")
    ):
        return True

    if re.search(r"(?<![a-z0-9])[a-z]\d{2}(?:\.\d+)?(?![a-z0-9])", text):
        if any(marker in compact for marker in ("공식", "명칭", "코드", "질병분류", "상병")):
            return True

    # Medication/label decisions with a vulnerable population or a material
    # safety property should use current primary evidence.  This is a semantic
    # policy, not a list of benchmark items or product names.
    medication_safety = (
        "임신",
        "수유",
        "소아",
        "어린이",
        "영아",
        "신생아",
        "고령",
        "금기",
        "상호작용",
        "먹어도돼",
        "먹어도되",
        "복용해도",
        "투여해도",
        "용량",
    )
    medication_or_ingestion = (
        "복용",
        "투여",
        "처방",
        "먹어도",
        "약을",
        "약은",
        "약이",
        "약과",
        "약의",
        "약물",
        "의약품",
    )
    if any(marker in compact for marker in medication_safety) and any(
        marker in compact for marker in medication_or_ingestion
    ):
        return True
    if any(marker in compact for marker in ("금기", "상호작용", "용량", "적응증")):
        return True

    if any(marker in compact for marker in ("진단기준", "치료목표", "목표수치")) and any(
        marker in compact for marker in ("현재", "최신", "공식", "지금")
    ):
        return True

    if any(marker in compact for marker in ("격리", "백신", "검사기준", "여행기준")) and any(
        marker in compact for marker in ("코로나", "감염병", "유행", "현재", "지금", "최신")
    ):
        return True

    evidence_artifacts = (
        "진료지침",
        "임상지침",
        "공식지침",
        "지침",
        "가이드라인",
        "guideline",
        "논문",
        "연구결과",
        "임상시험",
        "메타분석",
        "systematicreview",
        "paper",
        "study",
    )
    freshness_or_lookup = (
        "최신",
        "최근",
        "현행",
        "현재기준",
        "찾아",
        "검색",
        "확인",
        "근거",
        "출처",
        "인용",
        "citation",
        "reference",
        "알려",
        "보여",
        "요약",
        "있어",
        "뭐야",
        "무엇",
    )
    if any(
        definition in compact
        for definition in ("가이드라인이란", "진료지침이란", "임상지침이란", "논문이란")
    ) and not any(marker in compact for marker in ("특정", "질환", "치료", "진단", "약")):
        return False
    if any(artifact in compact for artifact in evidence_artifacts) and any(
        request in compact for request in freshness_or_lookup
    ):
        return True

    explicit_source_requests = (
        "출처를알려",
        "출처알려",
        "근거를찾아",
        "근거찾아",
        "근거를제시",
        "공식근거",
        "근거식별자",
        "논문찾아",
        "논문검색",
        "인용해",
        "citationplease",
        "providesources",
        "citesources",
    )
    return (
        any(request in compact for request in explicit_source_requests)
        or bool(re.search(r"(?:^|\s)출처(?:\s|$)", text))
        or bool(re.search(r"\b(?:source|sources|citation|references?)\b", text))
    )


def _explicitly_excludes_source_lookup(compact: str) -> bool:
    excluded = re.search(
        r"(?:검색|출처|근거|식약처|심평원|법령|법률|현행법|논문|지침)"
        r"(?:검색|출처|근거)?말고",
        compact,
    )
    if excluded is None:
        return False
    tail = compact[excluded.end() :]
    alternate_authority = any(
        marker in tail
        for marker in (
            "식약처",
            "심평원",
            "법령",
            "법률",
            "현행법",
            "논문",
            "지침",
            "허가사항",
            "공식라벨",
            "급여",
        )
    ) and any(marker in tail for marker in ("확인", "찾아", "검색", "출처", "근거"))
    if alternate_authority:
        return False
    return any(marker in tail for marker in ("일반", "원리", "뜻", "의미", "설명"))


def _authority_only_quoted_or_mentioned(text: str) -> bool:
    compact = _compact_text(text)
    if any(marker in compact for marker in ("인용", "단어뜻", "표현뜻", "문구", "언급")):
        return not any(
            marker in compact
            for marker in ("조문", "허가", "급여", "코드", "찾아", "검색", "확인", "근거", "출처")
        )
    return False


def _is_acknowledgement_only(value: str) -> bool:
    compact = _compact_text(value)
    return bool(
        re.fullmatch(
            r"(?:고마워(?:요)?|감사(?:합니다|해요)?|알겠(?:어|어요|습니다)?|"
            r"확인했(?:어|어요|습니다)?|오케이|okay|ok|됐(?:어|어요|습니다)?)",
            compact,
        )
    )


def _needs_prior_user_context(value: str) -> bool:
    compact = _compact_text(value)
    if any(
        compact.startswith(marker)
        for marker in ("그건됐고", "그건그만", "그얘기는됐고", "앞내용은됐고")
    ) or re.match(r"^\s*(?:never\s+mind|forget\s+that)\b", value, re.IGNORECASE):
        return False
    reference_count = _typed_reference_count(value)
    apposition_targets = re.findall(
        r"(?:그|이|해당)\s*(?:약|검사|수치|결과|증상)\s+"
        r"([0-9a-z가-힣][0-9a-z가-힣-]*)|"
        r"\b(?:this|that)\s+(?:drug|medicine|test|result|value|symptom)\s+"
        r"([a-z][a-z0-9-]*)\b",
        value,
        re.IGNORECASE,
    )
    concrete_appositions = sum(
        1
        for targets in apposition_targets
        if _concrete_target_tokens(targets[0] or targets[1])
    )
    explicit_apposition = reference_count > 0 and concrete_appositions == reference_count
    unresolved_reference = _contains_anaphora(value) and not explicit_apposition
    return unresolved_reference or _looks_like_elliptical_material_followup(value)


def _typed_reference_count(value: str) -> int:
    return len(
        re.findall(
            r"(?:그|이|해당)\s*(?:약|검사|수치|결과|증상)|"
            r"앞서\s*말한\s*(?:약|검사|수치|결과|증상)|"
            r"\b(?:this|that)\s+(?:drug|medicine|test|result|value|symptom)\b",
            value,
            re.IGNORECASE,
        )
    )


def _has_singular_drug_reference(value: str) -> bool:
    compact = _compact_text(value)
    return _typed_reference_count(value) == 1 and any(
        marker in compact
        for marker in ("그약", "이약", "해당약", "앞서말한약", "앞의약")
    )


def _has_multiple_drug_targets(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return bool(
        re.search(
            r"[0-9a-z가-힣-]{2,}\s*(?:과|와|및|,)\s*"
            r"[0-9a-z가-힣-]{2,}(?:을|를)?\s*"
            r"(?:복용|투여|처방|먹)",
            normalized,
        )
        or re.search(
            r"\b(?:take|takes|taking|prescribed)\b[^.\n]{0,80}"
            r"\b[a-z][a-z0-9-]{2,}\b\s*(?:and|,)\s*"
            r"\b[a-z][a-z0-9-]{2,}\b",
            normalized,
        )
    )


def _is_emergency_turn(messages: Sequence[ChatMessage]) -> bool:
    user_turns = _dialogue_user_turns(messages)
    if not user_turns:
        return False
    scoped_turns = [user_turns[-1]]
    if len(user_turns) >= 2 and _needs_prior_emergency_context(user_turns[-1]):
        scoped_turns.insert(0, user_turns[-2])
    text = unicodedata.normalize("NFKC", "\n".join(scoped_turns)).casefold()
    if _has_current_hazardous_ingestion_or_exposure(text):
        return True
    markers = (
        "의식이 없",
        "숨을 못",
        "호흡이 안",
        "심한 호흡곤란",
        "가슴 통증",
        "흉통",
        "마비",
        "말이 어눌",
        "경련",
        "자살",
        "죽고 싶",
        "과다복용",
        "많이 먹었",
        "일산화탄소",
        "청색증",
        "목이 붓",
        "목 붓",
        "쌕쌕",
        "지혈이 안",
        "지혈 안",
        "편측 위약",
        "발음 이상",
        "anaphyl",
        "unconscious",
        "can't breathe",
        "chest pain",
        "chest tightness",
        "overdose",
        "suicid",
        "face became uneven",
        "face is uneven",
        "face droop",
        "facial droop",
        "facial asymmetry",
        "arm became weak",
        "arm weakness",
        "slurred speech",
        "words are slurred",
    )
    if any(_current_emergency_marker(text, marker) for marker in markers):
        return True
    if _has_current_uncontrolled_bleeding(text):
        return True
    chest_pressure_markers = (
        "가슴 압박",
        "가슴 조임",
        "가슴이 조이",
        "가슴 짓누름",
        "가슴을 짓누르",
        "chest pressure",
        "chest tightness",
    )
    cardiopulmonary_companions = (
        "호흡곤란",
        "숨이 차",
        "숨참",
        "숨가쁨",
        "식은땀",
        "shortness of breath",
        "dyspnea",
        "diaphoresis",
        "sweating",
    )
    korean_pressure_matches = re.findall(
        r"가슴.{0,8}(?:짓눌|압박|조이|쥐어짜)",
        text,
    )
    pressure_is_current = any(
        _current_emergency_marker(text, marker) for marker in chest_pressure_markers
    ) or any(
        _current_emergency_marker(text, marker) for marker in korean_pressure_matches
    )
    if pressure_is_current and any(
        _current_emergency_marker(text, marker)
        for marker in cardiopulmonary_companions
    ):
        return True
    for match in re.finditer(r"(?<!\d)(\d{1,3})\s*(?:알|정).{0,12}(?:먹|복용)", text):
        if int(match.group(1)) >= 10 and not _noncurrent_quantity_context(text, match.start()):
            return True
    # A small surface-form guard catches spacing, punctuation and common fragment
    # forms before retrieval. It does not rewrite the user text or infer clinical
    # entities; the emergency final prompt independently checks
    # negation/history/quotes and returns a natural-language answer.
    compact = re.sub(r"[^0-9a-z가-힣]+", "", text)
    compact_markers = (
        "의식없",
        "반응없",
        "숨못쉬",
        "숨못쉼",
        "숨안쉬",
        "호흡안됨",
        "심한호흡곤란",
        "가슴통증",
        "흉통",
        "말어눌",
        "경련",
        "죽고싶",
        "자살",
        "과다복용",
        "많이먹었",
        "많이먹음",
        "일산화탄소",
        "청색증",
        "목붓",
        "쌕쌕",
        "지혈안",
        "편측위약",
        "발음이상",
        "anaphyl",
        "unconscious",
        "cantbreathe",
        "chestpain",
        "overdose",
        "suicid",
        "facebecameuneven",
        "faceisuneven",
        "facedroop",
        "facialdroop",
        "facialasymmetry",
        "armbecameweak",
        "armweakness",
        "slurredspeech",
        "wordsareslurred",
    )
    return any(_current_emergency_marker(compact, marker) for marker in compact_markers)


def _has_current_uncontrolled_bleeding(text: str) -> bool:
    patterns = (
        r"(?:피|출혈).{0,18}(?:안\s*멈|멈추지|멎지)",
        r"(?:피|출혈).{0,12}(?:계속|지속).{0,12}(?:나|흐르|쏟)",
        r"(?:계속|지속).{0,12}(?:피|출혈).{0,12}(?:나|흐르|쏟)",
        r"(?:압박|눌렀|누르고|누르는).{0,36}"
        r"(?:피|출혈|지혈).{0,20}"
        r"(?:안\s*멈|멈추지|계속|안\s*되|되지\s*않|조절되지\s*않)",
        r"(?:피|출혈).{0,32}(?:압박|눌러).{0,24}"
        r"(?:안\s*멈|멈추지|계속|소용\s*없)",
        r"(?:피|출혈).{0,28}"
        r"(?:분수처럼|뿜|분출|솟구|맥박처럼|박동성으로|쏟아|고이|고여|흥건)",
        r"(?:분수처럼|뿜|분출|솟구|맥박처럼|박동성으로|쏟아|고이|고여|흥건)"
        r".{0,28}(?:피|출혈)",
        r"(?:(?:붕대|거즈|수건|천).{0,24}(?:피|출혈).{0,20}"
        r"(?:흠뻑|젖|스며|배어|뚫고|통과)|"
        r"(?:피|출혈).{0,24}(?:붕대|거즈|수건|천).{0,20}"
        r"(?:흠뻑|젖|스며|배어|뚫고|통과))",
        r"(?:bleeding|blood).{0,32}(?:won['’]?t\s+stop|will\s+not\s+stop|"
        r"keeps?\s+(?:flowing|bleeding)|continu(?:es|ing)|"
        r"spurt|squirt|gush|pulsat|pool|soak(?:s|ed|ing)?\s+through)",
        r"(?:spurt|squirt|gush|pulsat|pool|soak(?:s|ed|ing)?\s+through)"
        r".{0,32}(?:bleeding|blood|bandage|dressing|gauze|towel|cloth)",
        r"(?:direct|firm|hard|steady)?\s*pressure.{0,48}"
        r"(?:does\s+not|doesn't|is\s+not|isn't|won['’]?t|will\s+not|failed?\s+to)"
        r".{0,20}(?:stop|control).{0,16}(?:bleeding|blood)?",
        r"(?:direct|firm|hard|steady)?\s*pressure.{0,40}"
        r"(?:fail(?:ed|s|ing)?|is\s+not|isn't|not)\s+(?:work(?:ing)?|enough)"
        r".{0,28}(?:bleeding|blood)",
        r"(?:bleeding|blood).{0,40}(?:despite|after|even\s+with)"
        r".{0,28}(?:direct|firm|hard|steady)?\s*pressure",
        r"(?:bandage|dressing|gauze|towel|cloth).{0,32}"
        r"(?:soak(?:s|ed|ing)?\s+through|saturat(?:e|ed|ing)).{0,24}"
        r"(?:bleeding|blood)?",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matched_text = match.group(0)
            if re.search(
                r"(?:\b(?:is|are|was|were)\s+not\b|\b(?:isn't|aren't|"
                r"wasn't|weren't)\b).{0,16}"
                r"(?:spurt|squirt|gush|pulsat|pool|soak|saturat)|"
                r"\b(?:would|could|might)\b.{0,24}"
                r"(?:spurt|squirt|gush|pulsat|pool|soak|saturat)",
                matched_text,
                re.IGNORECASE,
            ):
                continue
            if _emergency_span_is_current(text, match.start(), match.end()):
                return True
    return False


def _has_current_hazardous_ingestion_or_exposure(text: str) -> bool:
    """Recognize current hazardous contact without maintaining product-specific cases."""

    hazard = re.compile(
        r"(?:독극물|유독(?:성)?|독성|부식성?|화학\s*(?:물질|약품|제품)?|"
        r"세척제|세정제|배수구\s*(?:세정제|청소제)|락스|표백제|농약|살충제|"
        r"제초제|부동액|휘발유|등유|메탄올|산성\s*(?:물질|용액)|"
        r"염기성?\s*(?:물질|용액)|알칼리|건전지|배터리|전지|코인셀|"
        r"\b(?:poison(?:ous)?|toxic|corrosive|caustic|chemical|cleaner|"
        r"detergent|bleach|pesticide|insecticide|herbicide|antifreeze|"
        r"gasoline|kerosene|methanol|acid|alkali|battery|button\s+cell)\b)"
    )
    risky_substance = re.compile(
        r"(?:약|알약|정제|캡슐|시럽|보충제|비타민|술|알코올|"
        r"\b(?:medicines?|medications?|drugs?|pills?|tablets?|capsules?|"
        r"supplements?|vitamins?|alcohol|substances?)\b)"
    )
    unknown_quantity = re.compile(
        r"(?:(?:얼마나|몇\s*(?:알|정|개|모금)?|양|용량|수량).{0,24}"
        r"(?:모르|알\s*수\s*없|확인(?:이|을)?\s*(?:안|못))|"
        r"(?:모르|알\s*수\s*없|확인(?:이|을)?\s*(?:안|못)).{0,24}"
        r"(?:얼마나|몇\s*(?:알|정|개|모금)?|양|용량|수량)|"
        r"\bunknown\s+(?:amount|quantity|dose|number)\b|"
        r"\b(?:do\s+not|don't|cannot|can't)\s+know\s+how\s+(?:much|many)\b|"
        r"\bnot\s+sure\s+how\s+(?:much|many)\b|"
        r"\b(?:amount|quantity|dose|number)\b.{0,12}\b(?:unknown|unclear)\b)"
    )
    exposure_action = re.compile(
        r"(?:삼켰|삼킨|삼키|삼킴|먹었|먹은|먹어|마셨|마신|마셔|섭취|복용|"
        r"들이마셨|들이마신|흡입|노출|걸렸|박혔|눈에.{0,12}(?:들어|튀)|"
        r"피부에.{0,12}(?:묻|닿|쏟)|(?:입|코|귀)에.{0,12}넣|"
        r"\b(?:swallow(?:ed|ing|s)?|ingest(?:ed|ing|s|ion)?|"
        r"drink(?:ing|s)?|drank|drunk|consume(?:d|s|ing)?|ate|eaten|"
        r"took|taken|inhal(?:e|ed|es|ing|ation)|breathe[ds]?\s+in|"
        r"expos(?:e|ed|es|ing|ure)|splash(?:ed|es|ing)?|spill(?:ed|s|ing)?|"
        r"leak(?:ed|s|ing)?|lodg(?:e|ed|es|ing)|stuck)\b|"
        r"\bgot\b.{0,20}\b(?:eyes?|skin|mouth)\b|"
        r"\bput\b.{0,32}\b(?:mouth|nose|ear)\b)"
    )

    for clause_match in re.finditer(r"[^.!?。！？;\n]+", text):
        clause = clause_match.group(0)
        has_hazard = bool(hazard.search(clause))
        has_unknown_risky_quantity = bool(
            unknown_quantity.search(clause) and risky_substance.search(clause)
        )
        if not (has_hazard or has_unknown_risky_quantity):
            continue
        for action_match in exposure_action.finditer(clause):
            start = clause_match.start() + action_match.start()
            end = clause_match.start() + action_match.end()
            if _emergency_span_is_current(text, start, end):
                return True
    # Unknown quantity is often supplied in the immediately following sentence
    # ("I took some pills. I don't know how many."). Preserve that local
    # relationship without joining unrelated hazards and actions across clauses.
    for action_match in exposure_action.finditer(text):
        window = text[max(0, action_match.start() - 96) : action_match.end() + 128]
        action_clause_start = max(
            text.rfind(separator, 0, action_match.start())
            for separator in ".!?。！？;\n"
        )
        action_clause_end_candidates = [
            position
            for separator in ".!?。！？;\n"
            if (position := text.find(separator, action_match.end())) >= 0
        ]
        action_clause_end = (
            min(action_clause_end_candidates) if action_clause_end_candidates else len(text)
        )
        action_clause = text[action_clause_start + 1 : action_clause_end]
        if (
            risky_substance.search(action_clause)
            and unknown_quantity.search(window)
            and _emergency_span_is_current(
                text,
                action_match.start(),
                action_match.end(),
            )
        ):
            return True
    return False


def _emergency_span_is_current(text: str, start: int, end: int) -> bool:
    """Return whether a matched emergency concept is asserted as current."""

    before = text[max(0, start - 120) : start]
    after = text[end : end + 160]
    if re.search(
        r"(?:기사|뉴스|논문|예시|가상|교육|드라마|문제|퀴즈|만약|가정|혹시)"
        r".{0,56}$",
        before,
    ) or re.search(
        r"\b(?:if|what\s+if|suppose|assuming|hypothetical(?:ly)?|"
        r"in\s+a\s+(?:story|case|scenario)|article|news|paper|example|"
        r"training|fiction|what\s+does|definition\s+of)\b[^.!?\n]{0,80}$",
        before,
    ):
        return False
    if re.match(
        r".{0,28}(?:라?면|경우(?:에는|라면)?|때(?:에는)?|"
        r"(?:을|를)?\s*(?:예방|방지)(?:하|하는|할)?|"
        r"\bthen\b|\bwhat\s+should\b|"
        r"\b(?:prevent(?:ed|ing|ion)?|avoid(?:ed|ing|ance)?)\b)",
        after,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"(?:[\"“‘][^\"”’]{0,120})$",
        before,
    ) and re.match(r"[^\"”’]{0,120}[\"”’]", after):
        return False
    if re.match(
        r".{0,36}(?:(?:라고|라는)\s*(?:표현|문구|인용|문장|말)|"
        r"뜻|정의|번역|의미|\b(?:phrase|term|quote|sentence|headline|"
        r"definition|mean(?:s|ing)?|translation)\b)",
        after,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"(?:\b(?:did|do|does|am|is|are|was|were|have|has|had)\s+not\b\s*|"
        r"\b(?:didn't|don't|doesn't|isn't|aren't|wasn't|weren't|haven't|"
        r"hasn't|hadn't|never)\b\s*|(?:안|전혀)\s*|"
        r"\b(?:denies|without|no)\b[^.!?\n]{0,40})$",
        before,
    ):
        return False
    if re.match(
        r"\s*(?:지(?:는)?\s*(?:않|못)|하지\s*(?:않|못)|아니|없|"
        r"(?:did|does|do|was|were|is|are|has|have)\s+not\b|"
        r"(?:didn't|doesn't|isn't|aren't|hasn't|haven't)\b)",
        after,
        re.IGNORECASE,
    ):
        return False

    remote_history = re.search(
        r"(?:\d+\s*(?:년|개월)\s*전|오래전|과거|예전|어릴\s*때|작년|"
        r"\b(?:years?|months?)\s+ago\b|\blast\s+year\b|\bin\s+childhood\b|"
        r"\bhistory\s+of\b).{0,72}$",
        before,
        re.IGNORECASE,
    )
    recent_history = re.search(
        r"(?:어제|지난번|\byesterday\b|\bpreviously\b).{0,72}$",
        before,
        re.IGNORECASE,
    )
    resolved = re.search(
        r".{0,100}(?:지금|현재|이제|now|currently).{0,48}"
        r"(?:괜찮|회복|끝|멈|사라|없|not\s+(?:happening|bleeding|exposed)|"
        r"no\s+(?:symptoms?|problem)|fine|resolved|stopped)",
        after,
        re.IGNORECASE,
    )
    return not (remote_history or (recent_history and resolved))


def _needs_prior_emergency_context(value: str) -> bool:
    compact = _compact_text(value)
    return any(
        marker in compact
        for marker in ("아직그래", "계속그래", "그대로", "더심해", "악화", "여전히")
    )


def _current_emergency_marker(text: str, marker: str) -> bool:
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return False
        before = text[max(0, index - 32) : index]
        after = text[index + len(marker) : index + len(marker) + 64]
        quoted_or_hypothetical = bool(
            re.search(
                r"(?:기사|뉴스|논문|예시|가상|교육|드라마).{0,16}(?:환자|사람|사례)?$",
                before,
            )
            or re.search(r"(?:만약|가정|혹시).{0,16}$", before)
            or re.search(r"(?:표현|단어)(?:의)?\s*$", before)
            or re.match(r".{0,24}(?:뜻|정의)", after)
        )
        negated = bool(
            re.match(
                r".{0,10}(?:없(?:음|다|어요)?|아니(?:야|에요|다)?|않(?:아|아요|다)?|"
                r"끝났|사라졌|해소됐|괜찮아졌)",
                after,
            )
            or re.search(r"(?:no|denies|without)\s*$", before)
            or re.search(r"(?:no|denies|without)\s+.{0,32}(?:or|and)\s*$", before)
            or _current_clause_marker_negated(text, index, marker)
        )
        past_resolved = bool(
            re.search(r"(?:어제|과거|예전|지난번).{0,20}$", before)
            and re.match(r".{0,16}(?:끝났|사라졌|해소됐|지금은괜찮)", after)
        ) or bool(
            re.search(r"(?:어제|과거|예전|지난번).{0,20}$", before)
            and re.match(r".{0,20}있었지만.{0,12}지금.{0,12}없", after)
        ) or bool(
            re.search(r"(?:\d+년\s*전|오래전|과거|예전).{0,16}$", before)
            and re.match(r".{0,10}(?:병력|과거력|있었)", after)
        )
        past_resolved = past_resolved or _historical_marker_currently_negated(
            text,
            index,
            marker,
        )
        if not (quoted_or_hypothetical or negated or past_resolved):
            return True
        start = index + len(marker)


def _historical_marker_currently_negated(text: str, index: int, marker: str) -> bool:
    before = text[max(0, index - 80) : index]
    if not re.search(r"(?:\d+\s*년\s*전|오래전|과거|예전|지난번)", before):
        return False
    after = text[index + len(marker) : index + len(marker) + 180]
    time_match = re.search(r"(?:지금|현재)", after)
    if time_match is None:
        return False
    current = _compact_text(after[time_match.start() :])
    aliases = _emergency_concept_aliases(marker)
    for alias in aliases:
        alias_index = current.find(alias)
        while alias_index >= 0:
            tail = current[alias_index + len(alias) : alias_index + len(alias) + 32]
            absence = re.search(r"(?:없|않|아니)", tail)
            if absence is not None:
                preceding = tail[: absence.start()]
                if not re.search(r"(?:있|지속|심해|악화)", preceding):
                    return True
            alias_index = current.find(alias, alias_index + len(alias))
    return False


def _current_clause_marker_negated(text: str, index: int, marker: str) -> bool:
    before = text[max(0, index - 72) : index]
    if not re.search(r"(?:지금|현재)(?:은|는)?", before):
        return False
    tail = _compact_text(text[index + len(marker) : index + len(marker) + 120])
    absence = re.search(r"(?:전혀|모두|전부|다)?(?:없|않|아니)", tail)
    if absence is None:
        return False
    preceding = tail[: absence.start()]
    return not bool(re.search(r"(?:있|지속|심해|악화)", preceding))


def _emergency_concept_aliases(marker: str) -> tuple[str, ...]:
    compact_marker = _compact_text(marker)
    chest_aliases = (
        "가슴통증",
        "흉통",
        "가슴압박",
        "가슴조임",
        "가슴짓누름",
        "가슴짓누르",
        "chestpain",
        "chestpressure",
        "chesttightness",
    )
    breathing_aliases = (
        "숨못쉬",
        "숨안쉬",
        "숨이차",
        "숨참",
        "숨가쁨",
        "호흡곤란",
        "호흡안",
        "cantbreathe",
        "shortnessofbreath",
        "dyspnea",
    )
    if compact_marker in chest_aliases:
        return chest_aliases
    if compact_marker in breathing_aliases:
        return breathing_aliases
    return (compact_marker,)


def _noncurrent_quantity_context(text: str, index: int) -> bool:
    before = text[max(0, index - 24) : index]
    after = text[index : index + 48]
    return bool(
        re.search(r"(?:만약|가정|기사|뉴스|예시|과거|예전|\d+년 전).{0,16}$", before)
        or re.search(r"(?:먹지 않았|복용하지 않았|아니(?:야|에요|다))", after)
    )


def _citation_violations(content: str, retrieval: RetrievalResult) -> list[str]:
    citable_items = _citable_evidence_items(retrieval)
    allowed = {str(index) for index, _ in enumerate(citable_items, start=1)}
    normalized = unicodedata.normalize("NFKC", content).casefold()
    decoded = unicodedata.normalize("NFKC", unquote(normalized)).casefold()
    observed = re.findall(r"\[([0-9]+)\]", decoded)
    violations = [f"invalid_citation_id:{value}" for value in observed if value not in allowed]
    if allowed and not any(value in allowed for value in observed):
        violations.append("missing_allowed_citation")
    if re.search(r"(?<![a-z0-9])cite[_\s-]?uid(?![a-z0-9])", decoded):
        violations.append("cite_uid_token_exposed")
    for item in retrieval.items:
        normalized_uid = unicodedata.normalize("NFKC", item.cite_uid).casefold()
        decoded_uid = unicodedata.normalize("NFKC", unquote(normalized_uid)).casefold()
        if (
            normalized_uid in normalized
            or normalized_uid in decoded
            or decoded_uid in decoded
        ):
            violations.append("raw_cite_uid_exposed")
    return violations


def _source_classification(source_tool: str) -> tuple[str | None, str]:
    if source_tool.startswith("openapi_law_"):
        return "law", "regulatory_or_operational_authority"
    if source_tool.startswith("openapi_mfds_"):
        return "mfds", "regulatory_or_operational_authority"
    if source_tool.startswith("openapi_hira_") or source_tool == "hira_updates_search":
        return "hira", "regulatory_or_operational_authority"
    if source_tool.startswith("kcd_"):
        return "kcd", "regulatory_or_operational_authority"
    if source_tool == "adr_retrieve_drug_info":
        return "dailymed", "regulatory_or_operational_authority"
    return None, "unknown"


def _is_truncated_evidence(content: str) -> bool:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(value, Mapping) and value.get("truncated") is True


def _looks_like_tool_protocol(content: str) -> bool:
    normalized = unicodedata.normalize("NFKC", content).casefold()
    markers = (
        "<tool_call>",
        "</tool_call>",
        "<tool_calls>",
        "</tool_calls>",
        "<arg_key>",
        "<arg_value>",
        '"tool_call"',
        '"tool_calls"',
        '"function_call"',
        "generation-reroute-v2",
    )
    if any(marker in normalized for marker in markers):
        return True
    if re.search(
        r"<\s*/?\s*(?:tool|function)(?:[_:-]?calls?)?(?:\s|>|/)",
        normalized,
    ):
        return True
    if re.search(r"\b(?:tool_call|function_call)\s*\(", normalized):
        return True

    stripped = content.strip()
    structured_text = stripped
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        structured_text = fenced.group(1)
    if structured_text.startswith(("{", "[")):
        try:
            structured = json.loads(structured_text)
        except json.JSONDecodeError:
            pass
        else:
            if _structured_tool_protocol(structured):
                return True

    for function_name in _PROTOCOL_FUNCTION_NAMES:
        escaped = re.escape(function_name.casefold())
        if re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", normalized):
            return True
        if re.search(rf"<\s*/?\s*{escaped}(?:\s|>|/)", normalized):
            return True
        if re.search(rf"\b{escaped}\s*\(", normalized):
            return True
        if re.search(
            rf"[\"'](?:name|function_name)[\"']\s*:\s*[\"']{escaped}[\"']",
            normalized,
        ):
            return True
        if re.search(rf"[\"']{escaped}[\"']\s*:\s*\{{", normalized):
            return True
    return False


def _structured_tool_protocol(value: Any) -> bool:
    if isinstance(value, list):
        return any(_structured_tool_protocol(item) for item in value)
    if not isinstance(value, Mapping):
        return False
    normalized = {str(key).casefold(): item for key, item in value.items()}
    if {"tool_calls", "tool_call", "function_call"} & normalized.keys():
        return True
    if {"name", "arguments"} <= normalized.keys():
        return True
    schema_version = normalized.get("schema_version")
    if isinstance(schema_version, str) and schema_version.casefold().startswith(
        "generation-reroute"
    ):
        return True
    name = normalized.get("name")
    if isinstance(name, str) and name.casefold() in _PROTOCOL_FUNCTION_NAMES:
        return True
    function_name = normalized.get("function_name")
    if isinstance(function_name, str):
        return True
    for selector in ("tool", "action"):
        selected = normalized.get(selector)
        if isinstance(selected, str) and selected.casefold() in _PROTOCOL_FUNCTION_NAMES:
            return True
    if any(key in _PROTOCOL_FUNCTION_NAMES for key in normalized):
        return True
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function_keys = {str(key).casefold() for key in function}
        function_name = function.get("name")
        if {"name", "arguments"} <= function_keys:
            return True
        if (
            isinstance(function_name, str)
            and function_name.casefold() in _PROTOCOL_FUNCTION_NAMES
        ):
            return True
    if "tool" in normalized and {"arguments", "query"} & normalized.keys():
        return True
    if normalized.get("type") in {"function", "tool", "tool_call", "function_call"} and (
        "name" in normalized or "function" in normalized
    ):
        return True
    return any(_structured_tool_protocol(item) for item in normalized.values())
