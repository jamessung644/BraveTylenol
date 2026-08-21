"""L2-only direct and evidence-grounded medical answer generation."""

import re
from collections.abc import Sequence
from typing import Any

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import MalformedUpstreamResponseError, UpstreamTimeoutError
from lunit_hackathon.evidence import bounded_evidence_payload
from lunit_hackathon.prompts import (
    DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT,
    MEDICAL_GENERATION_SYSTEM_PROMPT,
)
from lunit_hackathon.schemas import ChatMessage, L2Completion, RetrievalResult

_EXPLICIT_CITATION_MARKER = re.compile(
    r"(?P<bracketed>\[\s*)?(?:[\"']?(?:cite_uid|citation)[\"']?)\s*[:=]\s*",
    re.IGNORECASE,
)
_PROTOCOL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "<arg_value>",
    "function_call",
)


class GenerationEngine:
    """Produces final medical text with L2; retrieval is supplied by the caller."""

    def __init__(self, l2_client: Any, settings: Settings) -> None:
        self._l2 = l2_client
        self._settings = settings

    async def direct_answer(
        self,
        messages: Sequence[ChatMessage],
        deadline: RequestDeadline,
    ) -> str:
        conversation = _medical_conversation(messages, DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT)
        return await self._complete_with_one_correction(
            conversation=conversation,
            retrieval=None,
            deadline=deadline,
            max_tokens=self._settings.max_completion_tokens,
        )

    async def grounded_answer(
        self,
        messages: Sequence[ChatMessage],
        retrieval: RetrievalResult,
        deadline: RequestDeadline,
    ) -> str:
        bounded_retrieval, evidence_payload = bounded_evidence_payload(
            retrieval,
            self._settings.max_evidence_chars,
        )
        conversation = _medical_conversation(messages, MEDICAL_GENERATION_SYSTEM_PROMPT)
        conversation.append({"role": "system", "content": evidence_payload})
        conversation.append(
            {"role": "system", "content": _grounding_instruction(bounded_retrieval)}
        )
        return await self._complete_with_one_correction(
            conversation=conversation,
            retrieval=bounded_retrieval,
            deadline=deadline,
            max_tokens=min(4_096, self._settings.max_completion_tokens),
        )

    async def _complete_with_one_correction(
        self,
        *,
        conversation: list[dict[str, Any]],
        retrieval: RetrievalResult | None,
        deadline: RequestDeadline,
        max_tokens: int,
    ) -> str:
        first = await self._complete(conversation, deadline, max_tokens)
        issue = _answer_issue(first.content, retrieval)
        if issue is None:
            return first.content.strip()
        if not deadline.can_spend(10.0):
            raise MalformedUpstreamResponseError("L2 final answer was malformed")

        correction_conversation = [
            *conversation,
            _assistant_message(first),
            {
                "role": "system",
                "content": _correction_instruction(issue, retrieval),
            },
        ]
        corrected = await self._complete(correction_conversation, deadline, max_tokens)
        if _answer_issue(corrected.content, retrieval) is not None:
            raise MalformedUpstreamResponseError("L2 correction was malformed")
        return corrected.content.strip()

    async def _complete(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        deadline: RequestDeadline,
        max_tokens: int,
    ) -> L2Completion:
        timeout = deadline.stage_timeout(self._settings.generation_timeout_seconds)
        if timeout <= 0:
            raise UpstreamTimeoutError("generation deadline exhausted")
        return await self._l2.complete(
            messages=messages,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            reasoning_effort=self._settings.generation_reasoning_effort,
        )


def _medical_conversation(
    messages: Sequence[ChatMessage], system_prompt: str
) -> list[dict[str, Any]]:
    return [
        ChatMessage(role="system", content=system_prompt).model_dump(exclude_none=True),
        *(message.model_dump(exclude_none=True) for message in messages),
    ]


def _grounding_instruction(retrieval: RetrievalResult) -> str:
    cite_uids = _actual_cite_uids(retrieval)
    if not cite_uids:
        return (
            "No selected evidence is available. Frame uncertainty explicitly and "
            "provide cautious general guidance only. You must not claim retrieval "
            "succeeded, that an official source was found, or that source-specific "
            "facts were verified. Do not expose internal protocol."
        )
    return (
        "Write a concise Korean-first user-facing answer. Use selected evidence only for "
        "source-dependent claims and preserve at least one relevant exact cite_uid "
        f"beside those claims: {', '.join(cite_uids)}. Never invent cite_uids, URLs, "
        "legal status, dosage, coverage, or unsupported clinical claims. Treat FAERS "
        "rows as observational safety signals that do not establish causality. When "
        "evidence metadata differs by jurisdiction or effective date, distinguish "
        "those facts explicitly rather than merging recommendations."
    )


def _correction_instruction(issue: str, retrieval: RetrievalResult | None) -> str:
    expected = ", ".join(_actual_cite_uids(retrieval)) if retrieval else "none"
    return (
        "Return only the complete corrected user-facing medical answer as plain text. "
        f"Remove internal protocol and unsupported citation identifiers. The prior answer "
        f"failed because: {issue}. The only allowed cite_uid values are: {expected}. "
        "Preserve a relevant allowed cite_uid when selected evidence exists. Do not call "
        "tools or describe this correction."
    )


def _actual_cite_uids(retrieval: RetrievalResult | None) -> tuple[str, ...]:
    if retrieval is None:
        return ()
    return tuple(
        dict.fromkeys(item.cite_uid.strip() for item in retrieval.items if item.cite_uid.strip())
    )


def _answer_issue(content: str | None, retrieval: RetrievalResult | None) -> str | None:
    if content is None or not content.strip():
        return "the answer was blank"
    if _looks_like_tool_protocol(content):
        return "the answer leaked tool protocol"
    actual = set(_actual_cite_uids(retrieval))
    mentioned = {cite_uid for cite_uid in actual if _contains_exact_cite_uid(content, cite_uid)}
    explicit_values = _explicit_citation_values(content, actual)
    invented = explicit_values - actual
    if invented:
        return f"it included unsupported cite_uid values: {', '.join(sorted(invented))}"
    if actual and not (mentioned & actual):
        return "it omitted all selected cite_uid values"
    return None


def _contains_exact_cite_uid(content: str, cite_uid: str) -> bool:
    boundary = (
        r"[\w./:#-]"
        if any(character.isalpha() and not character.isascii() for character in cite_uid)
        else r"[A-Za-z0-9_./:#-]"
    )
    pattern = rf"(?<!{boundary}){re.escape(cite_uid)}(?!{boundary})"
    return re.search(pattern, content) is not None


def _explicit_citation_values(content: str, actual: set[str]) -> set[str]:
    values: set[str] = set()
    allowlisted = tuple(sorted(actual, key=len, reverse=True))
    for marker in _EXPLICIT_CITATION_MARKER.finditer(content):
        remainder = content[marker.end() :]
        bracketed = marker.group("bracketed") is not None
        exact = _exact_marked_uid(remainder, bracketed, allowlisted)
        if exact is not None:
            values.add(exact)
            continue
        payload = _marker_payload(remainder, bracketed)
        if payload:
            values.add(payload)
    return values


def _exact_marked_uid(
    remainder: str,
    bracketed: bool,
    allowlisted: Sequence[str],
) -> str | None:
    if remainder[:1] in {"`", '"', "'"}:
        quote = remainder[0]
        end = remainder.find(quote, 1)
        if end > 0:
            quoted = remainder[1:end]
            return quoted if quoted in allowlisted else None

    for cite_uid in allowlisted:
        if not remainder.startswith(cite_uid):
            continue
        tail = remainder[len(cite_uid) :]
        if bracketed and tail.startswith("]"):
            return cite_uid
        if not bracketed and (not tail or tail.startswith(("\n", "\r"))):
            return cite_uid
    return None


def _marker_payload(remainder: str, bracketed: bool) -> str:
    if remainder[:1] in {"`", '"', "'"}:
        quote = remainder[0]
        end = remainder.find(quote, 1)
        return remainder[1:end].strip() if end > 0 else remainder[1:].strip()

    line = remainder.splitlines()[0] if remainder else ""
    if bracketed and "]" in line:
        return line[: line.rfind("]")].strip()
    return line.strip()


def _assistant_message(completion: L2Completion) -> dict[str, Any]:
    return ChatMessage(
        role="assistant", content=completion.content, tool_calls=completion.tool_calls or None
    ).model_dump(exclude_none=True)


def _looks_like_tool_protocol(content: str) -> bool:
    normalized = content.casefold()
    return any(marker in normalized for marker in _PROTOCOL_MARKERS)
