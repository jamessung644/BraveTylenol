import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from lunit_hackathon.errors import MalformedUpstreamResponseError
from lunit_hackathon.prompts import (
    MEDICAL_GENERATION_SYSTEM_PROMPT,
    RETRIEVE_RELEVANT_CONTENT_TOOL,
)
from lunit_hackathon.schemas import (
    ChatMessage,
    L2Completion,
    RetrievalResult,
    ToolCall,
)

logger = logging.getLogger(__name__)

_FINAL_ANSWER_TOOL_NAME = "submit_final_answer"


class GenerationEngine:
    """Lets L2 choose one retrieval and always leaves final wording to L2."""

    def __init__(self, l2_client: Any, retrieval_engine: Any) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        conversation = _medical_conversation(messages)
        tool_choice: str | dict[str, Any] = "auto"
        if _requires_retrieval(messages):
            tool_choice = {
                "type": "function",
                "function": {"name": "retrieve_relevant_content"},
            }
        first: L2Completion = await self._l2.complete(
            messages=conversation,
            tools=[RETRIEVE_RELEVANT_CONTENT_TOOL],
            tool_choice=tool_choice,
        )
        if not first.tool_calls:
            if first.content is not None and _looks_like_tool_protocol(first.content):
                conversation.append(_assistant_message(first))
                conversation.append(
                    {
                        "role": "system",
                        "content": (
                            "The prior text exposed internal tool protocol without making a valid "
                            "tool call. Submit a complete user-facing answer without tool syntax. "
                            "Do not claim that source-specific evidence was retrieved."
                        ),
                    }
                )
                fallback = await self._request_final_submission(conversation, None)
                return _submitted_content(fallback) or _final_content(fallback)
            return _final_content(first)

        conversation.append(_assistant_message(first))
        retrieval_call, query = _first_valid_retrieval(first.tool_calls)
        evidence: str | None = None
        retrieval_result: RetrievalResult | None = None

        for call in first.tool_calls:
            if call is retrieval_call:
                retrieval_result = await self._retrieval.retrieve(query)
                evidence = json.dumps(
                    retrieval_result.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                conversation.append(_tool_message(call.id, evidence))
            elif call.function.name != "retrieve_relevant_content":
                conversation.append(_protocol_error(call, "unexpected tool"))
            elif _valid_query(call) is None:
                conversation.append(_protocol_error(call, "invalid retrieval request"))
            else:
                conversation.append(_protocol_error(call, "retrieval already used"))

        if retrieval_result is not None:
            conversation.append(
                {
                    "role": "system",
                    "content": _grounding_instruction(retrieval_result),
                }
            )

        second = await self._request_final_submission(conversation, retrieval_result)
        submitted = _submitted_content(second)
        if submitted is not None:
            return await self._final_with_required_citations(
                conversation,
                L2Completion(content=submitted),
                retrieval_result,
            )
        if not second.tool_calls:
            return await self._final_with_required_citations(
                conversation,
                second,
                retrieval_result,
            )

        conversation.append(_assistant_message(second))
        for call in second.tool_calls:
            if call.function.name == _FINAL_ANSWER_TOOL_NAME:
                conversation.append(_protocol_error(call, "invalid final answer submission"))
            elif retrieval_result is None:
                conversation.append(_protocol_error(call, "retrieval unavailable"))
            else:
                conversation.append(_protocol_error(call, "unexpected tool"))
        conversation.append(
            {
                "role": "system",
                "content": (
                    "Provide the final user-facing medical answer now. "
                    "Do not call tools or describe tool use."
                ),
            }
        )
        final = await self._request_final_submission(conversation, retrieval_result)
        submitted = _submitted_content(final)
        if submitted is not None:
            final = L2Completion(content=submitted)
        return await self._final_with_required_citations(
            conversation,
            final,
            retrieval_result,
        )

    async def direct_answer(self, messages: Sequence[ChatMessage]) -> str:
        """L2-only safe fallback used when optional retrieval is unavailable."""

        conversation = _medical_conversation(messages)
        completion: L2Completion = await self._l2.complete(messages=conversation)
        content = _final_content(completion)
        if not _looks_like_tool_protocol(content):
            return content
        conversation.append(_assistant_message(completion))
        conversation.append(
            {
                "role": "system",
                "content": (
                    "Rewrite the complete user-facing answer without internal tool-call syntax. "
                    "No retrieval tools are available."
                ),
            }
        )
        fallback = await self._request_final_submission(conversation, None)
        return _submitted_content(fallback) or _final_content(fallback)

    async def _request_final_submission(
        self,
        conversation: list[dict[str, Any]],
        retrieval: RetrievalResult | None,
    ) -> L2Completion:
        return await self._l2.complete(
            messages=conversation,
            tools=[_final_answer_tool(retrieval)],
            tool_choice={
                "type": "function",
                "function": {"name": _FINAL_ANSWER_TOOL_NAME},
            },
        )

    async def _final_with_required_citations(
        self,
        conversation: list[dict[str, Any]],
        completion: L2Completion,
        retrieval: RetrievalResult | None,
    ) -> str:
        content = _final_content(completion)
        if _looks_like_tool_protocol(content):
            conversation.append(_assistant_message(completion))
            conversation.append(
                {
                    "role": "system",
                    "content": (
                        "Retrieval is finished and no tools are available. Rewrite the complete "
                        "user-facing answer without tool-call syntax or internal protocol. If "
                        "evidence was not found, say that clearly and give cautious guidance."
                    ),
                }
            )
            completion = await self._request_final_submission(conversation, retrieval)
            content = _submitted_content(completion) or _final_content(completion)

        missing = _missing_cite_uids(content, retrieval)
        if not missing:
            return content

        conversation.append(_assistant_message(completion))
        conversation.append(
            {
                "role": "system",
                "content": (
                    "Rewrite the complete final answer now. Keep supported content and remove "
                    "invented claims. "
                    "Include these exact cite_uid strings verbatim beside the claims "
                    f"they support: {', '.join(missing)}. Do not replace them with numbered "
                    "citations, call tools, or describe this correction."
                ),
            }
        )
        corrected = await self._request_final_submission(conversation, retrieval)
        corrected_content = _submitted_content(corrected) or _final_content(corrected)
        remaining = _missing_cite_uids(corrected_content, retrieval)
        if remaining:
            logger.warning("l2_citation_omitted citation_count=%d", len(remaining))
        return corrected_content


def _medical_conversation(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    conversation = [
        ChatMessage(
            role="system",
            content=MEDICAL_GENERATION_SYSTEM_PROMPT,
        ).model_dump(exclude_none=True)
    ]
    conversation.extend(message.model_dump(exclude_none=True) for message in messages)
    return conversation


def _assistant_message(completion: L2Completion) -> dict[str, Any]:
    return ChatMessage(
        role="assistant",
        content=completion.content,
        tool_calls=completion.tool_calls,
    ).model_dump(exclude_none=True)


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return ChatMessage(
        role="tool",
        tool_call_id=tool_call_id,
        content=content,
    ).model_dump(exclude_none=True)


def _protocol_error(call: ToolCall, error: str) -> dict[str, Any]:
    content = json.dumps({"error": error}, ensure_ascii=False, separators=(",", ":"))
    return _tool_message(call.id, content)


def _first_valid_retrieval(calls: Sequence[ToolCall]) -> tuple[ToolCall | None, str]:
    for call in calls:
        query = _valid_query(call)
        if call.function.name == "retrieve_relevant_content" and query is not None:
            return call, query
    return None, ""


def _valid_query(call: ToolCall) -> str | None:
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(arguments, Mapping) or set(arguments) != {"query"}:
        return None
    query = arguments["query"]
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query if query and len(query) <= 4_000 else None


def _final_content(completion: L2Completion) -> str:
    if completion.content is None or not completion.content.strip():
        raise MalformedUpstreamResponseError("L2 returned no final text")
    return completion.content


def _requires_retrieval(messages: Sequence[ChatMessage]) -> bool:
    text = "\n".join(message.content or "" for message in messages).casefold()
    markers = (
        "cite_uid",
        "kcd",
        "공식 근거",
        "근거 식별자",
        "출처",
        "식약처",
        "심평원",
        "법령",
        "의료법",
        "조문",
        "허가 정보",
        "허가사항",
        "최신",
        "논문",
        "연구 결과",
        "진료지침",
        "가이드라인",
    )
    return any(marker in text for marker in markers)


def _grounding_instruction(result: RetrievalResult) -> str:
    cite_uids = ", ".join(item.cite_uid for item in result.items)
    if not result.items:
        return (
            "Retrieval is finished and returned no citable evidence. No tools are available in the "
            "next step. Do not output tool-call syntax or claim that an official source was found. "
            "Answer cautiously from reliable general medical knowledge, clearly state that the "
            "requested source-specific evidence was not established, and suggest a more precise "
            "product name or other focused detail when it would enable a better lookup."
        )
    return (
        "Write the final answer using only the supplied evidence for source-dependent claims. "
        f"Include each relevant cite_uid verbatim in the answer: {cite_uids}. "
        "Never invent a cite_uid or unsupported diagnosis, classification hierarchy, inclusion or "
        "exclusion rule, coding/billing instruction, approval, contraindication, or "
        "recommendation. "
        "For official classifications, distinguish an official name from related, inclusion, and "
        "exclusion entries. Separate code listings alone do not establish an inclusion or "
        "exclusion relationship. Do not say a source explicitly states a relationship unless its "
        "content contains that rule; otherwise label the conclusion as an inference or say that "
        "the evidence does not establish it. "
        "State clearly when the retrieved evidence is partial or insufficient."
    )


def _missing_cite_uids(
    content: str,
    retrieval: RetrievalResult | None,
) -> list[str]:
    if retrieval is None:
        return []
    cite_uids = [item.cite_uid for item in retrieval.items]
    if any(cite_uid in content for cite_uid in cite_uids):
        return []
    return cite_uids


def _looks_like_tool_protocol(content: str) -> bool:
    normalized = content.casefold()
    markers = (
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "<arg_value>",
    )
    return any(marker in normalized for marker in markers)


def _final_answer_tool(retrieval: RetrievalResult | None) -> dict[str, Any]:
    cite_uids = [item.cite_uid for item in retrieval.items] if retrieval is not None else []
    citation_instruction = ""
    if cite_uids:
        citation_instruction = (
            " Include at least one of these exact cite_uid strings verbatim beside the claim it "
            f"supports: {', '.join(cite_uids)}."
        )
    return {
        "type": "function",
        "function": {
            "name": _FINAL_ANSWER_TOOL_NAME,
            "description": (
                "Submit the complete user-facing answer authored by Lunit L2."
                f"{citation_instruction} Do not include internal tool protocol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            f"The complete final user-facing answer.{citation_instruction}"
                        ),
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def _submitted_content(completion: L2Completion) -> str | None:
    for call in completion.tool_calls:
        if call.function.name != _FINAL_ANSWER_TOOL_NAME:
            continue
        try:
            arguments = json.loads(call.function.arguments)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(arguments, Mapping) or set(arguments) != {"answer"}:
            continue
        answer = arguments["answer"]
        if isinstance(answer, str) and answer.strip():
            return answer
    return None
