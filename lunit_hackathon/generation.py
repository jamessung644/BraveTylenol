import json
from collections.abc import Mapping, Sequence
from typing import Any

from lunit_hackathon.errors import MalformedUpstreamResponseError
from lunit_hackathon.fast_policy import choose_generation_profile
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


class GenerationEngine:
    """Lets L2 choose one retrieval and always leaves final wording to L2."""

    def __init__(self, l2_client: Any, retrieval_engine: Any) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        conversation = _medical_conversation(messages)
        first: L2Completion = await self._l2.complete(
            messages=conversation,
            tools=[RETRIEVE_RELEVANT_CONTENT_TOOL],
            tool_choice="auto",
        )
        if not first.tool_calls:
            return _final_content(first)

        conversation.append(_assistant_message(first))
        retrieval_call, query = _first_valid_retrieval(first.tool_calls)
        evidence: str | None = None

        for call in first.tool_calls:
            if call is retrieval_call:
                result: RetrievalResult = await self._retrieval.retrieve(query)
                evidence = json.dumps(
                    result.model_dump(mode="json"),
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

        second: L2Completion = await self._l2.complete(messages=conversation)
        if not second.tool_calls:
            return _final_content(second)

        conversation.append(_assistant_message(second))
        for call in second.tool_calls:
            if evidence is None:
                conversation.append(_protocol_error(call, "retrieval unavailable"))
            else:
                conversation.append(_tool_message(call.id, evidence))
        conversation.append(
            {
                "role": "system",
                "content": (
                    "Provide the final user-facing medical answer now. "
                    "Do not call tools or describe tool use."
                ),
            }
        )
        final: L2Completion = await self._l2.complete(messages=conversation)
        return _final_content(final)

    async def direct_answer(self, messages: Sequence[ChatMessage]) -> str:
        """One-call HealthBench path with a local latency policy."""

        profile = choose_generation_profile(messages)
        completion: L2Completion = await self._l2.complete(
            messages=_medical_conversation(messages, system_prompt=profile.system_prompt),
            max_tokens=profile.max_tokens,
        )
        return _final_content(completion)


def _medical_conversation(
    messages: Sequence[ChatMessage],
    *,
    system_prompt: str = MEDICAL_GENERATION_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    conversation = [
        ChatMessage(
            role="system",
            content=system_prompt,
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
