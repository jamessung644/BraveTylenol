import json
from collections.abc import Mapping, Sequence
from typing import Any

from harness.errors import MalformedUpstreamResponseError
from harness.prompts import GENERATION_SYSTEM_PROMPT, RETRIEVE_RELEVANT_CONTENT_TOOL
from harness.schemas import ChatMessage, L2Completion, RetrievalResult, ToolCall


class GenerationEngine:
    """Lets L2 choose one evidence retrieval, then returns L2's final text verbatim."""

    def __init__(self, l2_client: Any, retrieval_engine: Any) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        conversation = [_message(ChatMessage(role="system", content=GENERATION_SYSTEM_PROMPT))]
        conversation.extend(_message(message) for message in messages)
        first: L2Completion = await self._l2.complete(
            messages=list(conversation),
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
                evidence = _evidence(result)
                conversation.append(_tool_message(call.id, evidence))
            elif call.function.name != "retrieve_relevant_content":
                conversation.append(_protocol_error(call, "unexpected tool"))
            elif _valid_query(call) is None:
                conversation.append(_protocol_error(call, "invalid retrieval request"))
            else:
                conversation.append(_protocol_error(call, "retrieval already used"))

        second: L2Completion = await self._l2.complete(messages=list(conversation))
        if not second.tool_calls:
            return _final_content(second)

        conversation.append(_assistant_message(second))
        for call in second.tool_calls:
            if evidence is None:
                conversation.append(_protocol_error(call, "retrieval unavailable"))
            else:
                conversation.append(_tool_message(call.id, evidence))
        conversation.append(_message(ChatMessage(
            role="system",
            content="Provide the final user-facing text answer now. Do not call tools or describe tool use.",
        )))
        final: L2Completion = await self._l2.complete(messages=list(conversation))
        return _final_content(final)


def _message(message: ChatMessage) -> dict[str, Any]:
    return message.model_dump(exclude_none=True)


def _assistant_message(completion: L2Completion) -> dict[str, Any]:
    return _message(ChatMessage(role="assistant", content=completion.content, tool_calls=completion.tool_calls))


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    return _message(ChatMessage(role="tool", tool_call_id=tool_call_id, content=content))


def _protocol_error(call: ToolCall, error: str) -> dict[str, Any]:
    return _tool_message(call.id, json.dumps({"error": error}, ensure_ascii=False, separators=(",", ":")))


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
    return query if isinstance(query, str) and query.strip() else None


def _evidence(result: RetrievalResult) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _final_content(completion: L2Completion) -> str:
    if completion.content is None or not completion.content.strip():
        raise MalformedUpstreamResponseError("Upstream model returned no final text")
    return completion.content
