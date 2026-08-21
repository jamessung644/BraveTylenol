import json
from collections.abc import Mapping, Sequence
from typing import Any

from harness.errors import MalformedUpstreamResponseError
from harness.prompts import (
    EMERGENCY_RESPONSE_TEXT,
    GENERATION_SYSTEM_PROMPT,
    RETRIEVE_RELEVANT_CONTENT_TOOL,
)
from harness.schemas import ChatMessage, L2Completion, RetrievalResult, ToolCall


class GenerationEngine:
    """Lets L2 choose one evidence retrieval, then returns L2's final text verbatim."""

    def __init__(self, l2_client: Any, retrieval_engine: Any) -> None:
        self._l2 = l2_client
        self._retrieval = retrieval_engine

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        conversation = [_message(ChatMessage(role="system", content=GENERATION_SYSTEM_PROMPT))]
        conversation.extend(_message(message) for message in messages)
        if _requires_airway_emergency_response(messages):
            return EMERGENCY_RESPONSE_TEXT
        tool_choice: str | dict[str, Any] = "auto"
        if _requires_retrieval(messages):
            tool_choice = {"type": "function", "function": {"name": "retrieve_relevant_content"}}
        first: L2Completion = await self._l2.complete(
            messages=list(conversation),
            tools=[RETRIEVE_RELEVANT_CONTENT_TOOL],
            tool_choice=tool_choice,
        )
        if not first.tool_calls:
            return _final_content(first)

        conversation.append(_assistant_message(first))
        retrieval_call, query = _first_valid_retrieval(first.tool_calls)
        evidence: str | None = None
        retrieval_result: RetrievalResult | None = None
        for call in first.tool_calls:
            if call is retrieval_call:
                retrieval_result = await self._retrieval.retrieve(query)
                evidence = _evidence(retrieval_result)
                conversation.append(_tool_message(call.id, evidence))
            elif call.function.name != "retrieve_relevant_content":
                conversation.append(_protocol_error(call, "unexpected tool"))
            elif _valid_query(call) is None:
                conversation.append(_protocol_error(call, "invalid retrieval request"))
            else:
                conversation.append(_protocol_error(call, "retrieval already used"))

        if retrieval_result is not None and retrieval_result.items:
            if _is_kcd_request(messages):
                rendered = _render_kcd_evidence(
                    retrieval_result,
                    explain_relationship=_asks_kcd_relationship(messages),
                )
                if rendered is not None:
                    return rendered
            conversation.append(_citation_instruction(retrieval_result))

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


def _requires_retrieval(messages: Sequence[ChatMessage]) -> bool:
    user_text = " ".join(message.content or "" for message in messages if message.role == "user").casefold()
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
    return any(marker in user_text for marker in markers)


def _requires_airway_emergency_response(messages: Sequence[ChatMessage]) -> bool:
    user_text = " ".join(message.content or "" for message in messages if message.role == "user").casefold()
    breathing_markers = (
        "숨쉬기 어렵",
        "숨을 못",
        "숨이 안",
        "호흡 곤란",
        "호흡곤란",
    )
    swallowing_or_swelling_markers = (
        "침도 삼키",
        "삼키기 힘",
        "삼킬 수 없",
        "목이 붓",
        "목이 심하게 붓",
    )
    return any(marker in user_text for marker in breathing_markers) and any(
        marker in user_text for marker in swallowing_or_swelling_markers
    )


def _is_kcd_request(messages: Sequence[ChatMessage]) -> bool:
    return any(
        "kcd" in (message.content or "").casefold()
        for message in messages
        if message.role == "user"
    )


def _asks_kcd_relationship(messages: Sequence[ChatMessage]) -> bool:
    user_text = " ".join(message.content or "" for message in messages if message.role == "user").casefold()
    return any(marker in user_text for marker in ("포함", "배제", "관계", "상위", "하위", "별도 코드", "병기", "청구"))


def _render_kcd_evidence(result: RetrievalResult, *, explain_relationship: bool) -> str | None:
    records: dict[str, dict[str, Any]] = {}
    for item in result.items:
        if not item.source_tool.startswith("kcd_"):
            continue
        try:
            payload = json.loads(item.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
            continue
        record = dict(payload)
        record["cite_uid"] = item.cite_uid
        records.setdefault(payload["code"], record)
    if not records:
        return None

    lines = [
        "조회된 KCD 공식 항목은 다음과 같습니다.",
        "",
        "| 코드 | 한글 공식명 | 영문 공식명 | 판본 | 근거 식별자 |",
        "|---|---|---|---|---|",
    ]
    for code, record in records.items():
        lines.append(
            "| {code} | {kor} | {eng} | {revision} | `{cite_uid}` |".format(
                code=_markdown_cell(code),
                kor=_markdown_cell(_headword(record.get("kor"))),
                eng=_markdown_cell(_headword(record.get("eng"))),
                revision=_markdown_cell(record.get("revision")),
                cite_uid=_markdown_cell(record.get("cite_uid")),
            )
        )

    details: list[str] = []
    for code, record in records.items():
        subheadings = _subheadings(record.get("kor"))
        if subheadings:
            details.append(f"- {code} 세부표시: {', '.join(subheadings)}")
    if details:
        lines.extend(["", "공식 항목에 기재된 세부표시:", "", *details])

    source_versions = sorted({
        value for record in records.values()
        if isinstance((value := record.get("source_version")), str) and value
    })
    if explain_relationship:
        lines.extend([
            "",
            (
                "관계 해석의 한계: 이 조회 결과는 위 코드들을 각각 별도 항목으로 반환하지만, 코드 간 "
                "부모·자식 관계나 포함·배제, 병기, 청구 규칙을 나타내는 필드는 제공하지 않습니다. 따라서 "
                "이 근거만으로 한 코드가 다른 코드를 포함하거나 배제한다고 단정할 수 없습니다."
            ),
        ])
    if source_versions:
        lines.extend(["", f"출처 버전: {', '.join(source_versions)}"])
    return "\n".join(lines)


def _headword(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    headword = value.get("headword")
    return headword if isinstance(headword, str) else ""


def _subheadings(value: Any) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("subheadings"), list):
        return []
    return [item for item in value["subheadings"] if isinstance(item, str)]


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _evidence(result: RetrievalResult) -> str:
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _citation_instruction(result: RetrievalResult) -> dict[str, Any]:
    cite_uids = [item.cite_uid for item in result.items]
    return _message(ChatMessage(
        role="system",
        content=(
            "Write a concise final user-facing answer using only claims directly supported by the "
            "selected evidence. Include every cite_uid below verbatim in a Sources or 근거 section: "
            f"{json.dumps(cite_uids, ensure_ascii=False)}. Do not rename, expand, or fabricate "
            "identifiers. Do not add clinical diagnostic criteria, simultaneous-coding advice, "
            "billing rules, or inclusion/exclusion relationships that the evidence does not state. "
            "If the requested relationship is not explicit in the evidence, say so."
        ),
    ))


def _final_content(completion: L2Completion) -> str:
    if completion.content is None or not completion.content.strip():
        raise MalformedUpstreamResponseError("Upstream model returned no final text")
    return completion.content
