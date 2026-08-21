import json
from contextlib import asynccontextmanager

from lunit_hackathon.config import Settings
from lunit_hackathon.retrieval import RetrievalEngine
from lunit_hackathon.schemas import L2Completion, MCPCallResult, MCPTool, ToolCall


def call(call_id: str, name: str, arguments: dict | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, function={"name": name, "arguments": raw})


class ScriptedL2:
    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


class FakeMCP:
    def __init__(self, tools, results):
        self.tools = tools
        self.results = results
        self.calls = []

    @asynccontextmanager
    async def connect(self):
        yield self

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        result = self.results[name]
        if isinstance(result, Exception):
            raise result
        return result


def settings(monkeypatch, **updates):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    return Settings(_env_file=None).model_copy(update=updates)


def lookup_tool(name="lookup"):
    return MCPTool(
        name=name,
        description="authoritative lookup",
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    )


async def test_retrieval_discovers_calls_and_selects_citation(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup-1", "lookup", {"code": "I10"})]),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "kcd:I10", "relevance_score": 0.9}],
                            "note": "공식 분류 근거",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [lookup_tool()],
        {"lookup": MCPCallResult(content='{"cite_uid":"kcd:I10","name":"본태성 고혈압"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("I10은 어떤 질병인가?")

    assert result.status == "sufficient"
    assert result.items[0].cite_uid == "kcd:I10"
    assert "본태성 고혈압" in result.items[0].content
    assert mcp.calls == [("lookup", {"code": "I10"})]
    names = [item["function"]["name"] for item in l2.calls[0]["tools"]]
    assert names == ["lookup", "finalize_retrieval"]


async def test_retrieval_rejects_invalid_arguments_without_execution(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("bad", "lookup", {"wrong": "I10"})]),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {"status": "no_evidence", "items": [], "note": "none"},
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP([lookup_tool()], {})

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert result.status == "no_evidence"
    assert mcp.calls == []
    tool_errors = [message for message in l2.calls[1]["messages"] if message.role == "tool"]
    assert json.loads(tool_errors[0].content)["error"] == ("arguments do not match tool schema")


async def test_retrieval_enforces_hard_tool_budget(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call(str(index), "lookup", {"code": str(index)}) for index in range(3)]
            ),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "one", "relevance_score": 0.8}],
                            "note": "budget reached",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [lookup_tool()],
        {"lookup": MCPCallResult(content='{"cite_uid":"one","text":"evidence"}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=2),
    ).retrieve("query")

    assert len(mcp.calls) == 2
    assert result.status == "sufficient"
    assert [item.cite_uid for item in result.items] == ["one"]
    assert result.note == "budget reached"
    assert [tool["function"]["name"] for tool in l2.calls[1]["tools"]] == ["finalize_retrieval"]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == "finalize_retrieval"


async def test_retrieval_rejects_duplicate_calls_and_uses_previous_result(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("first", "lookup", {"code": "I10"})]),
            L2Completion(tool_calls=[call("duplicate", "lookup", {"code": "I10"})]),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "kcd:I10", "relevance_score": 1}],
                            "note": "done",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [lookup_tool()],
        {"lookup": MCPCallResult(content='{"cite_uid":"kcd:I10"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert mcp.calls == [("lookup", {"code": "I10"})]
    assert result.items[0].cite_uid == "kcd:I10"
    errors = [message for message in l2.calls[2]["messages"] if message.role == "tool"]
    assert "duplicate tool call" in json.loads(errors[-1].content)["error"]


async def test_retrieval_routes_kcd_tools_and_enforces_requested_revision(monkeypatch):
    kcd_tool = MCPTool(
        name="kcd_get_name",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "revision": {"type": "string"},
            },
            "required": ["code", "revision"],
            "additionalProperties": False,
        },
    )
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call(
                        "lookup",
                        "kcd_get_name",
                        {"code": "I10", "revision": "KCD-9"},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "kcd8:I10", "relevance_score": 1}],
                            "note": "done",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [kcd_tool, lookup_tool("unrelated_tool")],
        {"kcd_get_name": MCPCallResult(content='{"cite_uid":"kcd8:I10"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("KCD-8 I10의 공식 명칭")

    assert mcp.calls == [("kcd_get_name", {"code": "I10", "revision": "KCD-8"})]
    names = [tool["function"]["name"] for tool in l2.calls[0]["tools"]]
    assert names == ["kcd_get_name", "finalize_retrieval"]
    assert result.items[0].cite_uid == "kcd8:I10"


async def test_retrieval_uses_citation_specific_fragment(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", "lookup", {"code": "I10"})]),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "one", "relevance_score": 1}],
                            "note": "done",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [lookup_tool()],
        {
            "lookup": MCPCallResult(
                content='{"results":[{"cite_uid":"one"},{"cite_uid":"two"}]}',
                cite_uids=["one", "two"],
                citation_contents={
                    "one": '{"cite_uid":"one","text":"selected"}',
                    "two": '{"cite_uid":"two","text":"other"}',
                },
            )
        },
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert json.loads(result.items[0].content) == {
        "cite_uid": "one",
        "text": "selected",
    }


async def test_retrieval_routes_mfds_efficacy_and_precautions_to_one_tool(monkeypatch):
    indication_tool = MCPTool(
        name="openapi_mfds_get_drug_indication",
        input_schema={
            "type": "object",
            "properties": {
                "drug_name": {"type": "string"},
                "num_rows": {"type": "integer"},
            },
            "required": ["drug_name"],
            "additionalProperties": False,
        },
    )
    unrelated_tools = [
        lookup_tool(name)
        for name in (
            "adr_retrieve_drug_info",
            "openapi_hira_get_drug_price",
            "openapi_mfds_check_drug_permission",
            "openapi_mfds_find_drugs_by_ingredient",
        )
    ]
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call(
                        "lookup",
                        "openapi_mfds_get_drug_indication",
                        {"drug_name": "아세트아미노펜"},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "mfds:1", "relevance_score": 1}],
                            "note": "done",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [indication_tool, *unrelated_tools],
        {"openapi_mfds_get_drug_indication": MCPCallResult(content='{"cite_uid":"mfds:1"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "식약처 공식 근거로 약 효능과 주의사항을 알려줘"
    )

    names = [tool["function"]["name"] for tool in l2.calls[0]["tools"]]
    assert names == ["openapi_mfds_get_drug_indication", "finalize_retrieval"]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == "finalize_retrieval"
    assert mcp.calls == [
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "아세트아미노펜", "num_rows": 3},
        )
    ]
    assert result.items[0].cite_uid == "mfds:1"


async def test_retrieval_clamps_mfds_product_rows(monkeypatch):
    indication_tool = MCPTool(
        name="openapi_mfds_get_drug_indication",
        input_schema={
            "type": "object",
            "properties": {
                "drug_name": {"type": "string"},
                "num_rows": {"type": "integer", "minimum": 1},
            },
            "required": ["drug_name"],
            "additionalProperties": False,
        },
    )
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call(
                        "lookup",
                        "openapi_mfds_get_drug_indication",
                        {"drug_name": "타이레놀", "num_rows": 20},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    call(
                        "finish",
                        "finalize_retrieval",
                        {
                            "status": "sufficient",
                            "items": [{"cite_uid": "mfds:1", "relevance_score": 1}],
                            "note": "done",
                        },
                    )
                ]
            ),
        ]
    )
    mcp = FakeMCP(
        [indication_tool, lookup_tool("unrelated")],
        {"openapi_mfds_get_drug_indication": MCPCallResult(content='{"cite_uid":"mfds:1"}')},
    )

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("타이레놀 효능")

    assert mcp.calls == [
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "타이레놀", "num_rows": 3},
        )
    ]
