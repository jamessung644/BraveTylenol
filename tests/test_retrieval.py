import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from harness.errors import RetrievalError
from harness.retrieval import RetrievalEngine
from harness.schemas import L2Completion, MCPCallResult, MCPTool, ToolCall


def tool_call(call_id: str, name: str, arguments: object) -> ToolCall:
    return ToolCall(id=call_id, function={"name": name, "arguments": json.dumps(arguments) if not isinstance(arguments, str) else arguments})


class ScriptedL2Client:
    def __init__(self, completions: list[L2Completion]) -> None:
        self._completions = completions
        self.calls: list[dict[str, object]] = []

    async def complete(self, **kwargs: object) -> L2Completion:
        self.calls.append(kwargs)
        return self._completions.pop(0)


class FakeMCPClient:
    def __init__(self, tools: list[MCPTool], results: dict[str, MCPCallResult | Exception]) -> None:
        self.tools = tools
        self.results = results
        self.calls: list[tuple[str, dict[str, object]]] = []

    @asynccontextmanager
    async def connect(self):
        yield self

    async def list_tools(self) -> list[MCPTool]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, object]) -> MCPCallResult:
        self.calls.append((name, arguments))
        result = self.results[name]
        if isinstance(result, Exception):
            raise result
        return result


def tool(name: str = "lookup") -> MCPTool:
    return MCPTool(name=name, description=f"{name} description", input_schema={"type": "object"})


@pytest.mark.parametrize(
    "arguments",
    [{}, {"code": 7}, {"code": "I10", "unexpected": True}],
)
async def test_retrieval_rejects_arguments_that_violate_discovered_tool_schema(settings_with_key, arguments):
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("invalid", "lookup", arguments)]),
        L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "no_evidence", "items": [], "note": "done"})]),
    ])
    schema = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}
    mcp = FakeMCPClient([MCPTool(name="lookup", description="lookup", input_schema=schema)], {"lookup": MCPCallResult(content="{}")})

    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")

    assert mcp.calls == []
    assert result.status == "no_evidence"
    tool_messages = [message.content for message in l2.calls[1]["messages"] if message.role == "tool"]
    assert tool_messages == ['{"error":"arguments do not match tool schema"}']


async def test_retrieval_invalid_discovered_schema_triggers_fallback_error(settings_with_key):
    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call("call", "lookup", {})])])
    invalid_schema = {"type": "not-a-json-schema-type"}
    mcp = FakeMCPClient([MCPTool(name="lookup", description="lookup", input_schema=invalid_schema)], {"lookup": MCPCallResult(content="{}")})

    with pytest.raises(RetrievalError, match="schema"):
        await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")
    assert mcp.calls == []


async def test_retrieval_bounds_invalid_planner_turns_and_returns_no_evidence(settings_with_key):
    turns = settings_with_key.max_tool_calls + 2
    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call(str(index), "unknown", {})]) for index in range(turns)])

    result = await RetrievalEngine(l2, FakeMCPClient([tool()], {}), settings_with_key).retrieve("query")

    assert len(l2.calls) == turns
    assert result.status == "no_evidence"
    assert result.note == "Retrieval planner turn limit exhausted before finalization."


async def test_retrieval_planner_turn_limit_returns_accumulated_candidates_as_partial(settings_with_key):
    turns = settings_with_key.max_tool_calls + 2
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("valid", "lookup", {})]),
        *[L2Completion(tool_calls=[tool_call(str(index), "unknown", {})]) for index in range(1, turns)],
    ])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content='{"cite_uid":"kept"}')})

    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")

    assert len(l2.calls) == turns
    assert result.status == "partial"
    assert [item.cite_uid for item in result.items] == ["kept"]
    assert result.note == "Retrieval planner turn limit exhausted before finalization."


async def test_retrieval_executes_tool_then_resolves_finalized_evidence(settings_with_key):
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("call-1", "kcd_get_name", {"code": "I10"})]),
        L2Completion(tool_calls=[tool_call("call-2", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "kcd:I10", "relevance_score": 0.95}], "note": "KCD 공식 명칭을 확인함"})]),
    ])
    mcp = FakeMCPClient([tool("kcd_get_name")], {"kcd_get_name": MCPCallResult(content='{"cite_uid":"kcd:I10","name":"본태성 고혈압"}')})

    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("I10은 어떤 질병인가?")

    assert result.status == "sufficient"
    assert result.items[0].cite_uid == "kcd:I10"
    assert "본태성 고혈압" in result.items[0].content
    assert mcp.calls == [("kcd_get_name", {"code": "I10"})]
    assert [item["function"]["name"] for item in l2.calls[0]["tools"]] == ["kcd_get_name", "finalize_retrieval"]


async def test_retrieval_executes_same_turn_calls_concurrently(settings_with_key):
    started = 0
    release = asyncio.Event()

    class ConcurrentMCP(FakeMCPClient):
        async def call_tool(self, name, arguments):
            nonlocal started
            self.calls.append((name, arguments))
            started += 1
            if started == 2:
                release.set()
            await release.wait()
            return self.results[name]

    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call("a", "one", {}), tool_call("b", "two", {})]), L2Completion(tool_calls=[tool_call("f", "finalize_retrieval", {"status": "sufficient", "items": [], "note": ""})])])
    mcp = ConcurrentMCP([tool("one"), tool("two")], {"one": MCPCallResult(content='{"cite_uid":"one"}'), "two": MCPCallResult(content='{"cite_uid":"two"}')})

    await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")
    assert {name for name, _ in mcp.calls} == {"one", "two"}


@pytest.mark.parametrize("arguments", ["{", ["not", "an", "object"]])
async def test_retrieval_returns_protocol_errors_for_invalid_arguments(settings_with_key, arguments):
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("bad", "lookup", arguments)]),
        L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "no_evidence", "items": [], "note": "none"})]),
    ])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content="{}")})

    await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")
    assert mcp.calls == []
    assert "error" in l2.calls[1]["messages"][-2].content.lower()


async def test_retrieval_returns_unknown_and_mcp_errors_to_l2(settings_with_key):
    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call("unknown", "missing", {}), tool_call("bad", "lookup", {})]), L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "no_evidence", "items": [], "note": "none"})])])
    mcp = FakeMCPClient([tool()], {"lookup": RetrievalError("down")})

    await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")
    results = [message.content for message in l2.calls[1]["messages"] if message.role == "tool"]
    assert len(results) == 2
    assert all("error" in content.lower() for content in results)


async def test_retrieval_immediate_finalization_has_no_evidence(settings_with_key):
    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "sufficient", "items": [], "note": "non-citable fact"})])])
    result = await RetrievalEngine(l2, FakeMCPClient([tool()], {}), settings_with_key).retrieve("query")
    assert result.status == "no_evidence"
    assert result.note == "non-citable fact"


async def test_retrieval_processes_valid_sibling_before_finalizing(settings_with_key):
    l2 = ScriptedL2Client([L2Completion(tool_calls=[
        tool_call("lookup", "lookup", {}),
        tool_call("final", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "cite", "relevance_score": 1}], "note": ""}),
    ])])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content='{"cite_uid":"cite"}')})

    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")

    assert mcp.calls == [("lookup", {})]
    assert [item.cite_uid for item in result.items] == ["cite"]


@pytest.mark.parametrize("name, arguments", [("missing", {}), ("lookup", "{")])
async def test_retrieval_processes_invalid_sibling_before_finalizing(settings_with_key, monkeypatch, name, arguments):
    errors = []
    from harness import retrieval

    original_tool_error = retrieval._tool_error

    def record_tool_error(call, message):
        errors.append((call.id, message))
        return original_tool_error(call, message)

    monkeypatch.setattr(retrieval, "_tool_error", record_tool_error)
    l2 = ScriptedL2Client([L2Completion(tool_calls=[
        tool_call("invalid", name, arguments),
        tool_call("final", "finalize_retrieval", {"status": "no_evidence", "items": [], "note": "done"}),
    ])])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content='{"cite_uid":"unexpected"}')})

    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")

    assert mcp.calls == []
    assert errors[0][0] == "invalid"
    assert result.status == "no_evidence"
    assert result.note == "done"


async def test_retrieval_normalizes_duplicate_scores_and_total_evidence_limit(settings_with_key, monkeypatch):
    monkeypatch.setenv("MAX_EVIDENCE_CHARS", "2000")
    settings = type(settings_with_key)()
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("call", "lookup", {})]),
        L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "b", "relevance_score": 0.2}, {"cite_uid": "a", "relevance_score": 0.4}, {"cite_uid": "a", "relevance_score": 0.9}, {"cite_uid": "missing", "relevance_score": 1}], "note": "note"})]),
    ])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content='[{"cite_uid":"a","text":"A"},{"cite_uid":"b","text":"B"}]')})
    result = await RetrievalEngine(l2, mcp, settings).retrieve("query")
    assert [(item.cite_uid, item.relevance_score) for item in result.items] == [("a", 0.9), ("b", 0.2)]


async def test_retrieval_budget_is_hard_cap_and_returns_discovery_order(settings_with_key):
    l2 = ScriptedL2Client([L2Completion(tool_calls=[tool_call(str(i), "lookup", {"n": i}) for i in range(5)])])
    mcp = FakeMCPClient([tool()], {"lookup": MCPCallResult(content='{"cite_uid":"same"}')})
    result = await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")
    assert len(mcp.calls) == 4
    assert result.status == "partial"
    assert result.note == "MCP tool-call budget exhausted before finalization."
    assert [item.cite_uid for item in result.items] == ["same"]


async def test_retrieval_rejects_finalize_name_collision(settings_with_key):
    l2 = ScriptedL2Client([])
    mcp = FakeMCPClient([tool("finalize_retrieval")], {})
    with pytest.raises(RetrievalError, match="collision"):
        await RetrievalEngine(l2, mcp, settings_with_key).retrieve("query")


async def test_retrieval_caps_each_tool_result_before_returning_evidence(settings_with_key, monkeypatch):
    monkeypatch.setenv("MAX_TOOL_RESULT_CHARS", "1000")
    settings = type(settings_with_key)()
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("call", "lookup", {})]),
        L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "long", "relevance_score": 1}], "note": ""})]),
    ])
    content = json.dumps({"cite_uid": "long", "text": "x" * 2_000})
    result = await RetrievalEngine(l2, FakeMCPClient([tool()], {"lookup": MCPCallResult(content=content)}), settings).retrieve("query")
    assert len(result.items[0].content) == 1_000
    assert result.items[0].content.endswith("...[truncated]")


async def test_retrieval_caps_total_selected_evidence(settings_with_key, monkeypatch):
    monkeypatch.setenv("MAX_EVIDENCE_CHARS", "2000")
    settings = type(settings_with_key)()
    l2 = ScriptedL2Client([
        L2Completion(tool_calls=[tool_call("1", "one", {}), tool_call("2", "two", {})]),
        L2Completion(tool_calls=[tool_call("final", "finalize_retrieval", {"status": "sufficient", "items": [{"cite_uid": "one", "relevance_score": 1}, {"cite_uid": "two", "relevance_score": 0.5}], "note": ""})]),
    ])
    payload = lambda uid: json.dumps({"cite_uid": uid, "text": "x" * 1_400})
    mcp = FakeMCPClient([tool("one"), tool("two")], {"one": MCPCallResult(content=payload("one")), "two": MCPCallResult(content=payload("two"))})
    result = await RetrievalEngine(l2, mcp, settings).retrieve("query")
    assert sum(len(item.content) for item in result.items) == 2_000
    assert len(result.items) == 2
