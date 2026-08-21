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
            )
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
    assert result.status == "partial"
    assert [item.cite_uid for item in result.items] == ["one"]
    assert result.note == "MCP tool-call budget exhausted before finalization."
