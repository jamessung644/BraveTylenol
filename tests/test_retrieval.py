import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import RetrievalError
from lunit_hackathon.retrieval import RetrievalEngine
from lunit_hackathon.schemas import (
    L2Completion,
    MCPCallResult,
    MCPTool,
    MedicalDomain,
    RouteDecision,
    ToolCall,
)


def call(call_id: str, name: str, arguments: dict | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, function={"name": name, "arguments": raw})


def finalize(*items: dict, status: str = "sufficient") -> L2Completion:
    return L2Completion(
        tool_calls=[
            call(
                "finalize",
                "finalize_retrieval",
                {"status": status, "items": list(items), "note": "planner note"},
            )
        ]
    )


class ScriptedL2:
    def __init__(self, completions: list[L2Completion]) -> None:
        self.completions = list(completions)
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


class FakeMCP:
    def __init__(self, tools: list[MCPTool], results: dict) -> None:
        self.tools = tools
        self.results = results
        self.calls: list[tuple[str, dict]] = []
        self.connected = False

    @asynccontextmanager
    async def connect(self):
        self.connected = True
        yield self

    async def list_tools(self) -> list[MCPTool]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict) -> MCPCallResult:
        self.calls.append((name, arguments))
        value = self.results[name]
        if callable(value):
            value = value(name, arguments)
        if asyncio.iscoroutine(value):
            value = await value
        if isinstance(value, Exception):
            raise value
        return value


def settings(monkeypatch, **updates) -> Settings:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    return Settings(_env_file=None).model_copy(update=updates)


def route(*tool_names: str) -> RouteDecision:
    return RouteDecision(
        domains=frozenset({MedicalDomain.DRUG}),
        tool_names=tool_names,
        retrieval_required=True,
        verification_required=True,
    )


def deadline() -> RequestDeadline:
    return RequestDeadline.start(total_seconds=165.0)


def tool(name: str = "lookup") -> MCPTool:
    return MCPTool(
        name=name,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def result(cite_uid: str, **metadata: str) -> MCPCallResult:
    return MCPCallResult(content=json.dumps({"cite_uid": cite_uid, **metadata}))


async def test_retrieval_exposes_only_routed_discovered_tools(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("m", "mfds", {"query": "tylenol"})]),
            finalize({"cite_uid": "mfds:1", "relevance_score": 1.0}),
        ]
    )
    mcp = FakeMCP([tool("mfds"), tool("adr"), tool("law")], {"mfds": result("mfds:1")})

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "tylenol", route("mfds", "adr"), deadline()
    )

    assert [item["function"]["name"] for item in l2.calls[0]["tools"]] == [
        "mfds",
        "adr",
        "finalize_retrieval",
    ]


async def test_general_health_route_never_exposes_unrelated_discovered_tool(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("rag", "rag", {"query": "headache"})]),
            finalize(status="no_evidence"),
        ]
    )
    mcp = FakeMCP([tool("rag"), tool("law"), tool("drug")], {"rag": result("rag:1")})

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "headache", route("rag"), deadline()
    )

    assert [item["function"]["name"] for item in l2.calls[0]["tools"]] == [
        "rag",
        "finalize_retrieval",
    ]


async def test_no_routed_discovered_tool_fails_without_planner(monkeypatch):
    l2 = ScriptedL2([])
    mcp = FakeMCP([tool("law")], {})

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
            "query", route("mfds"), deadline()
        )

    assert caught.value.code == "mcp_no_routed_tools"
    assert l2.calls == []


async def test_parallel_partial_failure_keeps_actual_evidence(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call("good", "good", {"query": "a"}),
                    call("bad", "bad", {"query": "b"}),
                ]
            ),
            finalize({"cite_uid": "good:1", "relevance_score": 0.8}, status="partial"),
        ]
    )
    mcp = FakeMCP(
        [tool("good"), tool("bad")],
        {"good": result("good:1"), "bad": RetrievalError("failed", code="mcp_tool_call_failed")},
    )

    retrieved = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "query", route("good", "bad"), deadline()
    )

    assert retrieved.status == "partial"
    assert [item.cite_uid for item in retrieved.items] == ["good:1"]
    assert all("failed" not in item.content for item in retrieved.items)


async def test_total_failed_wave_raises_and_never_returns_tool_error_as_evidence(monkeypatch):
    l2 = ScriptedL2([L2Completion(tool_calls=[call("bad", "bad", {"query": "a"})])])
    mcp = FakeMCP(
        [tool("bad")], {"bad": RetrievalError("failed", code="mcp_tool_call_failed")}
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
            "query", route("bad"), deadline()
        )

    assert caught.value.code == "mcp_all_calls_failed"


async def test_two_execution_waves_overlap_and_finalization_is_third_turn(monkeypatch):
    wave_one_ready = asyncio.Event()
    wave_two_ready = asyncio.Event()
    active = {"one": 0, "two": 0}
    peaks = {"one": 0, "two": 0}

    async def concurrent_result(_: str, arguments: dict) -> MCPCallResult:
        wave = "one" if arguments["query"].startswith("one") else "two"
        active[wave] += 1
        peaks[wave] = max(peaks[wave], active[wave])
        if active[wave] == (3 if wave == "one" else 2):
            (wave_one_ready if wave == "one" else wave_two_ready).set()
        await (wave_one_ready if wave == "one" else wave_two_ready).wait()
        active[wave] -= 1
        return result(f"cite:{arguments['query']}")

    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call(f"one-{i}", "lookup", {"query": f"one-{i}"}) for i in range(3)]
            ),
            L2Completion(
                tool_calls=[call(f"two-{i}", "lookup", {"query": f"two-{i}"}) for i in range(2)]
            ),
            finalize({"cite_uid": "cite:one-0", "relevance_score": 1.0}),
        ]
    )
    mcp = FakeMCP([tool()], {"lookup": concurrent_result})

    retrieved = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "query", route("lookup"), deadline()
    )

    assert retrieved.status == "sufficient"
    assert len(mcp.calls) == 5
    assert len(l2.calls) == 3
    assert peaks == {"one": 3, "two": 2}
    assert [item["function"]["name"] for item in l2.calls[2]["tools"]] == ["finalize_retrieval"]


async def test_seventh_proposed_call_is_not_executed(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call(str(index), "lookup", {"query": str(index)}) for index in range(7)]
            ),
            finalize({"cite_uid": "cite:0", "relevance_score": 1.0}),
        ]
    )
    mcp = FakeMCP([tool()], {"lookup": lambda _, args: result(f"cite:{args['query']}")})

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "query", route("lookup"), deadline()
    )

    assert len(mcp.calls) == 6


async def test_duplicate_malformed_unknown_and_schema_invalid_calls_never_execute(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call("valid", "lookup", {"query": "ok"}),
                    call("duplicate", "lookup", {"query": "ok"}),
                    call("malformed", "lookup", "[]"),
                    call("unknown", "outside", {"query": "no"}),
                    call("schema", "lookup", {"wrong": "no"}),
                ]
            ),
            finalize({"cite_uid": "cite:ok", "relevance_score": 1.0}),
        ]
    )
    mcp = FakeMCP([tool()], {"lookup": result("cite:ok")})

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "query", route("lookup"), deadline()
    )

    assert mcp.calls == [("lookup", {"query": "ok"})]


async def test_no_call_recovery_is_forced_once_and_respects_budget(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[]),
            L2Completion(tool_calls=[call("recovery", "lookup", {"query": "ok"})]),
            finalize({"cite_uid": "cite:ok", "relevance_score": 1.0}),
        ]
    )
    mcp = FakeMCP([tool()], {"lookup": result("cite:ok")})

    await RetrievalEngine(l2, mcp, settings(monkeypatch, max_mcp_calls=1)).retrieve(
        "query", route("lookup"), deadline()
    )

    assert len(l2.calls) == 3
    assert [item["function"]["name"] for item in l2.calls[1]["tools"]] == ["lookup"]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == "lookup"
    assert len(mcp.calls) == 1


async def test_finalization_keeps_actual_ids_uses_max_score_and_populates_evidence(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", "lookup", {"query": "ok"})]),
            finalize(
                {"cite_uid": "invented", "relevance_score": 1.0},
                {"cite_uid": "cite:one", "relevance_score": 0.2},
                {"cite_uid": "cite:one", "relevance_score": 0.9},
            ),
        ]
    )
    mcp = FakeMCP(
        [tool()],
        {
            "lookup": result(
                "cite:one", title="Official title", url="https://example.test", jurisdiction="KR"
            )
        },
    )

    retrieved = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "query", route("lookup"), deadline()
    )

    assert [item.cite_uid for item in retrieved.items] == ["cite:one"]
    assert retrieved.items[0].relevance_score == 0.9
    assert retrieved.items[0].authority_rank == 0
    assert retrieved.items[0].title == "Official title"
    assert retrieved.items[0].url == "https://example.test"
    assert retrieved.items[0].jurisdiction == "KR"


async def test_evidence_is_authority_ranked_deduplicated_and_bounded(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call("low", "low", {"query": "low"}),
                    call("high", "openapi_mfds_get_drug_indication", {"query": "high"}),
                ]
            ),
            finalize(
                {"cite_uid": "low:1", "relevance_score": 1.0},
                {"cite_uid": "high:1", "relevance_score": 0.1},
            ),
        ]
    )
    same_content = json.dumps({"title": "same"})
    mcp = FakeMCP(
        [tool("low"), tool("openapi_mfds_get_drug_indication")],
        {
            "low": MCPCallResult(
                content='{"cite_uid":"low:1"}',
                citation_contents={"low:1": same_content},
            ),
            "openapi_mfds_get_drug_indication": MCPCallResult(
                content='{"cite_uid":"high:1"}',
                citation_contents={"high:1": same_content},
            ),
        },
    )

    engine = RetrievalEngine(l2, mcp, settings(monkeypatch, max_evidence_chars=2_000))
    retrieved = await engine.retrieve(
        "query", route("low", "openapi_mfds_get_drug_indication"), deadline()
    )

    assert [item.cite_uid for item in retrieved.items] == ["high:1"]
    assert sum(len(item.content) for item in retrieved.items) <= 24_000


async def test_expired_deadline_fails_before_connect(monkeypatch):
    mcp = FakeMCP([tool()], {})
    expired = RequestDeadline(expires_at=0.0, clock=lambda: 1.0)

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(ScriptedL2([]), mcp, settings(monkeypatch)).retrieve(
            "query", route("lookup"), expired
        )

    assert caught.value.code == "retrieval_deadline_exhausted"
    assert not mcp.connected


async def test_stage_timeout_maps_to_deadline_exhausted(monkeypatch):
    class SlowMCP(FakeMCP):
        async def list_tools(self) -> list[MCPTool]:
            await asyncio.sleep(0.05)
            return self.tools

    mcp = SlowMCP([tool()], {})
    short_deadline = RequestDeadline.start(total_seconds=0.01)
    timeout_settings = settings(monkeypatch, generation_timeout_seconds=0.0)

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(ScriptedL2([]), mcp, timeout_settings).retrieve(
            "query", route("lookup"), short_deadline
        )

    assert caught.value.code == "retrieval_deadline_exhausted"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (RetrievalError("connection", code="mcp_connection_failed"), "mcp_connection_failed"),
        (
            RetrievalError("discovery", code="mcp_tool_discovery_failed"),
            "mcp_tool_discovery_failed",
        ),
    ],
)
async def test_connection_and_discovery_errors_keep_typed_codes(
    monkeypatch, failure, expected_code
):
    class FailingMCP(FakeMCP):
        @asynccontextmanager
        async def connect(self):
            if expected_code == "mcp_connection_failed":
                raise failure
            yield self

        async def list_tools(self):
            if expected_code == "mcp_tool_discovery_failed":
                raise failure
            return self.tools

    with pytest.raises(RetrievalError) as caught:
        engine = RetrievalEngine(ScriptedL2([]), FailingMCP([tool()], {}), settings(monkeypatch))
        await engine.retrieve("query", route("lookup"), deadline())

    assert caught.value.code == expected_code


async def test_invalid_discovered_schema_keeps_typed_code(monkeypatch):
    invalid = MCPTool(name="lookup", input_schema={"type": "not-a-json-schema-type"})

    with pytest.raises(RetrievalError) as caught:
        engine = RetrievalEngine(ScriptedL2([]), FakeMCP([invalid], {}), settings(monkeypatch))
        await engine.retrieve("query", route("lookup"), deadline())

    assert caught.value.code == "mcp_schema_invalid"
