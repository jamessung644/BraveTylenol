import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import lunit_hackathon.retrieval as retrieval_module
from lunit_hackathon.artifacts import (
    FINALIZE_RETRIEVAL_TOOL,
    MCP_TOOL_ALIASES,
    RETRIEVAL_SYSTEM_PROMPT,
)
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import RetrievalError, UpstreamTimeoutError
from lunit_hackathon.mcp_client import SDKMCPConnection
from lunit_hackathon.retrieval import RetrievalEngine, ToolBindingRegistry
from lunit_hackathon.schemas import L2Completion, MCPCallResult, MCPTool, ToolCall

LOOKUP_ALIAS = "adr_retrieve_drug_info"
SECOND_LOOKUP_ALIAS = "kcd_get_name"
LAW_ALIASES = {
    "openapi_law_get_article",
    "openapi_law_list_articles",
    "openapi_law_search",
}
INDEX_ALIASES = {
    "index_get_document_structure",
    "index_get_page_content",
    "index_get_relevant_nodes",
    "index_keyword_search",
    "index_list_documents",
}
EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def call(call_id: str, name: str, arguments: dict | str) -> ToolCall:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id, function={"name": name, "arguments": raw})


class ScriptedL2:
    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        completion = self.completions.pop(0)
        if completion.tool_calls and completion.finish_reason is None:
            return completion.model_copy(update={"finish_reason": "tool_calls"})
        return completion


class FakeMCP:
    def __init__(self, tools, results):
        self.tools = tools
        self.results = results
        self.calls = []
        self.list_calls = 0

    @asynccontextmanager
    async def connect(self):
        yield self

    async def list_tools(self):
        self.list_calls += 1
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        result = self.results[name]
        if isinstance(result, Exception):
            raise result
        return result


class SDKBackedMCP:
    def __init__(self, sdk):
        self.sdk = sdk

    @asynccontextmanager
    async def connect(self):
        yield SDKMCPConnection(self.sdk, max_tool_result_chars=8_000)


def settings(monkeypatch, **updates):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
    monkeypatch.delenv("MAX_MCP_CALLS", raising=False)
    monkeypatch.delenv("MAX_TOOL_CALLS", raising=False)
    return Settings(_env_file=None).model_copy(update=updates)


def lookup_tool(
    name: str = LOOKUP_ALIAS,
    *,
    model_function_name: str | None = None,
) -> MCPTool:
    return MCPTool(
        name=name,
        model_function_name=model_function_name,
        description="authoritative lookup",
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    )


def canonical_tools(
    *overrides: MCPTool,
    extras: tuple[MCPTool, ...] = (),
) -> list[MCPTool]:
    by_alias = {tool.model_function_name or tool.name: tool for tool in overrides}
    assert set(by_alias).issubset(MCP_TOOL_ALIASES)
    tools = [
        by_alias.get(
            alias,
            MCPTool(
                name=alias,
                description=f"canonical {alias}",
                input_schema=copy.deepcopy(EMPTY_SCHEMA),
            ),
        )
        for alias in MCP_TOOL_ALIASES
    ]
    return [*tools, *extras]


def finalization(
    *,
    status: str = "sufficient",
    items: list[dict] | None = None,
    note: str = "done",
) -> L2Completion:
    selected = [{"cite_uid": "evidence:1", "relevance_score": 0.9}] if items is None else items
    return L2Completion(
        tool_calls=[
            call(
                "finish",
                "finalize_retrieval",
                {"status": status, "items": selected, "note": note},
            )
        ]
    )


def active_aliases(l2_call: dict) -> set[str]:
    prompt = l2_call["messages"][0].content
    return {alias for alias in MCP_TOOL_ALIASES if alias in prompt}


def test_release_artifact_call_ceiling_is_three():
    assert retrieval_module._ARTIFACT_MAX_MCP_CALLS == 3


async def test_retrieval_uses_artifact_prompt_exact_registry_and_local_finalizer(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup-1", LOOKUP_ALIAS, {"code": "I10"})]),
            finalization(
                items=[{"cite_uid": "kcd:I10", "relevance_score": 0.9}],
                note="공식 분류 근거",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"kcd:I10","name":"본태성 고혈압"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("ADR 공식 근거")

    assert result.status == "sufficient"
    assert result.execution_status == "ok"
    assert result.items[0].cite_uid == "kcd:I10"
    assert "본태성 고혈압" in result.items[0].content
    assert mcp.calls == [(LOOKUP_ALIAS, {"code": "I10"})]
    active_prompt = l2.calls[0]["messages"][0].content
    assert active_prompt != RETRIEVAL_SYSTEM_PROMPT
    assert "검색·열람 MCP 호출은 최대 1회다" in active_prompt
    assert l2.calls[0]["attempt_timeout_seconds"] == 25
    assert l2.calls[1]["attempt_timeout_seconds"] == 25
    names = [item["function"]["name"] for item in l2.calls[0]["tools"]]
    assert names == [LOOKUP_ALIAS, "finalize_retrieval"]
    assert active_aliases(l2.calls[0]) == set(names) - {"finalize_retrieval"}
    assert all(item["function"]["strict"] is True for item in l2.calls[0]["tools"])
    assert l2.calls[0]["tools"][-1] == FINALIZE_RETRIEVAL_TOOL
    assert all(name != "finalize_retrieval" for name, _ in mcp.calls)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_retrieval_rejects_tool_call_with_invalid_finish_reason_before_mcp(
    monkeypatch,
    finish_reason,
):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call("truncated", LOOKUP_ALIAS, {"code": "I10"})],
                finish_reason=finish_reason,
            )
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("ADR 공식 근거")

    assert caught.value.code == "retrieval_planner_finish_reason_invalid"
    assert mcp.calls == []


async def test_retrieval_accepts_official_provider_stop_for_structured_tool_call(
    monkeypatch,
):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call("provider-stop", LOOKUP_ALIAS, {"code": "I10"})],
                finish_reason="stop",
            ),
            finalization(),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("ADR 공식 근거")

    assert result.execution_status == "ok"
    assert mcp.calls == [(LOOKUP_ALIAS, {"code": "I10"})]


async def test_retrieval_maps_model_alias_to_exact_transport_name(monkeypatch):
    transport_name = "server/drug-lookup"
    discovered = lookup_tool(transport_name, model_function_name=LOOKUP_ALIAS)
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "A"})]),
            finalization(),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(discovered),
        {transport_name: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=2),
    ).retrieve("query")

    assert mcp.calls == [(transport_name, {"code": "A"})]
    assert result.items[0].source_tool == transport_name


async def test_verified_tool_registry_is_discovered_once_and_reused(monkeypatch):
    registry = ToolBindingRegistry()
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    for suffix in ("one", "two"):
        l2 = ScriptedL2(
            [
                L2Completion(tool_calls=[call(f"lookup-{suffix}", LOOKUP_ALIAS, {"code": suffix})]),
                finalization(),
            ]
        )
        await RetrievalEngine(
            l2,
            mcp,
            settings(monkeypatch),
            binding_registry=registry,
        ).retrieve("ADR 공식 근거")

    assert mcp.list_calls == 1


async def test_unrelated_bad_live_schema_is_quarantined_while_good_route_executes(
    monkeypatch,
):
    bad_unrelated_tool = MCPTool(
        name="rag_sql_query",
        description="unsupported open object",
        input_schema={
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "additionalProperties": True,
                }
            },
            "required": [],
        },
    )
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "I10"})]
            ),
            finalization(),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool(), bad_unrelated_tool),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(
        "ADR 공식 근거"
    )

    assert result.status == "sufficient"
    assert mcp.calls == [(LOOKUP_ALIAS, {"code": "I10"})]
    offered_names = {
        tool["function"]["name"] for tool in l2.calls[0]["tools"]
    }
    assert offered_names == {LOOKUP_ALIAS, "finalize_retrieval"}


async def test_real_sdk_boundary_quarantines_bad_unrelated_entry_and_executes_good_route(
    monkeypatch,
):
    class RawSDK:
        def __init__(self):
            self.calls = []

        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    *canonical_tools(lookup_tool()),
                    SimpleNamespace(
                        name="invalid tool name",
                        description="bad unrelated entry",
                        input_schema=["not", "an", "object"],
                    ),
                ]
            )

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return SimpleNamespace(
                content=[],
                structured_content={"cite_uid": "evidence:1"},
                is_error=False,
            )

    sdk = RawSDK()
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "I10"})]),
            finalization(),
        ]
    )

    result = await RetrievalEngine(
        l2,
        SDKBackedMCP(sdk),
        settings(monkeypatch),
    ).retrieve("ADR 공식 근거")

    assert result.status == "sufficient"
    assert sdk.calls == [(LOOKUP_ALIAS, {"code": "I10"})]


async def test_real_sdk_boundary_quarantines_duplicate_selected_alias(monkeypatch):
    class RawSDK:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[*canonical_tools(lookup_tool()), lookup_tool()]
            )

    l2 = ScriptedL2([])
    engine = RetrievalEngine(
        l2,
        SDKBackedMCP(RawSDK()),
        settings(monkeypatch),
    )

    with pytest.raises(RetrievalError) as caught:
        await engine.retrieve("ADR 공식 근거")

    assert caught.value.code == "retrieval_route_unavailable"
    assert l2.calls == []


@pytest.mark.parametrize("invalid_schema", [[], False, {}, None])
async def test_real_sdk_boundary_never_coerces_falsy_selected_schema(
    monkeypatch,
    invalid_schema,
):
    class RawSDK:
        async def list_tools(self):
            safe = [
                tool
                for tool in canonical_tools()
                if tool.name != LOOKUP_ALIAS
            ]
            return SimpleNamespace(
                tools=[
                    *safe,
                    SimpleNamespace(
                        name=LOOKUP_ALIAS,
                        description="invalid selected schema",
                        input_schema=invalid_schema,
                    ),
                ]
            )

    l2 = ScriptedL2([])
    engine = RetrievalEngine(
        l2,
        SDKBackedMCP(RawSDK()),
        settings(monkeypatch),
    )

    with pytest.raises(RetrievalError) as caught:
        await engine.retrieve("ADR 공식 근거")

    assert caught.value.code == "retrieval_route_unavailable"
    assert l2.calls == []


@pytest.mark.parametrize(
    ("query", "expected_names"),
    [
        (
            "식약쳐 타이레놀 허가",
            {
                "openapi_mfds_check_drug_permission",
                "openapi_mfds_find_drugs_by_ingredient",
                "openapi_mfds_get_drug_indication",
            },
        ),
        (
            "심평언 와파린 급여",
            {
                "hira_updates_search",
                "openapi_hira_disease_check_code",
                "openapi_hira_get_drug_price",
            },
        ),
    ],
)
async def test_noisy_authority_names_select_the_intended_minimal_registry(
    monkeypatch,
    query,
    expected_names,
):
    l2 = ScriptedL2([finalization(status="no_evidence", items=[])])
    mcp = FakeMCP(canonical_tools(), {})

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve(query)

    names = {tool["function"]["name"] for tool in l2.calls[0]["tools"]}
    assert names == {*expected_names, "finalize_retrieval"}
    assert active_aliases(l2.calls[0]) == expected_names
    assert mcp.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "공식 의약품 라벨에서 이 약의 금기와 경고를 확인해 줘",
        "What contraindications are listed in the official drug label?",
        "Check the prescribing information for adverse reactions.",
        "Read the package insert before answering.",
        "What is the FDA-approved indication?",
        "Summarize the FDA-approved label.",
    ],
)
async def test_official_drug_label_intent_uses_only_bounded_adr_route(
    monkeypatch,
    query,
):
    l2 = ScriptedL2([finalization(status="no_evidence", items=[])])

    await RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(), {}),
        settings(monkeypatch),
    ).retrieve(query)

    request = l2.calls[0]
    names = {tool["function"]["name"] for tool in request["tools"]}
    assert names == {"adr_retrieve_drug_info", "finalize_retrieval"}
    assert "검색·열람 MCP 호출은 최대 1회다" in request["messages"][0].content


@pytest.mark.parametrize(
    ("query", "expected_budget"),
    [
        ("식약처 타이레놀 허가", 1),
        ("심평원 아세트아미노펜 약가", 1),
        ("아픽사반과 이트라코나졸 상호작용", 1),
        ("KCD 당뇨병 코드를 찾아줘", 2),
        ("현재 당뇨 진단 기준 최신 진료지침", 3),
        ("감염병의 예방 및 관리에 관한 법률 시행 조문", 3),
    ],
)
async def test_domain_ceiling_is_rendered_into_exact_active_prompt(
    monkeypatch,
    query,
    expected_budget,
):
    l2 = ScriptedL2([finalization(status="no_evidence", items=[])])
    await RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(), {}),
        settings(monkeypatch),
    ).retrieve(query)

    request = l2.calls[0]
    remote_names = {
        tool["function"]["name"]
        for tool in request["tools"]
        if tool["function"]["name"] != "finalize_retrieval"
    }
    assert active_aliases(request) == remote_names
    assert f"검색·열람 MCP 호출은 최대 {expected_budget}회다" in (request["messages"][0].content)


async def test_research_chain_reaches_vector_evidence_on_third_hop(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call("sources", "rag_get_all_data_sources", {})]
            ),
            L2Completion(
                tool_calls=[call("detail", "rag_get_data_source_detail", {})]
            ),
            L2Completion(tool_calls=[call("vector", "rag_vector_query", {})]),
            finalization(
                items=[{"cite_uid": "research:vector:1", "relevance_score": 1}],
                note="research body retrieved",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "rag_get_all_data_sources": MCPCallResult(content='{"sources":["pubmed"]}'),
            "rag_get_data_source_detail": MCPCallResult(content='{"source":"pubmed"}'),
            "rag_vector_query": MCPCallResult(
                content='{"cite_uid":"research:vector:1","text":"study result"}'
            ),
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch),
    ).retrieve("Find the PubMed research evidence for aspirin.")

    assert [name for name, _ in mcp.calls] == [
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
        "rag_vector_query",
    ]
    assert result.items[0].cite_uid == "research:vector:1"
    assert "검색·열람 MCP 호출은 최대 3회다" in l2.calls[0]["messages"][0].content
    assert l2.calls[0]["tool_choice"]["function"]["name"] == (
        "rag_get_all_data_sources"
    )
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "rag_get_data_source_detail"
    )
    assert l2.calls[2]["tool_choice"]["function"]["name"] == "rag_vector_query"


@pytest.mark.parametrize("configured_budget", [1, 2])
async def test_research_chain_short_configured_budget_fails_closed_without_citation(
    monkeypatch,
    configured_budget,
):
    steps = [
        L2Completion(tool_calls=[call("sources", "rag_get_all_data_sources", {})])
    ]
    if configured_budget == 2:
        steps.append(
            L2Completion(
                tool_calls=[call("detail", "rag_get_data_source_detail", {})]
            )
        )
    steps.append(finalization(status="no_evidence", items=[], note="chain incomplete"))
    l2 = ScriptedL2(steps)
    mcp = FakeMCP(
        canonical_tools(),
        {
            "rag_get_all_data_sources": MCPCallResult(
                content='{"cite_uid":"discovery:sources","sources":["pubmed"]}'
            ),
            "rag_get_data_source_detail": MCPCallResult(
                content='{"cite_uid":"discovery:detail","source":"pubmed"}'
            ),
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=configured_budget),
    ).retrieve("Find the PubMed research evidence for aspirin.")

    assert result.status == "no_evidence"
    assert result.items == []
    assert len(mcp.calls) == configured_budget
    assert l2.calls[0]["tool_choice"]["function"]["name"] == (
        "rag_get_all_data_sources"
    )
    assert l2.calls[-1]["tool_choice"]["function"]["name"] == "finalize_retrieval"


async def test_guideline_ab_max_one_stops_before_page_and_yields_no_citation(
    monkeypatch,
):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("nodes", "index_get_relevant_nodes", {})]),
            finalization(
                status="no_evidence",
                items=[],
                note="원문 page를 확인하지 못함",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "index_get_relevant_nodes": MCPCallResult(
                content=(
                    '{"cite_uid":"discovery:diabetes-node",'
                    '"nodes":[{"node_id":"diabetes-diagnosis","page":17}]}'
                )
            )
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=1),
    ).retrieve("현재 당뇨 진단 기준 최신 진료지침")

    assert result.status == "no_evidence"
    assert result.items == []
    assert mcp.calls == [("index_get_relevant_nodes", {})]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == "finalize_retrieval"


async def test_discovery_only_uid_cannot_be_finalized_as_evidence(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("nodes", "index_get_relevant_nodes", {})]),
            finalization(
                items=[{"cite_uid": "discovery:diabetes-node", "relevance_score": 1}],
                note="중간 node만 관찰",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "index_get_relevant_nodes": MCPCallResult(
                content='{"cite_uid":"discovery:diabetes-node","page":17}',
                cite_uids=["discovery:diabetes-node"],
            )
        },
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            mcp,
            settings(monkeypatch, max_mcp_calls=1),
        ).retrieve("현재 당뇨 진단 기준 최신 진료지침")

    assert caught.value.code == "retrieval_finalize_unobserved_uid"


async def test_default_budget_guideline_reaches_citable_page_in_two_hops(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("nodes", "index_get_relevant_nodes", {})]),
            L2Completion(tool_calls=[call("page", "index_get_page_content", {})]),
            finalization(
                items=[{"cite_uid": "guideline:diabetes:p17", "relevance_score": 1}],
                note="원문 진단 기준 확인",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "index_get_relevant_nodes": MCPCallResult(
                content=(
                    '{"cite_uid":"discovery:diabetes-node",'
                    '"nodes":[{"node_id":"diabetes-diagnosis","page":17}]}'
                )
            ),
            "index_get_page_content": MCPCallResult(
                content=(
                    '{"cite_uid":"guideline:diabetes:p17","page":17,"text":"diagnostic criteria"}'
                )
            ),
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch),
    ).retrieve("현재 당뇨 진단 기준 최신 진료지침")

    assert result.status == "sufficient"
    assert [item.cite_uid for item in result.items] == ["guideline:diabetes:p17"]
    assert [name for name, _ in mcp.calls] == [
        "index_get_relevant_nodes",
        "index_get_page_content",
    ]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "index_get_page_content"
    )
    assert active_aliases(l2.calls[0]) == INDEX_ALIASES


async def test_default_budget_law_reaches_article_body_in_three_hops(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("law", "openapi_law_search", {})]),
            L2Completion(tool_calls=[call("articles", "openapi_law_list_articles", {})]),
            L2Completion(tool_calls=[call("article", "openapi_law_get_article", {})]),
            finalization(
                items=[{"cite_uid": "law:infection:article-49", "relevance_score": 1}],
                note="조문 원문과 시행일 확인",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "openapi_law_search": MCPCallResult(
                content=(
                    '{"cite_uid":"discovery:infection-law",'
                    '"laws":[{"mst":"123","title":"감염병예방법"}]}'
                )
            ),
            "openapi_law_list_articles": MCPCallResult(
                content=(
                    '{"cite_uid":"discovery:article-49",'
                    '"articles":[{"key":"article-49","title":"감염병의 예방 조치"}]}'
                )
            ),
            "openapi_law_get_article": MCPCallResult(
                content=(
                    '{"cite_uid":"law:infection:article-49",'
                    '"effective_date":"2026-01-01","text":"조문 원문"}'
                )
            ),
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch),
    ).retrieve("감염병의 예방 및 관리에 관한 법률 제49조 시행 조문")

    assert result.status == "sufficient"
    assert result.items[0].cite_uid == "law:infection:article-49"
    assert [name for name, _ in mcp.calls] == [
        "openapi_law_search",
        "openapi_law_list_articles",
        "openapi_law_get_article",
    ]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "openapi_law_list_articles"
    )
    assert l2.calls[2]["tool_choice"]["function"]["name"] == (
        "openapi_law_get_article"
    )
    assert active_aliases(l2.calls[0]) == LAW_ALIASES


async def test_law_discovery_cannot_finalize_before_required_content_hop(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("law", "openapi_law_search", {})]),
            finalization(status="no_evidence", items=[], note="검색 결과만 확인"),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(),
        {
            "openapi_law_search": MCPCallResult(
                content='{"laws":[{"mst":"123","title":"의료법"}]}'
            )
        },
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            mcp,
            settings(monkeypatch, max_mcp_calls=3),
        ).retrieve("의료법 현행 조문 원문")

    assert caught.value.code == "retrieval_chain_incomplete"
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "openapi_law_list_articles"
    )


async def test_repeated_discovery_alias_is_rejected_even_with_new_arguments(monkeypatch):
    index_list = lookup_tool(name="index_list_documents")
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[call("first", "index_list_documents", {"code": "A"})]
            ),
            L2Completion(
                tool_calls=[call("second", "index_list_documents", {"code": "B"})]
            ),
            finalization(status="no_evidence", items=[]),
        ]
    )
    mcp = FakeMCP(
        [index_list],
        {"index_list_documents": MCPCallResult(content='{"documents":[]}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=3),
    ).retrieve("현재 당뇨 진료지침")

    assert result.status == "no_evidence"
    assert result.execution_status == "schema_error"
    assert mcp.calls == [("index_list_documents", {"code": "A"})]
    errors = [message for message in l2.calls[2]["messages"] if message.role == "tool"]
    assert "discovery step already used" in json.loads(errors[-1].content)["error"]


async def test_ingredient_only_mfds_route_gets_one_product_label_followup(monkeypatch):
    ingredient_tool = MCPTool(
        name="openapi_mfds_find_drugs_by_ingredient",
        input_schema={
            "type": "object",
            "properties": {
                "ingredient": {"type": "string"},
                "num_rows": {"type": "integer", "minimum": 1},
            },
            "required": ["ingredient"],
            "additionalProperties": False,
        },
    )
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
                        "products",
                        "openapi_mfds_find_drugs_by_ingredient",
                        {"ingredient": "아세트아미노펜", "num_rows": None},
                    )
                ]
            ),
            L2Completion(
                tool_calls=[
                    call(
                        "label",
                        "openapi_mfds_get_drug_indication",
                        {"drug_name": "가상제품", "num_rows": None},
                    )
                ]
            ),
            finalization(
                items=[{"cite_uid": "mfds:product-label", "relevance_score": 1}],
                note="제품 허가사항 원문 확인",
            ),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(ingredient_tool, indication_tool),
        {
            "openapi_mfds_find_drugs_by_ingredient": MCPCallResult(
                content=(
                    '{"cite_uid":"mfds:product-label",'
                    '"products":[{"item_seq":"123"}]}'
                )
            ),
            "openapi_mfds_get_drug_indication": MCPCallResult(
                content='{"cite_uid":"mfds:product-label","indication":"허가 적응증"}'
            ),
        },
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch),
    ).retrieve("식약처에서 아세트아미노펜 성분 제품의 허가사항을 확인해 줘")

    assert [item.cite_uid for item in result.items] == ["mfds:product-label"]
    assert [name for name, _ in mcp.calls] == [
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_mfds_get_drug_indication",
    ]
    assert l2.calls[1]["tool_choice"]["function"]["name"] == (
        "openapi_mfds_get_drug_indication"
    )
    assert "검색·열람 MCP 호출은 최대 2회다" in l2.calls[0]["messages"][0].content


async def test_retrieval_rejects_invalid_arguments_without_execution(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("bad", LOOKUP_ALIAS, {"wrong": "I10"})]),
            finalization(status="no_evidence", items=[], note="none"),
        ]
    )
    mcp = FakeMCP(canonical_tools(lookup_tool()), {})

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=2),
    ).retrieve("query")

    assert result.status == "no_evidence"
    assert result.execution_status == "schema_error"
    assert result.semantic_reason == "no_match"
    assert mcp.calls == []
    tool_errors = [message for message in l2.calls[1]["messages"] if message.role == "tool"]
    assert json.loads(tool_errors[0].content)["error"] == ("arguments do not match tool schema")


async def test_retrieval_enforces_configured_tool_budget(monkeypatch):
    monkeypatch.setattr(retrieval_module, "_ARTIFACT_MAX_MCP_CALLS", 6)
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("0", LOOKUP_ALIAS, {"code": "0"})]),
            L2Completion(tool_calls=[call("1", LOOKUP_ALIAS, {"code": "1"})]),
            finalization(note="budget reached"),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1","text":"evidence"}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=2),
    ).retrieve("query")

    assert len(mcp.calls) == 2
    assert result.execution_status == "ok"
    assert result.note == "budget reached"
    assert l2.calls[2]["tools"] == l2.calls[0]["tools"]
    assert active_aliases(l2.calls[2]) == active_aliases(l2.calls[0])
    assert l2.calls[2]["tool_choice"]["function"]["name"] == "finalize_retrieval"


async def test_retrieval_caps_calls_at_artifact_prompt_budget(monkeypatch):
    monkeypatch.setattr(retrieval_module, "_ARTIFACT_MAX_MCP_CALLS", 3)
    l2 = ScriptedL2(
        [
            *(
                L2Completion(tool_calls=[call(str(index), LOOKUP_ALIAS, {"code": str(index)})])
                for index in range(3)
            ),
            finalization(),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"evidence:1"}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=12),
    ).retrieve("query")

    assert len(mcp.calls) == 3
    assert result.execution_status == "ok"


async def test_retrieval_fails_closed_when_no_tool_for_selected_route_survives(monkeypatch):
    l2 = ScriptedL2([])
    engine = RetrievalEngine(
        l2,
        FakeMCP([lookup_tool()], {}),
        settings(monkeypatch),
    )

    with pytest.raises(RetrievalError) as caught:
        await engine.retrieve("KCD 당뇨병 공식 코드")

    assert caught.value.code == "retrieval_route_unavailable"
    assert l2.calls == []


async def test_retrieval_keeps_safe_survivors_when_selected_chain_is_partial(monkeypatch):
    bad_article = MCPTool(
        name="openapi_law_get_article",
        description="lossy open object",
        input_schema={
            "type": "object",
            "properties": {"filters": {"type": "object", "additionalProperties": True}},
            "required": ["filters"],
        },
    )
    l2 = ScriptedL2([finalization(status="no_evidence", items=[])])
    engine = RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(bad_article), {}),
        settings(monkeypatch, max_mcp_calls=3),
    )

    result = await engine.retrieve("의료법 현행 조문 원문")

    assert result.status == "no_evidence"
    offered = {
        tool["function"]["name"] for tool in l2.calls[0]["tools"]
    }
    assert offered == {
        "openapi_law_search",
        "openapi_law_list_articles",
        "finalize_retrieval",
    }
    assert "openapi_law_get_article" not in l2.calls[0]["messages"][0].content


async def test_retrieval_quarantines_unapproved_discovery_entries(monkeypatch):
    extra = MCPTool(
        name="mcp__dev__shadow_tool",
        input_schema=copy.deepcopy(EMPTY_SCHEMA),
    )
    l2 = ScriptedL2([finalization(status="no_evidence", items=[])])

    result = await RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(extras=(extra,)), {}),
        settings(monkeypatch, max_mcp_calls=0),
    ).retrieve("query")

    assert result.status == "no_evidence"
    names = [tool["function"]["name"] for tool in l2.calls[0]["tools"]]
    assert names == ["finalize_retrieval"]


async def test_retrieval_timeout_becomes_recoverable_error(monkeypatch):
    class SlowMCP(FakeMCP):
        async def list_tools(self):
            await asyncio.sleep(1)
            return []

    short_settings = settings(monkeypatch).model_copy(update={"request_timeout_seconds": 0.01})
    engine = RetrievalEngine(ScriptedL2([]), SlowMCP([], {}), short_settings)

    with pytest.raises(RetrievalError) as caught:
        await engine.retrieve("query")

    assert caught.value.code == "retrieval_timeout"


async def test_retrieval_planner_timeout_becomes_recoverable_error(monkeypatch):
    class TimedOutL2(ScriptedL2):
        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            raise UpstreamTimeoutError("synthetic planner timeout")

    l2 = TimedOutL2([])
    engine = RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(lookup_tool()), {}),
        settings(monkeypatch),
    )

    with pytest.raises(RetrievalError) as caught:
        await engine.retrieve("ADR 공식 근거")

    assert caught.value.code == "retrieval_model_timeout"
    assert l2.calls[0]["attempt_timeout_seconds"] == 25


async def test_retrieval_rejects_duplicate_calls_and_uses_previous_result(monkeypatch):
    monkeypatch.setattr(retrieval_module, "_ARTIFACT_MAX_MCP_CALLS", 2)
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("first", LOOKUP_ALIAS, {"code": "I10"})]),
            L2Completion(tool_calls=[call("duplicate", LOOKUP_ALIAS, {"code": "I10"})]),
            finalization(items=[{"cite_uid": "kcd:I10", "relevance_score": 1}]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"kcd:I10"}')},
    )

    result = await RetrievalEngine(
        l2,
        mcp,
        settings(monkeypatch, max_mcp_calls=2),
    ).retrieve("query")

    assert mcp.calls == [(LOOKUP_ALIAS, {"code": "I10"})]
    assert result.items[0].cite_uid == "kcd:I10"
    errors = [message for message in l2.calls[2]["messages"] if message.role == "tool"]
    assert "duplicate tool call" in json.loads(errors[-1].content)["error"]


async def test_retrieval_enforces_requested_kcd_revision(monkeypatch):
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
            finalization(items=[{"cite_uid": "kcd8:I10", "relevance_score": 1}]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(kcd_tool),
        {"kcd_get_name": MCPCallResult(content='{"cite_uid":"kcd8:I10"}')},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("KCD-8 I10의 공식 명칭")

    assert mcp.calls == [("kcd_get_name", {"code": "I10", "revision": "KCD-8"})]
    assert result.items[0].cite_uid == "kcd8:I10"


async def test_retrieval_projects_optional_null_then_applies_row_cap(monkeypatch):
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
                        {"drug_name": "타이레놀", "num_rows": None},
                    )
                ]
            ),
            finalization(items=[{"cite_uid": "mfds:1", "relevance_score": 1}]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(indication_tool),
        {"openapi_mfds_get_drug_indication": MCPCallResult(content='{"cite_uid":"mfds:1"}')},
    )

    await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("식약처 타이레놀 허가 효능")

    assert mcp.calls == [
        (
            "openapi_mfds_get_drug_indication",
            {"drug_name": "타이레놀", "num_rows": 3},
        )
    ]


async def test_retrieval_uses_citation_specific_fragment(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "I10"})]),
            finalization(items=[{"cite_uid": "one", "relevance_score": 1}]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {
            LOOKUP_ALIAS: MCPCallResult(
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


async def test_retrieval_rejects_unobserved_finalizer_uid(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "I10"})]),
            finalization(items=[{"cite_uid": "stale:uid", "relevance_score": 1}]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"current:uid"}')},
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert caught.value.code == "retrieval_finalize_unobserved_uid"


@pytest.mark.parametrize(
    ("status", "items"),
    [
        ("sufficient", []),
        ("no_evidence", [{"cite_uid": "one", "relevance_score": 1}]),
        (
            "partial",
            [
                {"cite_uid": "one", "relevance_score": 1},
                {"cite_uid": "one", "relevance_score": 0.5},
            ],
        ),
    ],
)
async def test_retrieval_rejects_invalid_finalizer_cardinality(
    monkeypatch,
    status,
    items,
):
    l2 = ScriptedL2([finalization(status=status, items=items)])

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            FakeMCP(canonical_tools(), {}),
            settings(monkeypatch),
        ).retrieve("query")

    assert caught.value.code == "retrieval_finalize_invalid"


async def test_retrieval_rejects_uid_collision_across_sources(monkeypatch):
    monkeypatch.setattr(retrieval_module, "_ARTIFACT_MAX_MCP_CALLS", 2)
    second_tool = lookup_tool(SECOND_LOOKUP_ALIAS)
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("first", LOOKUP_ALIAS, {"code": "A"})]),
            L2Completion(tool_calls=[call("second", SECOND_LOOKUP_ALIAS, {"code": "B"})]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool(), second_tool),
        {
            LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"shared","text":"first"}'),
            SECOND_LOOKUP_ALIAS: MCPCallResult(content='{"cite_uid":"shared","text":"second"}'),
        },
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            mcp,
            settings(monkeypatch, max_mcp_calls=2),
        ).retrieve("KCD ADR 근거")

    assert caught.value.code == "retrieval_citation_conflict"


async def test_retrieval_records_source_unavailable_without_fabricating_evidence(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "A"})]),
            finalization(status="no_evidence", items=[], note="source failed"),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content="{}", is_error=True)},
    )

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert result.status == "no_evidence"
    assert result.items == []
    assert result.execution_status == "source_unavailable"
    assert result.semantic_reason == "no_match"


async def test_retrieval_maps_partial_status_to_coverage_gap(monkeypatch):
    l2 = ScriptedL2([finalization(status="partial", items=[], note="incomplete")])

    result = await RetrievalEngine(
        l2,
        FakeMCP(canonical_tools(), {}),
        settings(monkeypatch),
    ).retrieve("query")

    assert result.status == "partial"
    assert result.semantic_reason == "coverage_gap"
    assert result.execution_status == "ok"


async def test_retrieval_requires_finalizer_before_turn_limit(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "A"})]),
            L2Completion(tool_calls=[call("not-final", LOOKUP_ALIAS, {"code": "B"})]),
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool()),
        {LOOKUP_ALIAS: MCPCallResult(content="{}")},
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            mcp,
            settings(monkeypatch, max_mcp_calls=1),
        ).retrieve("query")

    assert caught.value.code == "retrieval_finalize_missing"


async def test_retrieval_rejects_mixed_local_finalizer_and_remote_call(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call("lookup", LOOKUP_ALIAS, {"code": "A"}),
                    call(
                        "finish",
                        "finalize_retrieval",
                        {"status": "no_evidence", "items": [], "note": ""},
                    ),
                ]
            )
        ]
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(
            l2,
            FakeMCP(canonical_tools(lookup_tool()), {}),
            settings(monkeypatch),
        ).retrieve("query")

    assert caught.value.code == "retrieval_planner_tool_call_count_invalid"


async def test_retrieval_rejects_parallel_remote_calls(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    call("first", LOOKUP_ALIAS, {"code": "A"}),
                    call("second", SECOND_LOOKUP_ALIAS, {"code": "B"}),
                ]
            )
        ]
    )
    mcp = FakeMCP(
        canonical_tools(lookup_tool(), lookup_tool(SECOND_LOOKUP_ALIAS)),
        {},
    )

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert caught.value.code == "retrieval_planner_tool_call_count_invalid"
    assert mcp.calls == []


async def test_retrieval_rejects_nonempty_text_mixed_with_tool_call(monkeypatch):
    l2 = ScriptedL2(
        [
            L2Completion(
                content="사용자에게 보여 줄 답변",
                tool_calls=[call("lookup", LOOKUP_ALIAS, {"code": "A"})],
            )
        ]
    )
    mcp = FakeMCP(canonical_tools(lookup_tool()), {})

    with pytest.raises(RetrievalError) as caught:
        await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert caught.value.code == "retrieval_planner_mixed_content"
    assert mcp.calls == []


@pytest.mark.parametrize(
    "invalid_arguments",
    ['{"code":"A","code":"B"}', '{"code":NaN}'],
)
async def test_retrieval_rejects_invalid_json_argument_values(
    monkeypatch,
    invalid_arguments,
):
    l2 = ScriptedL2(
        [
            L2Completion(
                tool_calls=[
                    ToolCall(
                        id="duplicate",
                        function={
                            "name": LOOKUP_ALIAS,
                            "arguments": invalid_arguments,
                        },
                    )
                ]
            ),
            finalization(status="no_evidence", items=[]),
        ]
    )
    mcp = FakeMCP(canonical_tools(lookup_tool()), {})

    result = await RetrievalEngine(l2, mcp, settings(monkeypatch)).retrieve("query")

    assert result.status == "no_evidence"
    assert result.execution_status == "schema_error"
    assert mcp.calls == []
