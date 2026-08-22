import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import RetrievalError
from lunit_hackathon.mcp_client import MCPClient, SDKMCPConnection
from lunit_hackathon.network_policy import OFFICIAL_MCP_URL


class FakeSDKClient:
    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="z_tool",
                    description=None,
                    input_schema={"type": "object"},
                ),
                SimpleNamespace(
                    name="a_tool",
                    description="first",
                    input_schema={"type": "object"},
                ),
            ]
        )

    async def call_tool(self, name, arguments):
        assert name == "a_tool"
        assert arguments == {"q": "test"}
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="result")],
            is_error=False,
            structured_content={"cite_uid": "source:1"},
        )


async def test_sdk_connection_normalizes_tools_and_results():
    connection = SDKMCPConnection(FakeSDKClient(), max_tool_result_chars=1_000)

    tools = await connection.list_tools()
    result = await connection.call_tool("a_tool", {"q": "test"})

    assert [tool.name for tool in tools] == ["a_tool", "z_tool"]
    assert '"cite_uid":"source:1"' in result.content
    assert '"text":"result"' not in result.content
    assert result.cite_uids == ["source:1"]
    assert json.loads(result.citation_contents["source:1"]) == {"cite_uid": "source:1"}
    assert not result.is_error


async def test_sdk_connection_collects_all_paginated_tool_pages():
    class PaginatedSDKClient:
        def __init__(self):
            self.cursors = []

        async def list_tools(self, *, cursor=None):
            self.cursors.append(cursor)
            if cursor is None:
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="z_tool",
                            description=None,
                            input_schema={"type": "object"},
                        )
                    ],
                    next_cursor="page-2",
                )
            assert cursor == "page-2"
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="a_tool",
                        description=None,
                        input_schema={"type": "object"},
                    )
                ],
                next_cursor=None,
            )

    sdk = PaginatedSDKClient()

    tools = await SDKMCPConnection(sdk, max_tool_result_chars=1_000).list_tools()

    assert [tool.name for tool in tools] == ["a_tool", "z_tool"]
    assert sdk.cursors == [None, "page-2"]


async def test_sdk_connection_quarantines_malformed_unrelated_entry():
    class MixedSDKClient:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="good_tool",
                        description="good",
                        input_schema={"type": "object"},
                    ),
                    SimpleNamespace(
                        name="bad tool name",
                        description="bad",
                        input_schema=["not", "an", "object"],
                    ),
                ]
            )

    tools = await SDKMCPConnection(
        MixedSDKClient(), max_tool_result_chars=1_000
    ).list_tools()

    assert [tool.name for tool in tools] == ["good_tool"]


async def test_sdk_connection_quarantines_all_duplicate_name_entries():
    class DuplicateSDKClient:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="good_tool",
                        description="good",
                        input_schema={"type": "object"},
                    ),
                    SimpleNamespace(
                        name="shadow_tool",
                        description="first",
                        input_schema={"type": "object"},
                    ),
                    SimpleNamespace(
                        name="shadow_tool",
                        description="second",
                        input_schema={"type": "object"},
                    ),
                ]
            )

    tools = await SDKMCPConnection(
        DuplicateSDKClient(), max_tool_result_chars=1_000
    ).list_tools()

    assert [tool.name for tool in tools] == ["good_tool"]


async def test_sdk_connection_does_not_keep_valid_half_of_ambiguous_duplicate():
    class AmbiguousSDKClient:
        async def list_tools(self):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="ambiguous_tool",
                        description="valid first entry",
                        input_schema={"type": "object"},
                    ),
                    SimpleNamespace(
                        name="ambiguous_tool",
                        description="malformed duplicate",
                        input_schema=["invalid"],
                    ),
                    SimpleNamespace(
                        name="independent_tool",
                        description="safe",
                        input_schema={"type": "object"},
                    ),
                ]
            )

    tools = await SDKMCPConnection(
        AmbiguousSDKClient(), max_tool_result_chars=1_000
    ).list_tools()

    assert [tool.name for tool in tools] == ["independent_tool"]


async def test_sdk_connection_rejects_repeated_pagination_cursor():
    class RepeatingSDKClient:
        async def list_tools(self, *, cursor=None):
            del cursor
            return SimpleNamespace(tools=[], next_cursor="same")

    with pytest.raises(RetrievalError) as caught:
        await SDKMCPConnection(
            RepeatingSDKClient(),
            max_tool_result_chars=1_000,
        ).list_tools()

    assert caught.value.code == "mcp_tool_discovery_cursor_repeated"


async def test_mcp_is_optional_and_fails_cleanly_when_url_missing(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
    settings = Settings(_env_file=None).model_copy(update={"mcp_url": None})

    with pytest.raises(RetrievalError, match="LUNIT_MCP_URL"):
        async with MCPClient(settings).connect():
            pass


async def test_mcp_passes_unentered_transport_context_to_sdk_client(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
    monkeypatch.setenv("LUNIT_MCP_URL", OFFICIAL_MCP_URL)
    settings = Settings(_env_file=None)
    state = {"transport_entered": False, "client_received": None}

    @asynccontextmanager
    async def http_client_factory(**kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer lunit_test_team_key"
        yield object()

    @asynccontextmanager
    async def transport_context():
        state["transport_entered"] = True
        yield ("read", "write", "session")

    transport = transport_context()

    def transport_factory(url, *, http_client):
        assert url == OFFICIAL_MCP_URL
        assert http_client is not None
        return transport

    @asynccontextmanager
    async def client_factory(received):
        state["client_received"] = received
        assert received is transport
        async with received:
            yield FakeSDKClient()

    async with MCPClient(
        settings,
        http_client_factory=http_client_factory,
        transport_factory=transport_factory,
        client_factory=client_factory,
    ).connect() as connection:
        assert [tool.name for tool in await connection.list_tools()] == ["a_tool", "z_tool"]

    assert state == {"transport_entered": True, "client_received": transport}


async def test_mcp_teardown_does_not_mask_caller_cancellation(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
    monkeypatch.setenv("LUNIT_MCP_URL", OFFICIAL_MCP_URL)
    settings = Settings(_env_file=None)

    @asynccontextmanager
    async def http_client_factory(**kwargs):
        del kwargs
        yield object()

    @asynccontextmanager
    async def transport_context():
        yield ("read", "write", "session")

    def transport_factory(url, *, http_client):
        del url, http_client
        return transport_context()

    @asynccontextmanager
    async def client_factory(transport):
        del transport
        try:
            yield FakeSDKClient()
        finally:
            raise RuntimeError("teardown failure")

    client = MCPClient(
        settings,
        http_client_factory=http_client_factory,
        transport_factory=transport_factory,
        client_factory=client_factory,
    )
    with pytest.raises(asyncio.CancelledError):
        async with client.connect():
            raise asyncio.CancelledError()


async def test_sdk_connection_keeps_citations_when_large_result_is_bounded():
    class LargeSDKClient(FakeSDKClient):
        async def call_tool(self, name, arguments):
            del name, arguments
            return SimpleNamespace(
                content=[],
                is_error=False,
                structured_content={
                    "results": [
                        {"cite_uid": "source:large", "text": "x" * 5_000},
                    ]
                },
            )

    result = await SDKMCPConnection(
        LargeSDKClient(),
        max_tool_result_chars=1_000,
    ).call_tool("large", {})

    assert len(result.content) <= 1_000
    assert json.loads(result.content)["truncated"] is True
    assert result.cite_uids == ["source:large"]
    assert len(result.citation_contents["source:large"]) <= 1_000


async def test_process_semaphore_bounds_entire_mcp_session_including_discovery(
    monkeypatch,
):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "lunit_test_team_key")
    settings = Settings(_env_file=None)
    state = {"active": 0, "peak": 0}

    @asynccontextmanager
    async def http_client_factory(**kwargs):
        del kwargs
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            yield object()
        finally:
            state["active"] -= 1

    @asynccontextmanager
    async def transport_context():
        yield ("read", "write", "session")

    def transport_factory(url, *, http_client):
        del url, http_client
        return transport_context()

    @asynccontextmanager
    async def client_factory(transport):
        del transport
        yield FakeSDKClient()

    client = MCPClient(
        settings,
        http_client_factory=http_client_factory,
        transport_factory=transport_factory,
        client_factory=client_factory,
        semaphore=asyncio.Semaphore(1),
    )

    async def discover() -> None:
        async with client.connect() as connection:
            await connection.list_tools()
            await asyncio.sleep(0.01)

    await asyncio.gather(discover(), discover())

    assert state == {"active": 0, "peak": 1}
