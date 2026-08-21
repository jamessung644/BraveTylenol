from contextlib import asynccontextmanager
from types import SimpleNamespace

from harness.mcp_client import MCPClient, SDKMCPConnection
from harness.schemas import MCPTool


def test_mcp_tool_converts_to_openai_function_tool():
    tool = MCPTool(
        name="kcd_get_name",
        description="KCD 질병명 조회",
        input_schema={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    )

    assert tool.as_openai_tool() == {
        "type": "function",
        "function": {
            "name": "kcd_get_name",
            "description": "KCD 질병명 조회",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }


async def test_connection_serializes_sdk_content_deterministically_and_truncates():
    sdk_client = FakeSDKClient(
        call_result=SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="plain text"),
                SimpleNamespace(type="image", data="base64-data", mime_type="image/png"),
                SimpleNamespace(type="resource_link", uri="file:///guide", name="guide", mime_type="text/markdown"),
                SimpleNamespace(
                    type="resource",
                    resource=SimpleNamespace(uri="file:///evidence", mime_type="application/json", text='{"z": 1}'),
                ),
            ],
            structured_content={"z": 1, "a": [2]},
            is_error=True,
        )
    )
    connection = SDKMCPConnection(sdk_client, max_tool_result_chars=1_000)

    result = await connection.call_tool("lookup", {"query": "혈압"})

    assert sdk_client.called == ("lookup", {"query": "혈압"})
    assert result.is_error is True
    assert result.content == (
        '{"content":[{"text":"plain text","type":"text"},'
        '{"mimeType":"image/png","type":"image"},'
        '{"mimeType":"text/markdown","name":"guide","type":"resource_link","uri":"file:///guide"},'
        '{"resource":{"mimeType":"application/json","text":"{\\"z\\": 1}","uri":"file:///evidence"},"type":"resource"}],'
        '"is_error":true,"structured_content":{"a":[2],"z":1}}'
    )

    truncated = SDKMCPConnection(sdk_client, max_tool_result_chars=80)
    short_result = await truncated.call_tool("lookup", {})
    assert len(short_result.content) == 80
    assert short_result.content.endswith("...[truncated]")


async def test_mcp_client_connects_with_bearer_http_client_and_delegates(settings_with_key):
    events = []
    sdk_client = FakeSDKClient(tools=[SimpleNamespace(name="zeta", description=None, input_schema={"type": "object"}), SimpleNamespace(name="alpha", description="A", input_schema={"type": "object", "properties": {}})])

    @asynccontextmanager
    async def fake_http_client(**kwargs):
        events.append(("http", kwargs))
        yield "http-client"

    @asynccontextmanager
    async def transport_context():
        yield "transport"

    def fake_transport(url, *, http_client):
        events.append(("transport", url, http_client))
        return transport_context()

    @asynccontextmanager
    async def fake_sdk_client(transport):
        events.append(("client", transport))
        yield sdk_client

    adapter = MCPClient(
        settings_with_key,
        http_client_factory=fake_http_client,
        transport_factory=fake_transport,
        client_factory=fake_sdk_client,
    )

    async with adapter.connect() as connection:
        tools = await connection.list_tools()
        result = await connection.call_tool("alpha", {"code": "A00"})

    assert [tool.name for tool in tools] == ["alpha", "zeta"]
    assert result.content == '{"content":[],"is_error":false}'
    assert sdk_client.called == ("alpha", {"code": "A00"})
    assert events[0][1]["headers"] == {"Authorization": "Bearer test-key"}
    assert events[0][1]["follow_redirects"] is True
    assert events[1] == ("transport", settings_with_key.mcp_url, "http-client")
    assert events[2][0] == "client"


class FakeSDKClient:
    def __init__(self, *, tools=None, call_result=None):
        self._tools = tools or []
        self._call_result = call_result or SimpleNamespace(content=[], is_error=False, structured_content=None)
        self.called = None

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        self.called = (name, arguments)
        return self._call_result
