from types import SimpleNamespace

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import RetrievalError
from lunit_hackathon.mcp_client import MCPClient, SDKMCPConnection


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
    assert '"text":"result"' in result.content
    assert '"cite_uid":"source:1"' in result.content
    assert not result.is_error


async def test_mcp_is_optional_and_fails_cleanly_when_url_missing(monkeypatch):
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    settings = Settings(_env_file=None)

    with pytest.raises(RetrievalError, match="LUNIT_MCP_URL"):
        async with MCPClient(settings).connect():
            pass
