"""Optional MCP SDK adapter isolated from the rest of the application."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

import httpx2
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import ConfigurationError, RetrievalError
from lunit_hackathon.schemas import MCPCallResult, MCPTool


class MCPConnectionProtocol(Protocol):
    async def list_tools(self) -> list[MCPTool]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult: ...


class MCPClientProtocol(Protocol):
    def connect(self) -> AbstractAsyncContextManager[MCPConnectionProtocol]: ...


class SDKMCPConnection:
    def __init__(self, client: Any, max_tool_result_chars: int) -> None:
        self._client = client
        self._max_tool_result_chars = max_tool_result_chars

    async def list_tools(self) -> list[MCPTool]:
        try:
            result = await self._client.list_tools()
            tools = [
                MCPTool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.input_schema or {"type": "object"},
                )
                for tool in result.tools
            ]
            return sorted(tools, key=lambda tool: tool.name)
        except Exception as error:
            raise RetrievalError(
                "MCP tool discovery failed",
                code="mcp_tool_discovery_failed",
            ) from error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        try:
            result = await self._client.call_tool(name, arguments)
            payload: dict[str, Any] = {"is_error": bool(result.is_error)}
            structured_content = getattr(result, "structured_content", None)
            if structured_content is not None:
                payload["structured_content"] = structured_content
            else:
                payload["content"] = [_serialize_content_block(block) for block in result.content]
            cite_uids = collect_cite_uids(payload)
            citation_contents = {
                cite_uid: bound_mcp_content(
                    fragment,
                    self._max_tool_result_chars,
                    [cite_uid],
                )
                for cite_uid, fragment in extract_citation_contents(payload).items()
            }
            return MCPCallResult(
                content=_bounded_json(
                    payload,
                    self._max_tool_result_chars,
                    cite_uids,
                ),
                is_error=bool(result.is_error),
                cite_uids=cite_uids,
                citation_contents=citation_contents,
            )
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError(
                "MCP tool call failed",
                code="mcp_tool_call_failed",
            ) from error


class MCPClient:
    """Creates one authenticated Streamable HTTP MCP session per retrieval run."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client_factory: Callable[..., AbstractAsyncContextManager[Any]] = httpx2.AsyncClient,
        transport_factory: Callable[..., AbstractAsyncContextManager[Any]] = streamable_http_client,
        client_factory: Callable[[Any], AbstractAsyncContextManager[Any]] = Client,
    ) -> None:
        self._settings = settings
        self._http_client_factory = http_client_factory
        self._transport_factory = transport_factory
        self._client_factory = client_factory

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPConnectionProtocol]:
        if not self._settings.mcp_url:
            raise RetrievalError("LUNIT_MCP_URL is not configured")
        if not self._settings.api_key:
            raise ConfigurationError("LUNIT_FM_API_KEY is required for MCP requests")

        caller_error: BaseException | None = None
        try:
            async with self._http_client_factory(
                headers={"Authorization": f"Bearer {self._settings.api_key}"},
                timeout=httpx2.Timeout(self._settings.request_timeout_seconds),
                follow_redirects=True,
            ) as http_client:
                transport = self._transport_factory(
                    self._settings.mcp_url,
                    http_client=http_client,
                )
                async with self._client_factory(transport) as client:
                    try:
                        yield SDKMCPConnection(
                            client,
                            self._settings.max_tool_result_chars,
                        )
                    except BaseException as error:
                        caller_error = error
                        raise
        except BaseException as error:
            if caller_error is not None:
                raise caller_error from None
            if not isinstance(error, Exception):
                raise
            if isinstance(error, (ConfigurationError, RetrievalError)):
                raise
            raise RetrievalError(
                "MCP connection failed",
                code="mcp_connection_failed",
            ) from error


def _serialize_content_block(block: Any) -> dict[str, Any]:
    block_type = getattr(block, "type", "unknown")
    if block_type == "text":
        return {"type": "text", "text": block.text}
    if block_type in {"image", "audio"}:
        return {
            "type": str(block_type),
            "mimeType": getattr(block, "mime_type", None),
        }
    if block_type == "resource_link":
        return _without_none(
            {
                "type": "resource_link",
                "name": block.name,
                "uri": str(block.uri),
                "mimeType": getattr(block, "mime_type", None),
            }
        )
    if block_type == "resource":
        resource = block.resource
        return {
            "type": "resource",
            "resource": _without_none(
                {
                    "uri": str(resource.uri),
                    "mimeType": getattr(resource, "mime_type", None),
                    "text": getattr(resource, "text", None),
                }
            ),
        }
    return {"type": str(block_type)}


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def collect_cite_uids(value: Any) -> list[str]:
    """Collect citation IDs from structured values and JSON-encoded text blocks."""

    found: list[str] = []
    seen: set[str] = set()

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 20:
            return
        if isinstance(item, dict):
            cite_uid = item.get("cite_uid")
            if isinstance(cite_uid, str) and cite_uid not in seen:
                seen.add(cite_uid)
                found.append(cite_uid)
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, str) and item.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(item)
            except json.JSONDecodeError:
                return
            visit(nested, depth + 1)

    visit(value)
    return found


def extract_citation_contents(value: Any) -> dict[str, str]:
    """Extract the smallest structured object that owns each citation ID."""

    found: dict[str, str] = {}

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 20:
            return
        if isinstance(item, dict):
            cite_uid = item.get("cite_uid")
            if isinstance(cite_uid, str) and cite_uid not in found:
                found[cite_uid] = json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, str) and item.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(item)
            except json.JSONDecodeError:
                return
            visit(nested, depth + 1)

    visit(value)
    return found


def bound_mcp_content(
    content: str,
    maximum: int,
    cite_uids: list[str] | None = None,
) -> str:
    """Bound MCP content while keeping the result valid JSON."""

    if len(content) <= maximum:
        return content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {"raw_content": content}
    resolved_cite_uids = cite_uids if cite_uids is not None else collect_cite_uids(payload)
    return _bounded_json(payload, maximum, resolved_cite_uids)


def _bounded_json(payload: Any, maximum: int, cite_uids: list[str]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(serialized) <= maximum:
        return serialized

    bases = [
        {
            "cite_uids": cite_uids,
            "original_chars": len(serialized),
            "truncated": True,
        },
        {
            "cite_uid_count": len(cite_uids),
            "original_chars": len(serialized),
            "truncated": True,
        },
        {"original_chars": len(serialized), "truncated": True},
        {"truncated": True},
    ]
    base = bases[-1]
    for candidate in bases:
        empty = json.dumps(
            {**candidate, "preview": ""},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(empty) <= maximum:
            base = candidate
            break

    def encode(preview: str) -> str:
        return json.dumps(
            {**base, "preview": preview},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    best = encode("")
    low = 0
    high = len(serialized)
    while low <= high:
        midpoint = (low + high) // 2
        candidate = encode(serialized[:midpoint])
        if len(candidate) <= maximum:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best
