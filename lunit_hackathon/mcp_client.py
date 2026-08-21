"""Authenticated Streamable HTTP adapter for the organizer-provided MCP server."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

import httpx2
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client

from lunit_hackathon.config import Settings
from lunit_hackathon.errors import ConfigurationError, RetrievalError
from lunit_hackathon.network_policy import require_official_mcp_endpoint
from lunit_hackathon.schemas import MCPCallResult, MCPTool

logger = logging.getLogger(__name__)


class MCPConnectionProtocol(Protocol):
    async def list_tools(self) -> list[MCPTool]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult: ...


class MCPClientProtocol(Protocol):
    def connect(self) -> AbstractAsyncContextManager[MCPConnectionProtocol]: ...


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


class SDKMCPConnection:
    def __init__(
        self,
        client: Any,
        max_tool_result_chars: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._client = client
        self._max_tool_result_chars = max_tool_result_chars
        self._semaphore = semaphore

    async def list_tools(self) -> list[MCPTool]:
        try:
            async with _concurrency_slot(self._semaphore):
                tools_by_name: dict[str, MCPTool] = {}
                observed_names: set[str] = set()
                duplicate_names: set[str] = set()
                seen_cursors: set[str] = set()
                observed_entries = 0
                cursor: Any | None = None
                for _ in range(64):
                    result = (
                        await self._client.list_tools()
                        if cursor is None
                        else await self._client.list_tools(cursor=cursor)
                    )
                    for tool in result.tools:
                        observed_entries += 1
                        raw_name = getattr(tool, "name", None)
                        safe_raw_name = (
                            raw_name
                            if isinstance(raw_name, str)
                            and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", raw_name)
                            else None
                        )
                        if safe_raw_name in duplicate_names:
                            continue
                        if safe_raw_name is not None and safe_raw_name in observed_names:
                            tools_by_name.pop(safe_raw_name, None)
                            duplicate_names.add(safe_raw_name)
                            logger.warning(
                                "mcp_discovery_tool_quarantined tool=%s reason_code=duplicate_name",
                                safe_raw_name,
                            )
                            continue
                        if safe_raw_name is not None:
                            observed_names.add(safe_raw_name)
                        normalized = _normalize_discovered_tool(tool)
                        if normalized is None:
                            continue
                        name = normalized.name
                        tools_by_name[name] = normalized
                    if observed_entries > 256:
                        raise RetrievalError(
                            "MCP tool discovery exceeded the tool bound",
                            code="mcp_tool_discovery_unbounded",
                        )
                    next_cursor = getattr(result, "next_cursor", None)
                    if next_cursor is None or str(next_cursor) == "":
                        return sorted(tools_by_name.values(), key=lambda tool: tool.name)
                    cursor_key = str(next_cursor)
                    if cursor_key in seen_cursors:
                        raise RetrievalError(
                            "MCP tool discovery repeated a pagination cursor",
                            code="mcp_tool_discovery_cursor_repeated",
                        )
                    seen_cursors.add(cursor_key)
                    cursor = next_cursor
                raise RetrievalError(
                    "MCP tool discovery exceeded the page bound",
                    code="mcp_tool_discovery_unbounded",
                )
        except Exception as error:
            if isinstance(error, RetrievalError):
                raise
            raise RetrievalError(
                "MCP tool discovery failed",
                code="mcp_tool_discovery_failed",
            ) from error

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        try:
            if self._semaphore is None:
                result = await self._client.call_tool(name, arguments)
            else:
                async with self._semaphore:
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


def _normalize_discovered_tool(tool: Any) -> MCPTool | None:
    """Quarantine one malformed discovery entry without disabling other routes."""

    raw_name: Any = None
    try:
        raw_name = tool.name
        normalized = MCPTool(
            name=raw_name,
            description=getattr(tool, "description", None) or "",
            input_schema=tool.input_schema,
        )
    except Exception:
        safe_name = (
            raw_name
            if isinstance(raw_name, str)
            and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", raw_name)
            else "invalid"
        )
        logger.warning(
            "mcp_discovery_tool_quarantined tool=%s reason_code=invalid_entry",
            safe_name,
        )
        return None
    return normalized


class MCPClient:
    """Creates one authenticated Streamable HTTP MCP session per retrieval run."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client_factory: Callable[..., AbstractAsyncContextManager[Any]] = httpx2.AsyncClient,
        transport_factory: Callable[..., AbstractAsyncContextManager[Any]] = streamable_http_client,
        client_factory: Callable[[Any], AbstractAsyncContextManager[Any]] = Client,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._settings = settings
        self._http_client_factory = http_client_factory
        self._transport_factory = transport_factory
        self._client_factory = client_factory
        self._semaphore = semaphore

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[MCPConnectionProtocol]:
        if not self._settings.mcp_url:
            raise RetrievalError("LUNIT_MCP_URL is not configured")
        # Enforce the exact organizer endpoint at the last boundary before an
        # HTTP-capable object is constructed.  This also covers unvalidated
        # Settings.model_copy callers.
        require_official_mcp_endpoint(self._settings.mcp_url)
        if not self._settings.api_key:
            raise ConfigurationError("LUNIT_FM_API_KEY is required for MCP requests")

        async with _concurrency_slot(self._semaphore):
            caller_error: BaseException | None = None
            try:
                async with self._http_client_factory(
                    headers={"Authorization": f"Bearer {self._settings.api_key}"},
                    timeout=httpx2.Timeout(self._settings.request_timeout_seconds),
                    follow_redirects=False,
                    trust_env=False,
                ) as http_client:
                    transport = self._transport_factory(
                        self._settings.mcp_url,
                        http_client=http_client,
                    )
                    async with self._client_factory(transport) as client:
                        try:
                            # The process-wide slot covers handshake, discovery and all
                            # calls in this request-scoped MCP session. Passing no nested
                            # semaphore avoids deadlocking a one-slot deployment.
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


@asynccontextmanager
async def _concurrency_slot(
    semaphore: asyncio.Semaphore | None,
) -> AsyncIterator[None]:
    if semaphore is None:
        yield
        return
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


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
                nested = json.loads(item, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError):
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
                    allow_nan=False,
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
                nested = json.loads(item, parse_constant=_reject_nonfinite_json)
            except (json.JSONDecodeError, ValueError):
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
        payload = json.loads(content, parse_constant=_reject_nonfinite_json)
    except (json.JSONDecodeError, ValueError):
        payload = {"raw_content": content}
    resolved_cite_uids = cite_uids if cite_uids is not None else collect_cite_uids(payload)
    return _bounded_json(payload, maximum, resolved_cite_uids)


def _bounded_json(payload: Any, maximum: int, cite_uids: list[str]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
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
            allow_nan=False,
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
            allow_nan=False,
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
