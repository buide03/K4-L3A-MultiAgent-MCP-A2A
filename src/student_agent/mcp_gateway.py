from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .contracts import Contracts


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str | None
    input_schema: dict[str, Any]


class MCPToolError(RuntimeError):
    """The Gateway accepted the MCP request but the selected tool failed."""


class MCPAuthorizationError(MCPToolError):
    """A tool failure that must abort rather than degrade into missing evidence."""


class EvidenceGateway:
    def __init__(self, session: ClientSession, contracts: Contracts) -> None:
        self._session = session
        self._contracts = contracts
        self._tool_specs: dict[str, ToolSpec] | None = None

    async def discover_tools(self, *, refresh: bool = False) -> dict[str, ToolSpec]:
        if self._tool_specs is None or refresh:
            response = await self._session.list_tools()
            specs: dict[str, ToolSpec] = {}
            for tool in response.tools:
                input_schema = getattr(tool, "input_schema", None)
                if input_schema is None:
                    input_schema = getattr(tool, "inputSchema", None)
                if not isinstance(input_schema, dict):
                    raise ValueError(f"MCP tool {tool.name} has no valid input schema")
                specs[tool.name] = ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    input_schema=dict(input_schema),
                )
            self._tool_specs = specs
        return dict(self._tool_specs)

    async def list_tools(self) -> list[str]:
        return sorted(await self.discover_tools())

    async def call(self, tool_name: str, *, case_id: str, **arguments: Any) -> dict[str, Any]:
        if "case_id" in arguments:
            raise ValueError("case_id must be supplied only through the dedicated argument")
        if tool_name not in await self.discover_tools():
            raise ValueError(f"MCP tool was not discovered: {tool_name}")
        payload = {"case_id": case_id, **arguments}
        result = await self._session.call_tool(tool_name, arguments=payload)
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            message = " ".join(
                block.text for block in result.content if getattr(block, "text", None)
            )
            detail = message or "unknown error"
            lowered = detail.lower()
            error_type = (
                MCPAuthorizationError
                if any(
                    marker in lowered
                    for marker in ("401", "403", "unauthorized", "forbidden")
                )
                else MCPToolError
            )
            raise error_type(f"MCP tool {tool_name} failed: {detail}")
        evidence = getattr(result, "structuredContent", None)
        if evidence is None:
            evidence = getattr(result, "structured_content", None)
        if evidence is None:
            text_blocks = [block.text for block in result.content if getattr(block, "text", None)]
            if len(text_blocks) != 1:
                raise ValueError(f"MCP tool {tool_name} did not return one evidence object")
            evidence = json.loads(text_blocks[0])
        self._contracts.validate_evidence(evidence, f"MCP tool {tool_name}")
        return evidence


@asynccontextmanager
async def connect_gateway(
    endpoint: str, team_api_key: str, contracts: Contracts
) -> AsyncIterator[EvidenceGateway]:
    headers = {"Authorization": f"Bearer {team_api_key}"}
    timeout = httpx2.Timeout(300.0, connect=30.0, write=30.0, pool=30.0)
    last_error: BaseException | None = None
    for attempt in range(3):
        stack = AsyncExitStack()
        try:
            # Transport retries cover TCP failures before a request is sent. The outer loop
            # also covers a transient JSON-RPC failure during session initialization.
            transport = httpx2.AsyncHTTPTransport(retries=2)
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(
                    headers=headers,
                    timeout=timeout,
                    transport=transport,
                )
            )
            read_stream, write_stream = await stack.enter_async_context(
                streamable_http_client(endpoint, http_client=http_client)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except BaseException as exc:
            last_error = exc
            with suppress(BaseException):
                await stack.aclose()
            if attempt == 2 or _contains_auth_failure(exc):
                raise
            await asyncio.sleep(0.25 * (attempt + 1))
            continue
        try:
            yield EvidenceGateway(session, contracts)
        finally:
            await stack.aclose()
        return
    if last_error is not None:  # pragma: no cover - defensive exhaustiveness
        raise last_error


def _contains_auth_failure(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(_contains_auth_failure(item) for item in exc.exceptions)
    message = str(exc).lower()
    return any(marker in message for marker in ("401", "403", "unauthorized", "forbidden"))
