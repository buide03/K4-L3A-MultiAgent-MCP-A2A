from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.mcp_gateway import EvidenceGateway

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self) -> None:
        self.list_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        self.list_count += 1
        tool = SimpleNamespace(
            name="get_order",
            description="authoritative order",
            input_schema={
                "type": "object",
                "required": ["case_id", "order_id"],
                "properties": {
                    "case_id": {"type": "string"},
                    "order_id": {"type": "string"},
                },
            },
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, tool_name: str, *, arguments: dict[str, Any]) -> Any:
        self.calls.append((tool_name, arguments))
        evidence = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_12345678901234567890",
            "result_hash": "sha256:" + "0" * 64,
            "domain": "order",
            "data": {"order_status": "delivered"},
        }
        return SimpleNamespace(isError=False, structuredContent=evidence, content=[])


class SnakeCaseSession(FakeSession):
    async def call_tool(self, tool_name: str, *, arguments: dict[str, Any]) -> Any:
        result = await super().call_tool(tool_name, arguments=arguments)
        return SimpleNamespace(
            is_error=False,
            structured_content=result.structuredContent,
            content=[],
        )


def test_gateway_discovers_specs_and_always_scopes_call_by_case() -> None:
    session = FakeSession()
    gateway = EvidenceGateway(  # type: ignore[arg-type]
        session,
        Contracts(ROOT / "contracts" / "schemas"),
    )

    async def scenario() -> dict[str, Any]:
        specs = await gateway.discover_tools()
        assert specs["get_order"].input_schema["required"] == ["case_id", "order_id"]
        assert await gateway.list_tools() == ["get_order"]
        return await gateway.call(
            "get_order",
            case_id="CASE_001",
            order_id="ORDER_001",
        )

    result = asyncio.run(scenario())

    assert session.list_count == 1
    assert session.calls == [
        ("get_order", {"case_id": "CASE_001", "order_id": "ORDER_001"})
    ]
    assert result["evidence_ref"] == "ev_12345678901234567890"


def test_gateway_rejects_undiscovered_tool_before_call() -> None:
    session = FakeSession()
    gateway = EvidenceGateway(  # type: ignore[arg-type]
        session,
        Contracts(ROOT / "contracts" / "schemas"),
    )
    with pytest.raises(ValueError, match="not discovered"):
        asyncio.run(gateway.call("invented_tool", case_id="CASE_001"))
    assert session.calls == []


def test_gateway_supports_snake_case_mcp_sdk_result_fields() -> None:
    gateway = EvidenceGateway(  # type: ignore[arg-type]
        SnakeCaseSession(),
        Contracts(ROOT / "contracts" / "schemas"),
    )
    result = asyncio.run(
        gateway.call("get_order", case_id="CASE_001", order_id="ORDER_001")
    )
    assert result["domain"] == "order"
