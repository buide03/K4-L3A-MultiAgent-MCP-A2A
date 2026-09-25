from __future__ import annotations

import asyncio
from typing import Any

import pytest

from student_agent.mcp_gateway import MCPToolError, ToolSpec
from student_agent.specialists import (
    ORDER_SPECIALIST,
    EvidenceCollector,
    ToolArgumentBinder,
)


def evidence(ref: str = "ev_12345678901234567890") -> dict[str, Any]:
    return {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": ref,
        "result_hash": "sha256:" + "0" * 64,
        "domain": "order",
        "data": {"order_status": "delivered"},
    }


class FakeTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, **event: Any) -> dict[str, Any]:
        self.events.append(event)
        return event


class FakeGateway:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.specs = {
            "get_order": ToolSpec(
                "get_order",
                "authoritative order",
                {
                    "type": "object",
                    "required": ["case_id", "order_id"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "order_id": {"type": "string"},
                    },
                },
            )
        }
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.fail_once = fail_once

    async def discover_tools(self, *, refresh: bool = False) -> dict[str, ToolSpec]:
        return dict(self.specs)

    async def call(
        self, tool_name: str, *, case_id: str, **arguments: Any
    ) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        if self.fail_once:
            self.fail_once = False
            raise ConnectError("temporary network failure")
        return evidence()


class ToolFailureGateway(FakeGateway):
    async def call(
        self, tool_name: str, *, case_id: str, **arguments: Any
    ) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        raise MCPToolError(f"MCP tool {tool_name} failed")


class ConnectError(Exception):
    pass


def test_argument_binder_uses_structured_fields_not_customer_prose() -> None:
    spec = FakeGateway().specs["get_order"]
    from_message = ToolArgumentBinder({"message": "please inspect order_id=guessed"})
    assert from_message.bind(spec) == []

    structured = ToolArgumentBinder({"order_id": "ORDER_001"})
    assert structured.bind(spec) == [{"order_id": "ORDER_001"}]


def test_collector_preserves_case_scope_evidence_and_trace() -> None:
    gateway = FakeGateway()
    trace = FakeTrace()
    collector = EvidenceCollector(
        case_id="CASE_001",
        gateway=gateway,  # type: ignore[arg-type]
        trace=trace,  # type: ignore[arg-type]
        retry_delay_seconds=0,
    )

    async def scenario() -> None:
        first = await collector.fetch(
            actor="order-item-agent",
            tool_name="get_order",
            arguments={"order_id": "ORDER_001"},
        )
        second = await collector.fetch(
            actor="order-item-agent",
            tool_name="get_order",
            arguments={"order_id": "ORDER_001"},
        )
        assert first == second == evidence()
        collector.consume(
            actor="order-item-agent",
            tool_name="get_order",
            evidence=first,
            decision_code="ORDER_STATUS_VERIFIED",
        )
        collector.consume(
            actor="order-item-agent",
            tool_name="get_order",
            evidence=first,
            decision_code="ORDER_STATUS_VERIFIED",
        )

    asyncio.run(scenario())

    assert gateway.calls == [("get_order", "CASE_001", {"order_id": "ORDER_001"})]
    assert collector.ledger.consumed_refs() == ["ev_12345678901234567890"]
    assert trace.events == [
        {
            "case_id": "CASE_001",
            "event_type": "tool_result_consumed",
            "actor": "order-item-agent",
            "decision_code": "ORDER_STATUS_VERIFIED",
            "tool_name": "get_order",
            "evidence_refs": ["ev_12345678901234567890"],
        }
    ]


def test_collector_retries_one_transient_failure_with_identical_arguments() -> None:
    gateway = FakeGateway(fail_once=True)
    collector = EvidenceCollector(
        case_id="CASE_001",
        gateway=gateway,  # type: ignore[arg-type]
        trace=FakeTrace(),  # type: ignore[arg-type]
        retry_delay_seconds=0,
    )
    result = asyncio.run(
        collector.fetch(
            actor="order-item-agent",
            tool_name="get_order",
            arguments={"order_id": "ORDER_001"},
        )
    )

    assert result == evidence()
    assert gateway.calls == [
        ("get_order", "CASE_001", {"order_id": "ORDER_001"}),
        ("get_order", "CASE_001", {"order_id": "ORDER_001"}),
    ]


def test_collector_rejects_scope_and_permission_violations() -> None:
    collector = EvidenceCollector(
        case_id="CASE_001",
        gateway=FakeGateway(),  # type: ignore[arg-type]
        trace=FakeTrace(),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must not pass case_id"):
        asyncio.run(
            collector.fetch(
                actor="order-item-agent",
                tool_name="get_order",
                arguments={"case_id": "CASE_002", "order_id": "ORDER_001"},
            )
        )
    with pytest.raises(PermissionError):
        asyncio.run(
            collector.fetch(
                actor="payment-agent",
                tool_name="get_order",
                arguments={"order_id": "ORDER_001"},
            )
        )


def test_order_specialist_queries_only_bindable_authoritative_tools() -> None:
    gateway = FakeGateway()
    trace = FakeTrace()
    collector = EvidenceCollector(
        case_id="CASE_001",
        gateway=gateway,  # type: ignore[arg-type]
        trace=trace,  # type: ignore[arg-type]
        retry_delay_seconds=0,
    )
    report = asyncio.run(
        ORDER_SPECIALIST.investigate(
            case={"case_id": "CASE_001", "order_id": "ORDER_001"},
            collector=collector,
            requested_tools=("get_order",),
        )
    )

    assert gateway.calls == [("get_order", "CASE_001", {"order_id": "ORDER_001"})]
    assert report["evidence_refs"] == ["ev_12345678901234567890"]
    assert report["facts"][0]["data"] == {"order_status": "delivered"}
    assert report["status"] == "completed"
    assert report["missing_evidence"] == []
    assert trace.events[0]["event_type"] == "tool_result_consumed"


def test_specialist_preserves_tool_failure_as_missing_evidence() -> None:
    gateway = ToolFailureGateway()
    collector = EvidenceCollector(
        case_id="CASE_001",
        gateway=gateway,  # type: ignore[arg-type]
        trace=FakeTrace(),  # type: ignore[arg-type]
        retry_delay_seconds=0,
    )
    report = asyncio.run(
        ORDER_SPECIALIST.investigate(
            case={"case_id": "CASE_001", "order_id": "ORDER_001"},
            collector=collector,
            requested_tools=("get_order",),
        )
    )
    assert report["status"] == "partial"
    assert report["facts"] == []
    assert report["missing_evidence"] == ["TOOL_CALL_FAILED:get_order"]
