from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.mcp_gateway import ToolSpec
from student_agent.trace import TraceWriter
from student_agent.workflow import (
    WorkflowNodes,
    WorkflowState,
    _merge_specialist_reports,
    _targeted_refetch,
    as_final_output,
    build_workflow,
    require_refetch_budget,
    require_tool_permission,
    solve_case,
)

ROOT = Path(__file__).resolve().parents[1]


def minimal_output() -> dict[str, Any]:
    return {
        "schema_version": "day09-l3a-output-v2",
        "case_id": "CASE_001",
        "assessment": {
            "primary_issue": "insufficient_evidence",
            "case_status": "needs_investigation",
            "confidence": 0.0,
        },
        "affected_entities": {
            "order_ids": [],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "root_cause_analysis": {"ranked_causes": [], "responsible_parties": []},
        "evidence_refs": [],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0,
            "refund_lines": [],
        },
        "resolution_actions": [],
    }


def test_langgraph_runs_the_bounded_pipeline() -> None:
    visited: list[str] = []

    async def triage(state: WorkflowState) -> dict[str, Any]:
        visited.append("triage")
        return {"next_node": "order_specialist"}

    async def order(state: WorkflowState) -> dict[str, Any]:
        visited.append("order_specialist")
        return {"next_node": "cross_verification"}

    async def unused_specialist(state: WorkflowState) -> dict[str, Any]:
        raise AssertionError("dynamic routing invoked an unplanned specialist")

    async def cross_verification(state: WorkflowState) -> dict[str, Any]:
        visited.append("cross_verification")
        return {"next_node": "policy_specialist"}

    async def policy(state: WorkflowState) -> dict[str, Any]:
        visited.append("policy_specialist")
        return {}

    async def judicial(state: WorkflowState) -> dict[str, Any]:
        visited.append("judicial_engine")
        return {}

    async def builder(state: WorkflowState) -> dict[str, Any]:
        visited.append("deterministic_builder")
        return {"output_candidate": minimal_output()}

    async def verifier(state: WorkflowState) -> dict[str, Any]:
        visited.append("deterministic_verifier")
        return {"final_output": state["output_candidate"], "next_node": "end"}

    graph = build_workflow(
        WorkflowNodes(
            triage=triage,
            order_specialist=order,
            payment_specialist=unused_specialist,
            shipment_specialist=unused_specialist,
            cross_verification=cross_verification,
            policy_specialist=policy,
            judicial_engine=judicial,
            deterministic_builder=builder,
            deterministic_verifier=verifier,
        )
    )
    result = asyncio.run(
        graph.ainvoke({"case": {"case_id": "CASE_001"}, "case_id": "CASE_001"})
    )

    assert visited == [
        "triage",
        "order_specialist",
        "cross_verification",
        "policy_specialist",
        "judicial_engine",
        "deterministic_builder",
        "deterministic_verifier",
    ]
    assert as_final_output(result) == minimal_output()


def test_tool_permissions_are_deny_by_default() -> None:
    require_tool_permission("payment-agent", "get_order_payments")
    with pytest.raises(PermissionError):
        require_tool_permission("payment-agent", "get_shipment_summary")
    with pytest.raises(PermissionError):
        require_tool_permission("unknown-agent", "get_order")


def test_targeted_refetch_is_limited_to_one_round() -> None:
    require_refetch_budget({"refetch_count": 0})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        require_refetch_budget({"refetch_count": 1})


def test_targeted_refetch_merges_new_evidence_and_resolves_the_gap() -> None:
    assert _targeted_refetch(["TOOL_CALL_FAILED:get_refund_timeline"]) == (
        "payment_specialist",
        "payment-agent",
        "get_refund_timeline",
    )
    previous = {
        "actor": "payment-agent",
        "status": "partial",
        "facts": [{"tool_name": "get_order_payments", "evidence_refs": ["ev_old"]}],
        "anomaly_flags": [],
        "missing_evidence": ["TOOL_CALL_FAILED:get_refund_timeline"],
        "evidence_refs": ["ev_old"],
    }
    current = {
        "actor": "payment-agent",
        "status": "completed",
        "facts": [{"tool_name": "get_refund_timeline", "evidence_refs": ["ev_new"]}],
        "anomaly_flags": [],
        "missing_evidence": [],
        "evidence_refs": ["ev_new"],
    }
    merged = _merge_specialist_reports(previous, current)
    assert merged["status"] == "completed"
    assert merged["missing_evidence"] == []
    assert merged["evidence_refs"] == ["ev_old", "ev_new"]
    assert len(merged["facts"]) == 2


class CaseGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        order_tools = {
            "get_order",
            "get_order_items",
            "get_order_payments",
            "get_payment_timeline",
            "get_refund_timeline",
        }
        self.specs = {
            name: ToolSpec(
                name,
                name,
                {
                    "type": "object",
                    "required": ["case_id", "order_id"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "order_id": {"type": "string"},
                    },
                },
            )
            for name in order_tools
        }
        self.specs["get_policy"] = ToolSpec(
            "get_policy",
            "policy",
            {
                "type": "object",
                "required": ["case_id", "policy_version", "issue_code"],
                "properties": {
                    "case_id": {"type": "string"},
                    "policy_version": {"type": "string"},
                    "issue_code": {"type": "string"},
                },
            },
        )

    async def discover_tools(self, *, refresh: bool = False) -> dict[str, ToolSpec]:
        return dict(self.specs)

    async def call(
        self, tool_name: str, *, case_id: str, **arguments: Any
    ) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        if tool_name == "get_order":
            data = {
                "order_id": arguments["order_id"],
                "order_status": "canceled",
                "paid": True,
            }
        elif tool_name == "get_order_items":
            data = {
                "order_id": arguments["order_id"],
                "item_id": "ITEM_001",
                "seller_id": "SELLER_001",
            }
        elif tool_name == "get_order_payments":
            data = {
                "order_id": arguments["order_id"],
                "payment_reference": "PAY_001",
                "captured_total_brl": 100.0,
            }
        elif tool_name == "get_payment_timeline":
            data = {"payment_status": "captured"}
        elif tool_name == "get_refund_timeline":
            data = {
                "refunded_total_brl": 0.0,
                "refundable_total_brl": 100.0,
            }
        else:
            data = {
                "policy_id": arguments["policy_version"],
                "responsible_party": "platform",
            }
        domain_by_tool = {
            "get_order": "order",
            "get_order_items": "item",
            "get_order_payments": "payment",
            "get_payment_timeline": "payment",
            "get_refund_timeline": "refund",
            "get_policy": "policy",
        }
        suffix = tool_name.removeprefix("get_") + "_12345678901234567890"
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_" + suffix,
            "result_hash": "sha256:" + "0" * 64,
            "domain": domain_by_tool[tool_name],
            "data": data,
        }


def test_released_case_shape_runs_end_to_end_through_langgraph(tmp_path: Path) -> None:
    contracts = Contracts(ROOT / "contracts" / "schemas")
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    gateway = CaseGateway()
    case = {
        "case_id": "L3A_CASE_001",
        "opened_at": "2018-01-01T09:00:00-03:00",
        "customer_request": {
            "language": "vi",
            "message": "customer statement is not evidence",
            "claimed_order_id": "ORDER_001",
            "claims": [
                {"claim_id": "claim-001-a", "topic": "canceled_order_paid"},
                {"claim_id": "claim-001-b", "topic": "requested_full_refund"},
            ],
        },
        "policy_version": "EC_POLICY_V1",
    }

    output = asyncio.run(solve_case(case, gateway, trace))  # type: ignore[arg-type]

    contracts.validate_output(output, "graph output")
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0
    assert output["claim_assessments"][0]["verdict"] == "supported"
    assert all(call[1] == "L3A_CASE_001" for call in gateway.calls)
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert any(event["event_type"] == "policy_decided" for event in events)
    assert any(event["event_type"] == "verification_completed" for event in events)
