from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from .contracts import Contracts
from .mcp_gateway import EvidenceGateway
from .policy_engine import build_l3a_output, cross_verify_reports, propose_decision
from .specialists import (
    ORDER_SPECIALIST,
    PAYMENT_SPECIALIST,
    POLICY_SPECIALIST,
    SHIPMENT_SPECIALIST,
    TOOL_PERMISSIONS,
    EvidenceCollector,
    SpecialistAgent,
)
from .trace import TraceWriter
from .verifier import DeterministicVerifier

NodeName = Literal[
    "order_specialist",
    "payment_specialist",
    "shipment_specialist",
    "cross_verification",
    "policy_specialist",
    "judicial_engine",
    "deterministic_builder",
    "deterministic_verifier",
    "end",
]

class WorkflowState(TypedDict, total=False):
    """Private graph state; none of these fields are public output fields."""

    case: dict[str, Any]
    case_id: str
    specialist_plan: tuple[str, ...]
    tool_plan: dict[str, tuple[str, ...]]
    completed_specialists: tuple[str, ...]
    evidence_ledger: tuple[dict[str, Any], ...]
    specialist_reports: dict[str, dict[str, Any]]
    normalized_facts: dict[str, Any]
    data_conflicts: tuple[dict[str, Any], ...]
    missing_evidence: tuple[str, ...]
    refetch_count: int
    policy_decision: dict[str, Any]
    judicial_proposal: dict[str, Any]
    output_candidate: dict[str, Any]
    final_output: dict[str, Any]
    next_node: NodeName


GraphUpdate = dict[str, Any]
GraphNode = Callable[[WorkflowState], Awaitable[GraphUpdate]]


@dataclass(frozen=True)
class WorkflowNodes:
    """Business nodes injected into the fixed, contract-safe graph topology."""

    triage: GraphNode
    order_specialist: GraphNode
    payment_specialist: GraphNode
    shipment_specialist: GraphNode
    cross_verification: GraphNode
    policy_specialist: GraphNode
    judicial_engine: GraphNode
    deterministic_builder: GraphNode
    deterministic_verifier: GraphNode


def _route(state: WorkflowState) -> NodeName:
    try:
        return state["next_node"]
    except KeyError as exc:
        raise RuntimeError("LangGraph node did not select next_node") from exc


def _route_after_verification(state: WorkflowState) -> Literal["end"]:
    if state.get("next_node") != "end":
        raise RuntimeError("deterministic verifier must terminate the graph")
    if not isinstance(state.get("final_output"), dict):
        raise RuntimeError("deterministic verifier did not produce final_output")
    return "end"


def build_workflow(nodes: WorkflowNodes) -> Any:
    """Compile the bounded A2A state machine.

    Business nodes may choose only edges declared here. Cross-verification can route back to a
    specialist for one targeted refetch; the node implementation must enforce refetch_count <= 1.
    """

    graph = StateGraph(WorkflowState)
    graph.add_node("triage", nodes.triage)
    graph.add_node("order_specialist", nodes.order_specialist)
    graph.add_node("payment_specialist", nodes.payment_specialist)
    graph.add_node("shipment_specialist", nodes.shipment_specialist)
    graph.add_node("cross_verification", nodes.cross_verification)
    graph.add_node("policy_specialist", nodes.policy_specialist)
    graph.add_node("judicial_engine", nodes.judicial_engine)
    graph.add_node("deterministic_builder", nodes.deterministic_builder)
    graph.add_node("deterministic_verifier", nodes.deterministic_verifier)

    graph.add_edge(START, "triage")
    specialist_routes: dict[NodeName, str] = {
        "order_specialist": "order_specialist",
        "payment_specialist": "payment_specialist",
        "shipment_specialist": "shipment_specialist",
        "cross_verification": "cross_verification",
        "policy_specialist": "policy_specialist",
        "judicial_engine": "judicial_engine",
        "deterministic_builder": "deterministic_builder",
        "deterministic_verifier": "deterministic_verifier",
        "end": END,
    }
    graph.add_conditional_edges("triage", _route, specialist_routes)
    for specialist in ("order_specialist", "payment_specialist", "shipment_specialist"):
        graph.add_conditional_edges(specialist, _route, specialist_routes)
    graph.add_conditional_edges("cross_verification", _route, specialist_routes)
    graph.add_edge("policy_specialist", "judicial_engine")
    graph.add_edge("judicial_engine", "deterministic_builder")
    graph.add_edge("deterministic_builder", "deterministic_verifier")
    graph.add_conditional_edges(
        "deterministic_verifier",
        _route_after_verification,
        {"end": END},
    )
    return graph.compile()


def require_tool_permission(actor: str, tool_name: str) -> None:
    """Deny an MCP call unless the architecture explicitly grants it."""

    allowed = TOOL_PERMISSIONS.get(actor)
    if allowed is None or tool_name not in allowed:
        raise PermissionError(f"actor {actor!r} is not allowed to call {tool_name!r}")


def require_refetch_budget(state: WorkflowState) -> None:
    """Fail closed before a second cross-verification refetch round."""

    if state.get("refetch_count", 0) >= 1:
        raise RuntimeError("targeted MCP refetch budget exhausted")


def make_specialist_node(agent: SpecialistAgent, collector: EvidenceCollector) -> GraphNode:
    """Adapt an evidence specialist to LangGraph state and observable handoff events."""

    actor_to_node: dict[str, NodeName] = {
        "order-item-agent": "order_specialist",
        "payment-agent": "payment_specialist",
        "shipment-agent": "shipment_specialist",
        "policy-agent": "policy_specialist",
    }
    current_node = actor_to_node.get(agent.actor)
    if current_node is None:
        raise ValueError(f"specialist actor is not routable: {agent.actor}")

    async def run(state: WorkflowState) -> GraphUpdate:
        case = state.get("case")
        if not isinstance(case, dict) or case.get("case_id") != collector.case_id:
            raise ValueError("LangGraph state is outside the specialist collector case scope")
        collector.trace.emit(
            case_id=collector.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=agent.actor,
            decision_code="SPECIALIST_REVIEW_REQUIRED",
        )
        reports = dict(state.get("specialist_reports", {}))
        report = await agent.investigate(
            case=case,
            collector=collector,
            context=reports,
            requested_tools=state.get("tool_plan", {}).get(agent.actor, agent.tools),
        )
        previous = reports.get(agent.actor)
        reports[agent.actor] = (
            _merge_specialist_reports(previous, report)
            if isinstance(previous, Mapping)
            else report
        )
        report = reports[agent.actor]
        completed = tuple((*state.get("completed_specialists", ()), current_node))
        next_node: NodeName = "cross_verification"
        for planned in state.get("specialist_plan", ()):
            if planned not in completed:
                next_node = cast(NodeName, planned)
                break
        target = "cross-verification-node" if next_node == "cross_verification" else "coordinator"
        collector.trace.emit(
            case_id=collector.case_id,
            event_type="handoff",
            actor=agent.actor,
            target=target,
            decision_code=(
                "SPECIALIST_REVIEW_PARTIAL"
                if report["missing_evidence"]
                else "SPECIALIST_REVIEW_COMPLETED"
            ),
            evidence_refs=report["evidence_refs"][:20],
        )
        return {
            "specialist_reports": reports,
            "completed_specialists": completed,
            "evidence_ledger": collector.ledger.snapshot(),
            "next_node": next_node,
        }

    return run


def make_cross_verification_node(trace: TraceWriter) -> GraphNode:
    async def run(state: WorkflowState) -> GraphUpdate:
        verified = cross_verify_reports(state.get("specialist_reports", {}))
        refetch = _targeted_refetch(verified["missing_evidence"])
        if refetch is not None and state.get("refetch_count", 0) < 1:
            node, actor, tool_name = refetch
            tool_plan = dict(state.get("tool_plan", {}))
            tool_plan[actor] = (tool_name,)
            trace.emit(
                case_id=state["case_id"],
                event_type="handoff",
                actor="cross-verification-node",
                target=actor,
                decision_code="TARGETED_REFETCH_REQUIRED",
                attributes={"tool_name": tool_name},
            )
            return {
                "normalized_facts": verified["normalized_facts"],
                "data_conflicts": tuple(verified["data_conflicts"]),
                "missing_evidence": tuple(verified["missing_evidence"]),
                "refetch_count": state.get("refetch_count", 0) + 1,
                "tool_plan": tool_plan,
                "next_node": node,
            }
        trace.emit(
            case_id=state["case_id"],
            event_type="handoff",
            actor="cross-verification-node",
            target="policy-agent",
            decision_code=(
                "SOURCE_CONFLICT_DETECTED"
                if verified["data_conflicts"]
                else "FACTS_CROSS_VERIFIED"
            ),
            evidence_refs=verified["evidence_refs"][:20],
        )
        return {
            "normalized_facts": verified["normalized_facts"],
            "data_conflicts": tuple(verified["data_conflicts"]),
            "missing_evidence": tuple(verified["missing_evidence"]),
            "next_node": "policy_specialist",
        }

    return run


def _targeted_refetch(
    missing_evidence: list[str],
) -> tuple[NodeName, str, str] | None:
    actor_nodes: tuple[tuple[NodeName, str], ...] = (
        ("order_specialist", "order-item-agent"),
        ("payment_specialist", "payment-agent"),
        ("shipment_specialist", "shipment-agent"),
    )
    for gap in missing_evidence:
        prefix = "TOOL_CALL_FAILED:"
        if not gap.startswith(prefix):
            continue
        tool_name = gap.removeprefix(prefix)
        for node, actor in actor_nodes:
            if tool_name in TOOL_PERMISSIONS[actor]:
                return node, actor, tool_name
    return None


def _merge_specialist_reports(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    current_tools = {
        str(fact.get("tool_name"))
        for fact in current.get("facts", [])
        if isinstance(fact, Mapping)
    }
    resolved = {f"TOOL_CALL_FAILED:{tool_name}" for tool_name in current_tools}
    missing = [
        str(item)
        for item in previous.get("missing_evidence", [])
        if str(item) not in resolved
    ]
    missing.extend(str(item) for item in current.get("missing_evidence", []))
    facts = [*previous.get("facts", []), *current.get("facts", [])]
    evidence_refs = list(
        dict.fromkeys(
            [
                *previous.get("evidence_refs", []),
                *current.get("evidence_refs", []),
            ]
        )
    )
    return {
        "actor": current.get("actor", previous.get("actor")),
        "status": "completed" if facts and not missing else "partial",
        "facts": facts,
        "anomaly_flags": list(
            dict.fromkeys(
                [
                    *previous.get("anomaly_flags", []),
                    *current.get("anomaly_flags", []),
                ]
            )
        ),
        "missing_evidence": list(dict.fromkeys(missing)),
        "evidence_refs": evidence_refs,
    }


def make_triage_node(trace: TraceWriter) -> GraphNode:
    async def run(state: WorkflowState) -> GraphUpdate:
        case = _normalized_case(state["case"])
        topics = _claim_topics(case)
        specialist_plan: list[str] = ["order_specialist", "payment_specialist"]
        shipment_topics = {
            "late_delivery_seller",
            "late_delivery_logistics",
            "unsupported_claim",
        }
        if topics & shipment_topics:
            specialist_plan.append("shipment_specialist")

        order_tools = ["get_order", "get_order_items"]
        if topics & {"late_delivery_seller", "unavailable_order_paid"}:
            order_tools.append("get_sellers")
        payment_tools = ["get_order_payments"]
        if topics & {
            "canceled_order_paid",
            "unavailable_order_paid",
            "valid_split_payment",
            "payment_mismatch",
            "duplicate_charge",
            "refund_pending",
            "refund_failed",
            "unsupported_claim",
        }:
            payment_tools.append("get_payment_timeline")
        if topics & {
            "canceled_order_paid",
            "unavailable_order_paid",
            "duplicate_charge",
            "refund_pending",
            "refund_failed",
        }:
            payment_tools.append("get_refund_timeline")
        tool_plan: dict[str, tuple[str, ...]] = {
            "order-item-agent": tuple(order_tools),
            "payment-agent": tuple(payment_tools),
            "shipment-agent": ("get_shipment_summary",),
            "policy-agent": ("get_policy",),
        }
        trace.emit(
            case_id=state["case_id"],
            event_type="handoff",
            actor="coordinator",
            target="order-item-agent",
            decision_code="CASE_TRIAGED",
            attributes={"specialist_count": len(specialist_plan)},
        )
        return {
            "case": case,
            "specialist_plan": tuple(specialist_plan),
            "completed_specialists": (),
            "specialist_reports": {},
            "evidence_ledger": (),
            "refetch_count": 0,
            "tool_plan": tool_plan,
            "next_node": cast(NodeName, specialist_plan[0]),
        }

    return run


def make_policy_node(collector: EvidenceCollector) -> GraphNode:
    async def run(state: WorkflowState) -> GraphUpdate:
        collector.trace.emit(
            case_id=collector.case_id,
            event_type="task_assigned",
            actor="coordinator",
            target="policy-agent",
            decision_code="POLICY_REVIEW_REQUIRED",
        )
        reports = dict(state.get("specialist_reports", {}))
        report = await POLICY_SPECIALIST.investigate(
            case=state["case"],
            collector=collector,
            context={
                "normalized_facts": state.get("normalized_facts", {}),
                "specialist_reports": reports,
            },
            requested_tools=state.get("tool_plan", {}).get(
                "policy-agent", POLICY_SPECIALIST.tools
            ),
        )
        reports["policy-agent"] = report
        decision_code = (
            "POLICY_EVIDENCE_COLLECTED"
            if report["evidence_refs"]
            else "POLICY_EVIDENCE_MISSING"
        )
        collector.trace.emit(
            case_id=collector.case_id,
            event_type="policy_decided",
            actor="policy-agent",
            decision_code=decision_code,
            evidence_refs=report["evidence_refs"][:20],
        )
        return {
            "specialist_reports": reports,
            "evidence_ledger": collector.ledger.snapshot(),
            "policy_decision": report,
        }

    return run


def make_judicial_node() -> GraphNode:
    async def run(state: WorkflowState) -> GraphUpdate:
        verified = cross_verify_reports(state.get("specialist_reports", {}))
        proposal = propose_decision(
            verified,
            claimed_topics=(
                topic
                for topic in _claim_topics(state["case"])
                if topic != "requested_full_refund"
            ),
        )
        return {
            "normalized_facts": verified["normalized_facts"],
            "data_conflicts": tuple(verified["data_conflicts"]),
            "missing_evidence": tuple(verified["missing_evidence"]),
            "judicial_proposal": proposal,
        }

    return run


def make_builder_node() -> GraphNode:
    async def run(state: WorkflowState) -> GraphUpdate:
        cross_verified = {
            "normalized_facts": state.get("normalized_facts", {}),
            "data_conflicts": list(state.get("data_conflicts", ())),
            "missing_evidence": list(state.get("missing_evidence", ())),
            "evidence_refs": [
                record["evidence_ref"]
                for record in state.get("evidence_ledger", ())
                if record.get("consumed") is True
            ],
            "evidence_domains": [
                record["domain"]
                for record in state.get("evidence_ledger", ())
                if record.get("consumed") is True
            ],
        }
        output = build_l3a_output(
            case_id=state["case_id"],
            cross_verified=cross_verified,
            proposal=state["judicial_proposal"],
            claims=state["case"].get("customer_request", {}).get("claims", []),
        )
        return {"output_candidate": output}

    return run


def make_verifier_node(contracts: Contracts, trace: TraceWriter) -> GraphNode:
    verifier = DeterministicVerifier(contracts)

    async def run(state: WorkflowState) -> GraphUpdate:
        output = verifier.verify(
            state["output_candidate"],
            case_id=state["case_id"],
            evidence_ledger=state.get("evidence_ledger", ()),
        )
        trace.emit(
            case_id=state["case_id"],
            event_type="verification_completed",
            actor="deterministic-verifier",
            decision_code="OUTPUT_VERIFIED",
            evidence_refs=output["evidence_refs"][:20],
        )
        return {"final_output": output, "next_node": "end"}

    return run


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run the evidence-first L3A LangGraph for one released competition case."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError("case is missing a valid case_id")
    collector = EvidenceCollector(case_id=case_id, gateway=gateway, trace=trace)
    nodes = WorkflowNodes(
        triage=make_triage_node(trace),
        order_specialist=make_specialist_node(ORDER_SPECIALIST, collector),
        payment_specialist=make_specialist_node(PAYMENT_SPECIALIST, collector),
        shipment_specialist=make_specialist_node(SHIPMENT_SPECIALIST, collector),
        cross_verification=make_cross_verification_node(trace),
        policy_specialist=make_policy_node(collector),
        judicial_engine=make_judicial_node(),
        deterministic_builder=make_builder_node(),
        deterministic_verifier=make_verifier_node(trace.contracts, trace),
    )
    final_state = await build_workflow(nodes).ainvoke(
        {"case": case, "case_id": case_id},
        config={"recursion_limit": 20},
    )
    return as_final_output(final_state)


def as_final_output(state: WorkflowState) -> dict[str, Any]:
    """Return the verifier-owned final object without adding or renaming contract fields."""

    output = state.get("final_output")
    if not isinstance(output, dict):
        raise RuntimeError("workflow completed without a verified final_output")
    return cast(dict[str, Any], output)


def _normalized_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    customer_request = case.get("customer_request")
    if not isinstance(customer_request, dict):
        raise ValueError("case customer_request must be an object")
    normalized["customer_request"] = dict(customer_request)
    claimed_order_id = customer_request.get("claimed_order_id")
    if isinstance(claimed_order_id, str) and claimed_order_id:
        normalized["order_id"] = claimed_order_id
    topics = _claim_topics(case)
    primary_claim = next((topic for topic in topics if topic != "requested_full_refund"), None)
    if primary_claim is not None:
        normalized["topic"] = primary_claim
        normalized["issue_code"] = primary_claim
        normalized["primary_issue"] = primary_claim
    return normalized


def _claim_topics(case: Mapping[str, Any]) -> set[str]:
    customer_request = case.get("customer_request", {})
    if not isinstance(customer_request, Mapping):
        return set()
    claims = customer_request.get("claims", [])
    if not isinstance(claims, list):
        return set()
    return {
        str(claim["topic"])
        for claim in claims
        if isinstance(claim, Mapping) and isinstance(claim.get("topic"), str)
    }
