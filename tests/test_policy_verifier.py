from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from student_agent.contracts import Contracts
from student_agent.policy_engine import (
    build_l3a_output,
    calibrate_confidence,
    cross_verify_reports,
    propose_decision,
)
from student_agent.submission import REQUIRED_LIFECYCLE, validate_case_lifecycle
from student_agent.verifier import DeterministicVerifier, VerificationError

ROOT = Path(__file__).resolve().parents[1]


def reports() -> dict[str, dict[str, object]]:
    return {
        "payment-agent": {
            "facts": [
                {
                    "tool_name": "get_order_payments",
                    "domain": "payment",
                    "data": {
                        "duplicate_charge": True,
                        "refundable_total_brl": 100.0,
                        "payment_reference": "PAY_001",
                    },
                    "evidence_refs": ["ev_12345678901234567890"],
                }
            ],
            "missing_evidence": [],
        },
        "policy-agent": {
            "facts": [
                {
                    "tool_name": "get_policy",
                    "domain": "policy",
                    "data": {"policy_id": "DUPLICATE_CHARGE_REFUND"},
                    "evidence_refs": ["ev_abcdefghijabcdefghij"],
                }
            ],
            "missing_evidence": [],
        },
    }


def ledger() -> tuple[dict[str, object], ...]:
    return (
        {
            "case_id": "CASE_001",
            "actor": "payment-agent",
            "tool_name": "get_order_payments",
            "evidence_ref": "ev_12345678901234567890",
            "result_hash": "sha256:" + "0" * 64,
            "domain": "payment",
            "consumed": True,
        },
        {
            "case_id": "CASE_001",
            "actor": "policy-agent",
            "tool_name": "get_policy",
            "evidence_ref": "ev_abcdefghijabcdefghij",
            "result_hash": "sha256:" + "1" * 64,
            "domain": "policy",
            "consumed": True,
        },
    )


def duplicate_charge_output() -> dict[str, object]:
    verified = cross_verify_reports(reports())
    proposal = propose_decision(verified)
    return build_l3a_output(
        case_id="CASE_001",
        cross_verified=verified,
        proposal=proposal,
    )


def test_policy_engine_builds_schema_valid_consistent_refund() -> None:
    output = duplicate_charge_output()
    assert output["assessment"]["primary_issue"] == "duplicate_charge"  # type: ignore[index]
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0  # type: ignore[index]
    assert output["root_cause_analysis"]["responsible_parties"] == [  # type: ignore[index]
        {"party_type": "payment_provider", "party_id": None}
    ]

    verifier = DeterministicVerifier(Contracts(ROOT / "contracts" / "schemas"))
    assert verifier.verify(output, case_id="CASE_001", evidence_ledger=ledger()) == output


def test_verifier_rejects_financial_inconsistency() -> None:
    output = deepcopy(duplicate_charge_output())
    output["financial_resolution"]["recommended_refund_brl"] = 99.0  # type: ignore[index]
    verifier = DeterministicVerifier(Contracts(ROOT / "contracts" / "schemas"))
    with pytest.raises(VerificationError, match="refund line total"):
        verifier.verify(output, case_id="CASE_001", evidence_ledger=ledger())


def test_verifier_allows_policy_documentation_for_no_action() -> None:
    output = deepcopy(duplicate_charge_output())
    output["assessment"] = {  # type: ignore[index]
        "primary_issue": "valid_split_payment",
        "case_status": "no_action",
        "confidence": 0.8,
    }
    output["financial_resolution"] = {  # type: ignore[index]
        "currency": "BRL",
        "recommended_refund_brl": 0.0,
        "refund_lines": [],
    }
    output["resolution_actions"] = ["document_no_action"]
    output["root_cause_analysis"] = {  # type: ignore[index]
        "ranked_causes": [{"cause_code": "VALID_SPLIT_PAYMENT", "rank": 1}],
        "responsible_parties": [{"party_type": "customer", "party_id": None}],
    }
    verifier = DeterministicVerifier(Contracts(ROOT / "contracts" / "schemas"))
    verifier.verify(output, case_id="CASE_001", evidence_ledger=ledger())

    output["resolution_actions"] = ["issue_refund"]
    with pytest.raises(VerificationError, match="only document_no_action"):
        verifier.verify(output, case_id="CASE_001", evidence_ledger=ledger())


def test_verifier_rejects_unconsumed_or_missing_policy_evidence() -> None:
    output = duplicate_charge_output()
    verifier = DeterministicVerifier(Contracts(ROOT / "contracts" / "schemas"))
    payment_only = ledger()[:1]
    with pytest.raises(VerificationError, match="unconsumed evidence"):
        verifier.verify(output, case_id="CASE_001", evidence_ledger=payment_only)


def test_conflicts_and_missing_evidence_reduce_confidence() -> None:
    complete = calibrate_confidence(
        issue="payment_mismatch",
        evidence_refs=["ev_a", "ev_b"],
        evidence_domains=["payment", "policy"],
        conflicts=[],
        missing_evidence=[],
    )
    incomplete = calibrate_confidence(
        issue="payment_mismatch",
        evidence_refs=["ev_a", "ev_b"],
        evidence_domains=["payment", "policy"],
        conflicts=[{"field": "payment_status"}],
        missing_evidence=["MISSING_TOOL_ARGUMENTS:get_payment_timeline"],
    )
    assert incomplete < complete < 1.0


def test_policy_catalog_does_not_create_case_facts_and_selected_rule_is_applied() -> None:
    source = reports()
    source["payment-agent"]["facts"][0]["data"] = {  # type: ignore[index]
        "order_status": "canceled",
        "order_id": "ORDER_001",
    }
    source["policy-agent"]["facts"][0]["data"] = {  # type: ignore[index]
        "rules": {
            "canceled_order_paid": {
                "case_status": "action_required",
                "recommended_action": "issue_refund",
                "refund_brl": 79.0,
                "responsible_parties": [
                    {"party_type": "platform", "party_id": None}
                ],
            },
            "duplicate_charge": {
                "case_status": "action_required",
                "recommended_action": "refund_duplicate_charge",
                "refund_brl": 64.0,
                "responsible_parties": [
                    {"party_type": "payment_provider", "party_id": None}
                ],
            },
        }
    }
    verified = cross_verify_reports(source)
    proposal = propose_decision(verified, claimed_topics=["canceled_order_paid"])
    output = build_l3a_output(
        case_id="CASE_001",
        cross_verified=verified,
        proposal=proposal,
    )
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"  # type: ignore[index]
    assert output["financial_resolution"]["recommended_refund_brl"] == 79.0  # type: ignore[index]
    assert output["resolution_actions"] == ["issue_refund"]


def test_submission_requires_ordered_lifecycle_and_evidence_linkage() -> None:
    output = duplicate_charge_output()
    events = [
        {
            "event_type": event_type,
            "evidence_refs": (
                output["evidence_refs"] if event_type == "tool_result_consumed" else []
            ),
        }
        for event_type in REQUIRED_LIFECYCLE
    ]
    validate_case_lifecycle("CASE_001", output, events)

    with pytest.raises(ValueError, match="policy_decided"):
        validate_case_lifecycle(
            "CASE_001",
            output,
            [event for event in events if event["event_type"] != "policy_decided"],
        )
    with pytest.raises(ValueError, match="absent from tool_result_consumed"):
        broken = deepcopy(events)
        broken[2]["evidence_refs"] = []
        validate_case_lifecycle("CASE_001", output, broken)
