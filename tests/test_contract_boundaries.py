from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from student_agent.contracts import ContractError, Contracts

ROOT = Path(__file__).resolve().parents[1]


def contracts() -> Contracts:
    return Contracts(ROOT / "contracts" / "schemas")


def minimal_l3a_output() -> dict[str, object]:
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


def test_minimal_l3a_output_matches_locked_public_contract() -> None:
    contracts().validate_output(minimal_l3a_output(), "minimal output")


def test_l3a_output_rejects_unknown_root_field() -> None:
    output = minimal_l3a_output()
    output["agent_notes"] = "must never enter the public output"
    with pytest.raises(ContractError, match="Additional properties"):
        contracts().validate_output(output, "output with extra field")


def test_l3a_output_rejects_unknown_nested_field() -> None:
    output = minimal_l3a_output()
    assessment = output["assessment"]
    assert isinstance(assessment, dict)
    assessment["reasoning"] = "must remain private"
    with pytest.raises(ContractError, match="Additional properties"):
        contracts().validate_output(output, "output with nested extra field")


def test_evidence_envelope_rejects_unknown_field() -> None:
    evidence = {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": "ev_12345678901234567890",
        "result_hash": "sha256:" + "0" * 64,
        "domain": "order",
        "data": {},
        "debug": True,
    }
    with pytest.raises(ContractError, match="Additional properties"):
        contracts().validate_evidence(evidence)


def test_trace_rejects_unknown_field() -> None:
    event = {
        "schema_version": "day09-trace-event-v1",
        "event_id": "evt_123456789012",
        "case_id": "CASE_001",
        "event_type": "case_received",
        "occurred_at": "2026-09-25T00:00:00Z",
        "actor": "coordinator",
        "reasoning": "must never enter the observable trace",
    }
    with pytest.raises(ContractError, match="Additional properties"):
        contracts().validate_trace(event, "trace with extra field")


def test_manifest_rejects_unknown_field() -> None:
    manifest = {
        "schema_version": "day09-submission-manifest-v2",
        "competition_id": "day09-multiagent-mcp-a2a",
        "variant_id": "l3a",
        "case_set_version": "test-v1",
        "output_schema_version": "day09-l3a-output-v2",
        "trace_schema_version": "day09-trace-event-v1",
        "generated_at": "2026-09-25T00:00:00Z",
        "notes": "must never enter the public manifest",
    }
    with pytest.raises(ContractError, match="Additional properties"):
        contracts().validate_manifest(manifest)


def test_all_public_contract_roots_reject_additional_properties() -> None:
    schema_root = ROOT / "contracts" / "schemas"
    expected = {
        "l3a-output-v2.schema.json",
        "l3b-output-v2.schema.json",
        "mcp-evidence-response-v1.schema.json",
        "submission-manifest-v2.schema.json",
        "trace-event-v1.schema.json",
    }
    assert {path.name for path in schema_root.glob("*.schema.json")} == expected
    for name in expected:
        schema = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False, name


def test_contract_lock_detects_schema_modification(tmp_path: Path) -> None:
    copied_contracts = tmp_path / "contracts"
    shutil.copytree(ROOT / "contracts", copied_contracts)
    schema_path = copied_contracts / "schemas" / "l3a-output-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["title"] = "tampered contract"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ContractError, match="checksum mismatch"):
        Contracts(copied_contracts / "schemas")


def test_contract_lock_detects_schema_inventory_change(tmp_path: Path) -> None:
    copied_contracts = tmp_path / "contracts"
    shutil.copytree(ROOT / "contracts", copied_contracts)
    extra = copied_contracts / "schemas" / "unregistered.schema.json"
    extra.write_text("{}", encoding="utf-8")

    with pytest.raises(ContractError, match="inventory mismatch"):
        Contracts(copied_contracts / "schemas")
