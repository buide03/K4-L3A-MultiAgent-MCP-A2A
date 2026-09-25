from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

PRIMARY_ISSUES = frozenset(
    {
        "canceled_order_paid",
        "unavailable_order_paid",
        "late_delivery_seller",
        "late_delivery_logistics",
        "valid_split_payment",
        "payment_mismatch",
        "duplicate_charge",
        "refund_pending",
        "refund_failed",
        "unsupported_claim",
        "insufficient_evidence",
    }
)

PARTY_BY_ISSUE = {
    "late_delivery_seller": "seller",
    "late_delivery_logistics": "logistics_provider",
    "valid_split_payment": "customer",
    "payment_mismatch": "payment_provider",
    "duplicate_charge": "payment_provider",
    "refund_pending": "payment_provider",
    "refund_failed": "payment_provider",
}

REFUND_ISSUES = frozenset(
    {
        "canceled_order_paid",
        "unavailable_order_paid",
        "payment_mismatch",
        "duplicate_charge",
        "refund_failed",
    }
)

CONFLICT_FIELDS = frozenset(
    {
        "order_status",
        "payment_status",
        "payment_verdict",
        "refund_status",
        "shipment_status",
        "shipment_verdict",
        "captured_total_brl",
        "refunded_total_brl",
        "refundable_total_brl",
        "responsible_party",
    }
)
NORMALIZED_FIELDS = CONFLICT_FIELDS | {
    "primary_issue",
    "duplicate_charge",
    "paid",
    "recommended_refund_brl",
    "duplicate_amount_brl",
    "payment_reference",
}


def cross_verify_reports(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    observed: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    policy_rules: dict[str, Any] = {}
    evidence_refs: list[str] = []
    evidence_domains: set[str] = set()
    missing_evidence: list[str] = []

    for actor, report in reports.items():
        missing_evidence.extend(str(item) for item in report.get("missing_evidence", []))
        for fact in report.get("facts", []):
            if not isinstance(fact, Mapping):
                continue
            source = str(fact.get("tool_name") or actor)[:80]
            domain = fact.get("domain")
            if isinstance(domain, str):
                evidence_domains.add(domain)
            evidence_refs.extend(
                ref for ref in fact.get("evidence_refs", []) if isinstance(ref, str)
            )
            data = fact.get("data")
            if domain == "policy" and isinstance(data, Mapping):
                rules = data.get("rules")
                if isinstance(rules, Mapping):
                    policy_rules.update(
                        (str(key), value) for key, value in rules.items()
                    )
                continue
            if isinstance(data, Mapping | list):
                for key, value in _walk_fields(data):
                    if key in NORMALIZED_FIELDS or key.endswith("_id") or key.endswith("_ids"):
                        observed[key].append((source, value))

    normalized: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    for field, entries in observed.items():
        distinct = _distinct_values(value for _, value in entries)
        if field.endswith("_id") or field.endswith("_ids"):
            flattened: list[Any] = []
            for value in distinct:
                flattened.extend(value if isinstance(value, list) else [value])
            normalized[field] = _distinct_values(flattened)
        elif len(distinct) == 1:
            normalized[field] = distinct[0]
        elif field in CONFLICT_FIELDS:
            conflicts.append(
                {
                    "field": field,
                    "sources": _unique_strings(source for source, _ in entries)[:5],
                    "selected_source": None,
                    "resolution_code": "UNRESOLVED_SOURCE_CONFLICT",
                }
            )

    return {
        "normalized_facts": normalized,
        "data_conflicts": conflicts[:5],
        "evidence_refs": _unique_strings(evidence_refs),
        "evidence_domains": sorted(evidence_domains),
        "missing_evidence": _unique_strings(missing_evidence),
        "policy_rules": policy_rules,
    }


def propose_decision(
    cross_verified: Mapping[str, Any],
    *,
    claimed_topics: Iterable[str] = (),
) -> dict[str, Any]:
    facts = cross_verified.get("normalized_facts", {})
    if not isinstance(facts, Mapping):
        facts = {}
    evidence_domains = cross_verified.get("evidence_domains", [])
    issue = _primary_issue(
        facts,
        bool(cross_verified.get("evidence_refs")),
        claimed_topics=claimed_topics,
        evidence_domains=evidence_domains,
    )
    rule = _policy_rule(cross_verified.get("policy_rules"), issue)
    rule_parties = rule.get("responsible_parties")
    responsible_parties = (
        [dict(item) for item in rule_parties if isinstance(item, Mapping)]
        if isinstance(rule_parties, list)
        else []
    )
    party_type = (
        str(responsible_parties[0].get("party_type"))
        if responsible_parties
        else _responsible_party(issue, facts)
    )
    confidence = calibrate_confidence(
        issue=issue,
        evidence_refs=cross_verified.get("evidence_refs", []),
        evidence_domains=evidence_domains,
        conflicts=cross_verified.get("data_conflicts", []),
        missing_evidence=cross_verified.get("missing_evidence", []),
    )
    return {
        "primary_issue": issue,
        "case_status": rule.get("case_status", _case_status(issue)),
        "confidence": confidence,
        "cause_code": issue.upper(),
        "responsible_party": party_type,
        "responsible_parties": responsible_parties,
        "recommended_refund_brl": rule.get("refund_brl"),
        "resolution_actions": (
            [str(rule["recommended_action"])]
            if isinstance(rule.get("recommended_action"), str)
            else _resolution_actions(issue)
        ),
    }


def build_l3a_output(
    *,
    case_id: str,
    cross_verified: Mapping[str, Any],
    proposal: Mapping[str, Any],
    claims: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    facts = cross_verified.get("normalized_facts", {})
    if not isinstance(facts, Mapping):
        facts = {}
    issue = str(proposal["primary_issue"])
    party_type = str(proposal["responsible_party"])
    refund = _money(proposal.get("recommended_refund_brl"))
    if refund is None:
        refund = _recommended_refund(issue, facts)
    evidence_refs = _unique_strings(cross_verified.get("evidence_refs", []))[:30]
    entity_id = _first_id(facts, "payment_reference", "order_id")
    refund_lines = []
    if refund > Decimal("0"):
        refund_lines.append(
            {
                "reason_code": issue.upper(),
                "amount_brl": float(refund),
                "entity_id": entity_id,
            }
        )
    proposed_parties = proposal.get("responsible_parties")
    if isinstance(proposed_parties, list) and proposed_parties:
        responsible_parties = [
            {
                "party_type": str(item["party_type"]),
                "party_id": item.get("party_id"),
            }
            for item in proposed_parties
            if isinstance(item, Mapping) and "party_type" in item
        ][:5]
    else:
        responsible_parties = []
        if party_type != "unknown":
            responsible_parties.append(
                {
                    "party_type": party_type,
                    "party_id": _party_id(party_type, facts),
                }
            )

    output: dict[str, Any] = {
        "schema_version": "day09-l3a-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": issue,
            "case_status": proposal["case_status"],
            "confidence": proposal["confidence"],
        },
        "affected_entities": {
            "order_ids": _ids(facts, "order_id", "order_ids"),
            "item_ids": _ids(facts, "item_id", "item_ids", "order_item_id"),
            "seller_ids": _ids(facts, "seller_id", "seller_ids"),
            "payment_references": _ids(
                facts, "payment_reference", "payment_references"
            ),
            "shipment_ids": _ids(facts, "shipment_id", "shipment_ids"),
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": proposal["cause_code"], "rank": 1}],
            "responsible_parties": responsible_parties,
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": list(cross_verified.get("data_conflicts", []))[:5],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": float(refund),
            "refund_lines": refund_lines,
        },
        "resolution_actions": list(proposal["resolution_actions"]),
    }
    claim_assessments = _claim_assessments(
        claims=claims,
        primary_issue=issue,
        case_status=str(proposal["case_status"]),
        confidence=float(proposal["confidence"]),
        evidence_refs=evidence_refs,
        refund=refund,
    )
    if claim_assessments:
        output["claim_assessments"] = claim_assessments
    return output


def calibrate_confidence(
    *,
    issue: str,
    evidence_refs: Iterable[Any],
    evidence_domains: Iterable[Any],
    conflicts: Iterable[Any],
    missing_evidence: Iterable[Any],
) -> float:
    ref_count = len(set(str(item) for item in evidence_refs))
    domain_count = len(set(str(item) for item in evidence_domains))
    conflict_count = len(list(conflicts))
    missing_count = len(list(missing_evidence))
    score = Decimal("0.35")
    score += min(domain_count, 3) * Decimal("0.12")
    score += min(ref_count, 3) * Decimal("0.06")
    score -= min(conflict_count, 3) * Decimal("0.15")
    score -= min(missing_count, 3) * Decimal("0.12")
    if issue == "insufficient_evidence":
        score = min(score, Decimal("0.45"))
    score = min(Decimal("0.95"), max(Decimal("0.05"), score))
    return float(score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _primary_issue(
    facts: Mapping[str, Any],
    has_evidence: bool,
    *,
    claimed_topics: Iterable[str] = (),
    evidence_domains: Iterable[Any] = (),
) -> str:
    explicit = facts.get("primary_issue")
    if explicit in PRIMARY_ISSUES:
        return str(explicit)
    domains = {str(domain) for domain in evidence_domains}
    for topic in claimed_topics:
        if topic in PRIMARY_ISSUES and _candidate_supported(topic, facts, domains):
            return topic
    if _truthy(facts.get("duplicate_charge")) or _contains(facts, "duplicate"):
        return "duplicate_charge"
    refund_status = str(facts.get("refund_status", "")).lower()
    if "fail" in refund_status:
        return "refund_failed"
    if "pending" in refund_status or "processing" in refund_status:
        return "refund_pending"
    order_status = str(facts.get("order_status", "")).lower()
    paid = _is_paid(facts)
    if "cancel" in order_status and paid:
        return "canceled_order_paid"
    if "unavailable" in order_status and paid:
        return "unavailable_order_paid"
    shipment = str(
        facts.get("shipment_verdict", facts.get("shipment_status", ""))
    ).lower()
    if "seller" in shipment and ("late" in shipment or "delay" in shipment):
        return "late_delivery_seller"
    if "logistic" in shipment and ("late" in shipment or "delay" in shipment):
        return "late_delivery_logistics"
    payment = str(
        facts.get("payment_verdict", facts.get("payment_status", ""))
    ).lower()
    if "mismatch" in payment:
        return "payment_mismatch"
    if "split" in payment and ("valid" in payment or "reconciled" in payment):
        return "valid_split_payment"
    if not has_evidence:
        return "insufficient_evidence"
    return "unsupported_claim"


def _candidate_supported(
    topic: str,
    facts: Mapping[str, Any],
    evidence_domains: set[str],
) -> bool:
    """Use a customer topic only as a hypothesis gated by authoritative domains/facts."""

    order_status = str(facts.get("order_status", "")).lower()
    if topic == "canceled_order_paid":
        return "cancel" in order_status and "payment" in evidence_domains
    if topic == "unavailable_order_paid":
        return "order" in evidence_domains and "payment" in evidence_domains
    if topic in {"late_delivery_seller", "late_delivery_logistics"}:
        return "shipment" in evidence_domains
    if topic in {"valid_split_payment", "payment_mismatch", "duplicate_charge"}:
        return "payment" in evidence_domains
    if topic in {"refund_pending", "refund_failed"}:
        return "refund" in evidence_domains
    if topic == "unsupported_claim":
        return bool(evidence_domains - {"policy"})
    return False


def _policy_rule(value: Any, issue: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    rule = value.get(issue)
    return rule if isinstance(rule, Mapping) else {}


def _responsible_party(issue: str, facts: Mapping[str, Any]) -> str:
    explicit = facts.get("responsible_party")
    allowed = {
        "seller",
        "platform",
        "logistics_provider",
        "payment_provider",
        "customer",
        "unknown",
    }
    if explicit in allowed:
        return str(explicit)
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        return "platform"
    return PARTY_BY_ISSUE.get(issue, "unknown")


def _case_status(issue: str) -> str:
    if issue in {"valid_split_payment", "unsupported_claim"}:
        return "no_action"
    if issue == "insufficient_evidence":
        return "needs_investigation"
    return "action_required"


def _resolution_actions(issue: str) -> list[str]:
    if issue in REFUND_ISSUES:
        return ["INITIATE_OR_CORRECT_REFUND", "NOTIFY_CUSTOMER"]
    if issue == "refund_pending":
        return ["MONITOR_REFUND", "NOTIFY_CUSTOMER"]
    if issue == "late_delivery_seller":
        return ["REVIEW_SELLER_PERFORMANCE", "NOTIFY_CUSTOMER"]
    if issue == "late_delivery_logistics":
        return ["ESCALATE_LOGISTICS_DELAY", "NOTIFY_CUSTOMER"]
    if issue == "insufficient_evidence":
        return ["REQUEST_ADDITIONAL_EVIDENCE"]
    return []


def _recommended_refund(issue: str, facts: Mapping[str, Any]) -> Decimal:
    if issue not in REFUND_ISSUES:
        return Decimal("0.00")
    for key in ("recommended_refund_brl", "refundable_total_brl", "duplicate_amount_brl"):
        amount = _money(facts.get(key))
        if amount is not None:
            return amount
    captured = _money(facts.get("captured_total_brl"))
    refunded = _money(facts.get("refunded_total_brl")) or Decimal("0")
    if captured is None:
        return Decimal("0.00")
    return max(Decimal("0.00"), captured - refunded).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _money(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _is_paid(facts: Mapping[str, Any]) -> bool:
    if _truthy(facts.get("paid")):
        return True
    status = str(facts.get("payment_status", "")).lower()
    return status in {"approved", "captured", "paid", "reconciled"}


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes"}


def _contains(facts: Mapping[str, Any], needle: str) -> bool:
    return any(needle in str(value).lower() for value in facts.values())


def _walk_fields(value: Mapping[str, Any] | list[Any]) -> Iterable[tuple[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping | list):
                yield from _walk_fields(item)
        return
    for key, nested in value.items():
        if isinstance(key, str):
            yield key, nested
        if isinstance(nested, Mapping | list):
            yield from _walk_fields(nested)


def _distinct_values(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    markers: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker not in markers:
            markers.add(marker)
            result.append(value)
    return result


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _ids(facts: Mapping[str, Any], *keys: str) -> list[str]:
    result: list[str] = []
    for key in keys:
        value = facts.get(key)
        if isinstance(value, list):
            result.extend(str(item) for item in value if item is not None)
        elif value is not None:
            result.append(str(value))
    return _unique_strings(result)[:20]


def _first_id(facts: Mapping[str, Any], *keys: str) -> str | None:
    values = _ids(facts, *keys)
    return values[0] if values else None


def _party_id(party_type: str, facts: Mapping[str, Any]) -> str | None:
    if party_type == "seller":
        return _first_id(facts, "seller_id", "seller_ids")
    return None


def _claim_assessments(
    *,
    claims: Iterable[Mapping[str, Any]],
    primary_issue: str,
    case_status: str,
    confidence: float,
    evidence_refs: list[str],
    refund: Decimal,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = claim.get("claim_id")
        topic = claim.get("topic")
        if not isinstance(claim_id, str) or not isinstance(topic, str):
            continue
        if primary_issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif topic == "requested_full_refund":
            verdict = "supported" if refund > 0 else "unsupported"
        elif topic == primary_issue:
            verdict = "supported"
        elif primary_issue == "unsupported_claim" or case_status == "no_action":
            verdict = "unsupported"
        else:
            verdict = "partially_supported"
        result.append(
            {
                "claim_id": claim_id,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_refs": evidence_refs,
            }
        )
    return result[:5]
