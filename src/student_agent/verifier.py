from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from .contracts import ContractError, Contracts

EXPECTED_PARTY = {
    "canceled_order_paid": "platform",
    "unavailable_order_paid": "seller",
    "late_delivery_seller": "seller",
    "late_delivery_logistics": "logistics_provider",
    "valid_split_payment": "customer",
    "payment_mismatch": "payment_provider",
    "duplicate_charge": "payment_provider",
    "refund_pending": "payment_provider",
    "refund_failed": "payment_provider",
    "unsupported_claim": "customer",
}
EXPECTED_STATUS = {
    "canceled_order_paid": "action_required",
    "unavailable_order_paid": "action_required",
    "late_delivery_seller": "action_required",
    "late_delivery_logistics": "action_required",
    "valid_split_payment": "no_action",
    "payment_mismatch": "action_required",
    "duplicate_charge": "action_required",
    "refund_pending": "needs_investigation",
    "refund_failed": "action_required",
    "unsupported_claim": "no_action",
    "insufficient_evidence": "needs_investigation",
}
POSITIVE_REFUND_ISSUES = frozenset(
    {
        "canceled_order_paid",
        "unavailable_order_paid",
        "late_delivery_seller",
        "late_delivery_logistics",
        "payment_mismatch",
        "duplicate_charge",
        "refund_failed",
    }
)


class VerificationError(ValueError):
    pass


class DeterministicVerifier:
    def __init__(self, contracts: Contracts) -> None:
        self.contracts = contracts

    def verify(
        self,
        output: dict[str, Any],
        *,
        case_id: str,
        evidence_ledger: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        try:
            self.contracts.validate_output(output, f"outputs/{case_id}.json")
        except ContractError as exc:
            raise VerificationError(str(exc)) from exc
        if output["case_id"] != case_id:
            raise VerificationError("output case_id does not match verifier scope")

        consumed = {
            str(record["evidence_ref"])
            for record in evidence_ledger
            if record.get("case_id") == case_id and record.get("consumed") is True
        }
        submitted = set(output["evidence_refs"])
        if not submitted <= consumed:
            unknown = sorted(submitted - consumed)
            raise VerificationError(f"output contains unconsumed evidence refs: {unknown}")

        issue = output["assessment"]["primary_issue"]
        if issue not in {"insufficient_evidence", "unsupported_claim"} and not submitted:
            raise VerificationError("a substantive decision requires consumed MCP evidence")
        consumed_domains = {
            str(record.get("domain"))
            for record in evidence_ledger
            if record.get("case_id") == case_id
            and record.get("consumed") is True
            and record.get("evidence_ref") in submitted
        }
        if output["assessment"]["case_status"] == "action_required" and (
            "policy" not in consumed_domains
        ):
            raise VerificationError("action_required decision requires consumed policy evidence")
        self._verify_financial_resolution(output)
        self._verify_status_and_actions(output)
        self._verify_responsibility(output)
        self._verify_root_cause_ranks(output)
        self._verify_confidence(output)
        return output

    @staticmethod
    def _verify_financial_resolution(output: Mapping[str, Any]) -> None:
        financial = output["financial_resolution"]
        recommended = _decimal(financial["recommended_refund_brl"])
        line_total = sum(
            (_decimal(line["amount_brl"]) for line in financial["refund_lines"]),
            Decimal("0"),
        )
        if recommended != line_total:
            raise VerificationError("recommended refund does not equal refund line total")
        if recommended > 0 and not financial["refund_lines"]:
            raise VerificationError("positive refund requires at least one refund line")

    @staticmethod
    def _verify_status_and_actions(output: Mapping[str, Any]) -> None:
        status = output["assessment"]["case_status"]
        issue = output["assessment"]["primary_issue"]
        actions = output["resolution_actions"]
        refund = _decimal(output["financial_resolution"]["recommended_refund_brl"])
        expected_status = EXPECTED_STATUS[issue]
        if status != expected_status:
            raise VerificationError(
                f"primary issue {issue!r} requires case status {expected_status!r}"
            )
        if status == "action_required" and not actions:
            raise VerificationError("action_required case must contain a resolution action")
        if issue in POSITIVE_REFUND_ISSUES and refund <= 0:
            raise VerificationError(f"primary issue {issue!r} requires a positive refund")
        if status == "no_action" and refund > 0:
            raise VerificationError("no_action case cannot recommend a positive refund")
        if status == "no_action":
            unsafe_actions = {
                str(action).lower() for action in actions
            } - {"document_no_action"}
            if unsafe_actions:
                raise VerificationError(
                    "no_action case can contain only document_no_action"
                )

    @staticmethod
    def _verify_responsibility(output: Mapping[str, Any]) -> None:
        issue = output["assessment"]["primary_issue"]
        expected = EXPECTED_PARTY.get(issue)
        if expected is None:
            return
        actual = {
            item["party_type"] for item in output["root_cause_analysis"]["responsible_parties"]
        }
        if expected not in actual:
            raise VerificationError(
                f"primary issue {issue!r} requires responsible party {expected!r}"
            )

    @staticmethod
    def _verify_root_cause_ranks(output: Mapping[str, Any]) -> None:
        ranks = [item["rank"] for item in output["root_cause_analysis"]["ranked_causes"]]
        if ranks != list(range(1, len(ranks) + 1)):
            raise VerificationError("root-cause ranks must be unique and contiguous")

    @staticmethod
    def _verify_confidence(output: Mapping[str, Any]) -> None:
        confidence = output["assessment"]["confidence"]
        conflicts = output["data_conflicts"]
        issue = output["assessment"]["primary_issue"]
        if conflicts and confidence > 0.8:
            raise VerificationError("confidence is too high for unresolved data conflicts")
        if issue == "insufficient_evidence" and confidence > 0.45:
            raise VerificationError("confidence is too high for insufficient evidence")


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise VerificationError(f"invalid monetary value: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise VerificationError(f"invalid monetary value: {value!r}")
    return result
