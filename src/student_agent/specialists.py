from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .mcp_gateway import (
    EvidenceGateway,
    MCPAuthorizationError,
    MCPToolError,
    ToolSpec,
)
from .trace import TraceWriter

TOOL_PERMISSIONS: dict[str, frozenset[str]] = {
    "coordinator": frozenset(),
    "order-item-agent": frozenset(
        {
            "get_customer_history",
            "get_order",
            "get_order_items",
            "get_product_context",
            "get_sellers",
        }
    ),
    "payment-agent": frozenset(
        {"get_order_payments", "get_payment_timeline", "get_refund_timeline"}
    ),
    "shipment-agent": frozenset({"get_shipment_summary"}),
    "policy-agent": frozenset({"get_policy"}),
    "judicial-engine": frozenset(),
    "deterministic-builder": frozenset(),
    "deterministic-verifier": frozenset(),
}

MAX_CALLS_PER_TOOL = 20
TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "TimeoutError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
    }
)


def require_tool_permission(actor: str, tool_name: str) -> None:
    allowed = TOOL_PERMISSIONS.get(actor)
    if allowed is None or tool_name not in allowed:
        raise PermissionError(f"actor {actor!r} is not allowed to call {tool_name!r}")


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(_is_transient(item) for item in exc.exceptions)
    return type(exc).__name__ in TRANSIENT_EXCEPTION_NAMES


@dataclass
class EvidenceRecord:
    case_id: str
    actor: str
    tool_name: str
    evidence: dict[str, Any]
    consumptions: set[tuple[str, str]] = field(default_factory=set)


class EvidenceLedger:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._records: dict[str, EvidenceRecord] = {}

    def add(self, *, actor: str, tool_name: str, evidence: dict[str, Any]) -> EvidenceRecord:
        evidence_ref = evidence["evidence_ref"]
        existing = self._records.get(evidence_ref)
        if existing is not None:
            if existing.case_id != self.case_id or existing.tool_name != tool_name:
                raise ValueError("evidence_ref collision across case or tool scope")
            return existing
        record = EvidenceRecord(self.case_id, actor, tool_name, evidence)
        self._records[evidence_ref] = record
        return record

    def get(self, evidence_ref: str) -> EvidenceRecord:
        try:
            return self._records[evidence_ref]
        except KeyError as exc:
            message = f"evidence_ref is not in the current case ledger: {evidence_ref}"
            raise ValueError(message) from exc

    def consumed_refs(self) -> list[str]:
        return sorted(ref for ref, record in self._records.items() if record.consumptions)

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "case_id": record.case_id,
                "actor": record.actor,
                "tool_name": record.tool_name,
                "evidence_ref": evidence_ref,
                "result_hash": record.evidence["result_hash"],
                "domain": record.evidence["domain"],
                "consumed": bool(record.consumptions),
            }
            for evidence_ref, record in sorted(self._records.items())
        )


class EvidenceCollector:
    """Case-scoped, permission-aware access to authoritative MCP evidence."""

    def __init__(
        self,
        *,
        case_id: str,
        gateway: EvidenceGateway,
        trace: TraceWriter,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        self.case_id = case_id
        self.gateway = gateway
        self.trace = trace
        self.retry_delay_seconds = retry_delay_seconds
        self.ledger = EvidenceLedger(case_id)
        self._cache: dict[str, dict[str, Any]] = {}

    async def fetch(
        self,
        *,
        actor: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        require_tool_permission(actor, tool_name)
        if "case_id" in arguments:
            raise ValueError("specialists must not pass case_id inside MCP arguments")
        discovered = await self.gateway.discover_tools()
        if tool_name not in discovered:
            discovered = await self.gateway.discover_tools(refresh=True)
        if tool_name not in discovered:
            raise ValueError(f"authorized MCP tool is unavailable: {tool_name}")

        canonical_arguments = json.dumps(
            dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        cache_key = f"{self.case_id}\0{tool_name}\0{canonical_arguments}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        for attempt in range(2):
            try:
                evidence = await self.gateway.call(
                    tool_name,
                    case_id=self.case_id,
                    **dict(arguments),
                )
                break
            except Exception as exc:
                if attempt == 1 or not _is_transient(exc):
                    raise
                await asyncio.sleep(self.retry_delay_seconds)
        self.ledger.add(actor=actor, tool_name=tool_name, evidence=evidence)
        self._cache[cache_key] = evidence
        return evidence

    def consume(
        self,
        *,
        actor: str,
        tool_name: str,
        evidence: dict[str, Any],
        decision_code: str,
    ) -> None:
        require_tool_permission(actor, tool_name)
        evidence_ref = evidence.get("evidence_ref")
        if not isinstance(evidence_ref, str):
            raise ValueError("cannot consume evidence without a valid evidence_ref")
        record = self.ledger.get(evidence_ref)
        if record.actor != actor or record.tool_name != tool_name:
            raise ValueError("evidence consumer does not match the recorded actor and tool")
        consumption = (actor, decision_code)
        if consumption in record.consumptions:
            return
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            decision_code=decision_code,
            tool_name=tool_name,
            evidence_refs=[evidence_ref],
        )
        record.consumptions.add(consumption)


class ToolArgumentBinder:
    """Bind only exact structured fields; never parse identifiers from customer prose."""

    def __init__(self, *sources: Any) -> None:
        self._values: dict[str, list[Any]] = defaultdict(list)
        for source in sources:
            self._index(source)

    def _index(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if isinstance(key, str):
                    if isinstance(nested, list) and all(_is_scalar(item) for item in nested):
                        self._values[key].extend(nested)
                    elif _is_scalar(nested):
                        self._values[key].append(nested)
                self._index(nested)
        elif isinstance(value, list):
            for item in value:
                self._index(item)

    def bind(self, spec: ToolSpec) -> list[dict[str, Any]]:
        schema = spec.input_schema
        required = [name for name in schema.get("required", []) if name != "case_id"]
        properties = schema.get("properties", {})
        if not required:
            return [{}]

        candidates: list[list[Any]] = []
        for name in required:
            values = _unique(self._values.get(name, []))
            if not values:
                return []
            property_schema = properties.get(name, {})
            if property_schema.get("type") == "array":
                candidates.append([values])
            else:
                candidates.append(values)

        return [
            dict(zip(required, combination, strict=True))
            for combination in list(product(*candidates))[:MAX_CALLS_PER_TOOL]
        ]


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


@dataclass(frozen=True)
class SpecialistAgent:
    actor: str
    tools: tuple[str, ...]

    async def investigate(
        self,
        *,
        case: dict[str, Any],
        collector: EvidenceCollector,
        context: Mapping[str, Any] | None = None,
        requested_tools: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        if case.get("case_id") != collector.case_id:
            raise ValueError("specialist case_id does not match the evidence collector scope")
        specs = await collector.gateway.discover_tools()
        binder = ToolArgumentBinder(case, context or {})
        facts: list[dict[str, Any]] = []
        missing_evidence: list[str] = []
        warnings: list[str] = []
        selected_tools = tuple(requested_tools) if requested_tools is not None else self.tools
        unauthorized = sorted(set(selected_tools) - set(self.tools))
        if unauthorized:
            raise PermissionError(
                f"specialist {self.actor!r} received out-of-scope tools: {unauthorized}"
            )

        for tool_name in selected_tools:
            require_tool_permission(self.actor, tool_name)
            spec = specs.get(tool_name)
            if spec is None:
                missing_evidence.append(f"TOOL_UNAVAILABLE:{tool_name}")
                continue
            calls = binder.bind(spec)
            if not calls:
                missing_evidence.append(f"MISSING_TOOL_ARGUMENTS:{tool_name}")
                continue
            for arguments in calls:
                try:
                    evidence = await collector.fetch(
                        actor=self.actor,
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                except MCPAuthorizationError:
                    raise
                except MCPToolError:
                    # A missing optional timeline/entity is not evidence and must never be
                    # synthesized. Preserve the gap so confidence and the final decision can
                    # degrade, while allowing independent authoritative sources to complete.
                    missing_evidence.append(f"TOOL_CALL_FAILED:{tool_name}")
                    continue
                decision_code = f"{evidence['domain'].upper()}_EVIDENCE_CONSUMED"
                collector.consume(
                    actor=self.actor,
                    tool_name=tool_name,
                    evidence=evidence,
                    decision_code=decision_code,
                )
                facts.append(
                    {
                        "tool_name": tool_name,
                        "domain": evidence["domain"],
                        "data": evidence["data"],
                        "evidence_refs": [evidence["evidence_ref"]],
                    }
                )
                warnings.extend(evidence.get("warnings", []))

        return {
            "actor": self.actor,
            "status": "completed" if facts and not missing_evidence else "partial",
            "facts": facts,
            "anomaly_flags": _unique(warnings),
            "missing_evidence": missing_evidence,
            "evidence_refs": _unique(
                ref for fact in facts for ref in fact["evidence_refs"]
            ),
        }


ORDER_SPECIALIST = SpecialistAgent(
    "order-item-agent",
    (
        "get_order",
        "get_order_items",
        "get_product_context",
        "get_sellers",
        "get_customer_history",
    ),
)
PAYMENT_SPECIALIST = SpecialistAgent(
    "payment-agent",
    ("get_order_payments", "get_payment_timeline", "get_refund_timeline"),
)
SHIPMENT_SPECIALIST = SpecialistAgent("shipment-agent", ("get_shipment_summary",))
POLICY_SPECIALIST = SpecialistAgent("policy-agent", ("get_policy",))
