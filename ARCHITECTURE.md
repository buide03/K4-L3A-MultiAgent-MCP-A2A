# L3A Architecture Record

This record describes observable, testable design decisions. It never contains API keys,
private prompts, chain-of-thought, or invented evidence. The JSON Schemas in
`contracts/schemas/` are immutable public contracts and take precedence over this document
and the implementation.

## 1. System overview

```text
inputs/<case_id>.json
          |
          v
+------------------------------------+
| Router & Triage Coordinator        |
| classify claims -> build case DAG  |
+------------------+-----------------+
                   | dependency-aware dynamic dispatch
       +-----------+--------------------+-------------------+
       v                                v                   v
+-------------------+        +-------------------+  +-------------------+
| Order/Item        |        | Payment           |  | Shipment          |
| Specialist        |        | Specialist        |  | Specialist        |
+---------+---------+        +---------+---------+  +---------+---------+
          |                            |                      |
          +----------------------------+----------------------+
                                       | calls only through
                                       v
                          +----------------------------+
                          | MCP Evidence Collector     |
                          | permission + schema + retry|
                          | case-local evidence ledger |
                          +-------------+--------------+
                                        | evidence-linked reports
                                        v
                          +----------------------------+
                          | Cross-Verification Node    |
                          | conflicts + missing facts  |
                          +-------------+--------------+
                                        | targeted refetch, max one round
                                        v
                          +----------------------------+
                          | Policy Specialist          |
                          | authoritative get_policy   |
                          +-------------+--------------+
                                        | normalized facts + policy
                                        v
                          +----------------------------+
                          | Judicial Engine            |
                          | semantic judgment only     |
                          +-------------+--------------+
                                        | decision proposal
                                        v
                          +----------------------------+
                          | Deterministic Builder      |
                          | exact fields + BRL math    |
                          +-------------+--------------+
                                        | raw contract-shaped dict
                                        v
                          +----------------------------+
                          | Deterministic Verifier     |
                          | JSON Schema + invariants   |
                          | provenance + trace linkage |
                          +-------------+--------------+
                                        |
                                        v
                              [FINAL L3A CONTRACT]
```

The agents are logical roles within one Python process and the bounded state machine is
implemented with LangGraph `StateGraph`. LangGraph is orchestration only and is not treated as
a source of business truth. The coordinator is the only component allowed to build the
per-case DAG, schedule work, or authorize a targeted refetch. This keeps routing deterministic
and prevents handoff loops.

The evidence collector is a shared service, not a decision-making agent. It wraps the MCP
gateway, enforces deny-by-default tool permissions, validates every response against
`mcp-evidence-response-v1.schema.json`, caches only within the current case, and preserves
server-issued `evidence_ref` and `result_hash` values unchanged.

The judicial engine may use an LLM to synthesize a semantic proposal from normalized facts
and authoritative policy. It cannot call MCP, invent entities or evidence, calculate the final
refund, serialize the final output, or bypass deterministic verification.

## 2. Agent ownership and tool permissions

| Actor | Input | Responsibility | Allowed MCP tools | Output / handoff |
| --- | --- | --- | --- | --- |
| `coordinator` | Raw case | Validate `case_id`, classify claims, build a dependency-aware DAG, schedule bounded work | None | Tasks to required specialists only |
| `order-item-agent` | Case identifiers and claims | Resolve orders, items, products, sellers, and affected entities | `get_order`, `get_order_items`, `get_product_context`, `get_sellers`, `get_customer_history` only for entity resolution | Entity map, order facts, local anomaly flags, evidence refs |
| `payment-agent` | Verified order IDs | Reconcile captures, split payments, duplicates, and refunds | `get_order_payments`, `get_payment_timeline`, `get_refund_timeline` | Payment facts, anomaly flags, monetary inputs, evidence refs |
| `shipment-agent` | Verified order/item IDs | Build delivery timeline and distinguish seller delay from logistics delay | `get_shipment_summary` | Shipment facts, anomaly flags, responsible candidates, evidence refs |
| `evidence-collector` | Authorized specialist requests | Enforce permission, call MCP, validate envelopes, retry transient failures, maintain the case ledger | All discovered tools, but only on behalf of an authorized actor | Immutable evidence bundles |
| `cross-verification-node` | Specialist reports and case ledger | Detect conflicts, missing required evidence, and unsupported local conclusions | None directly; requests one targeted refetch through coordinator | Normalized facts, conflicts, or one bounded refetch request |
| `policy-agent` | Cross-verified facts and candidate issue | Retrieve and apply authoritative policy constraints | `get_policy` | Policy facts and permitted resolution boundaries |
| `judicial-engine` | Normalized facts, conflicts, and policy evidence | Propose primary issue, status, cause, responsible party, claim verdicts, actions, and confidence | None | Semantic decision proposal; no final arithmetic or serialization |
| `deterministic-builder` | Semantic proposal plus verified monetary facts | Build only fields admitted by the L3A schema and calculate BRL totals | None | Contract-shaped raw dictionary |
| `deterministic-verifier` | Raw dictionary, trace index, and evidence ledger | Enforce JSON Schema, provenance, arithmetic, cross-field consistency, and confidence bounds | None | Validated L3A output or fail-closed rejection |

Permissions are deny-by-default. Tool names come from runtime discovery; the implementation
must fail closed if a required tool is absent and must never guess an alternative name.

## 3. A2A protocol and state machine

Internal messages are not public output fields. They carry a generated message ID, `case_id`,
sender, recipient, task type, attempt number, verified entity IDs, and existing evidence refs.
Only events and compact decision codes are written to the public trace.

```text
RECEIVED
   -> TRIAGED
   -> SPECIALIST_REVIEW (only selected DAG nodes; independent nodes may run concurrently)
   -> CROSS_VERIFICATION
        -> TARGETED_REFETCH (optional, maximum one round)
        -> CROSS_VERIFICATION
   -> POLICY_REVIEW
   -> JUDICIAL_PROPOSAL
   -> DETERMINISTIC_BUILD
   -> DETERMINISTIC_VERIFICATION
   -> FINALIZED
```

Rules:

1. Every task, response, MCP call, evidence record, trace event, and output is correlated by
   exactly one `case_id`.
2. Triage selects the smallest evidence DAG that can resolve the claims. It never calls every
   specialist by default.
3. Order/entity resolution runs before dependent tasks unless the input already supplies an
   order ID that evidence confirms. Payment and shipment work may then run concurrently.
4. Every assignment emits `task_assigned`; every completed specialist response emits
   `handoff` with sender and target.
5. Every evidence object actually used emits `tool_result_consumed` with its real tool name
   and evidence refs. Merely fetching evidence does not make it relevant to the final output.
6. Cross-verification may request exactly one targeted refetch round and must identify the
   missing fact, authorized actor, tool, and existing verified identifier. It cannot broaden
   the investigation without coordinator approval.
7. The policy decision emits `policy_decided`; deterministic verification emits
   `verification_completed`. The CLI owns `case_received` and `case_finalized`.
8. A task has at most two attempts (initial attempt plus one retry). The coordinator rejects a
   third attempt and never routes a task back to an earlier agent without a new decision code.
9. Internal messages and trace attributes contain facts and codes only, never private
   reasoning, raw prompts, secrets, or large MCP payloads.

## 4. Evidence lifecycle

1. A specialist requests an allowed, discovered tool through the evidence collector and
   passes the exact current `case_id`.
2. The gateway validates the returned envelope before it reaches an agent. Invalid envelopes
   are unusable and are never converted into fallback evidence.
3. The collector records `tool_name`, `domain`, `result_hash`, and the unchanged
   `evidence_ref` in a case-local ledger. Raw `data` remains in memory only as long as needed
   for that case.
4. Cache keys include `case_id`, tool name, and canonical arguments. No cache entry crosses a
   case boundary.
5. Specialist reports link each fact or verdict to supporting evidence refs. The verifier
   includes only evidence relevant to the submitted conclusion; unused calls are not added to
   output merely to increase evidence count.
6. The verifier rejects unknown, cross-team, cross-run, or cross-case evidence. Server audit
   remains the provenance authority.

The client validates the public format of `result_hash` and preserves it unchanged. It does
not recompute the digest unless the competition publishes a canonical serialization and hash
scope; hashing a locally re-serialized payload could otherwise create false mismatches.

Customer statements are claims, not evidence. Missing MCP evidence becomes
`insufficient_evidence` or `needs_investigation`; it never becomes guessed data, an invented
identifier, or a fabricated `evidence_ref`.

Policy evidence is isolated from operational facts. The `rules` catalog returned by
`get_policy` may contain every supported issue, so rule names and amounts are never treated as
facts about the current case. The engine first confirms a claim hypothesis with order,
payment, refund, shipment, item, or seller evidence, then selects only that issue's policy
rule for status, responsibility, action, and refund amount.

## 5. Failure and retry policy

| Failure | Retry | Fallback | Observable event / code |
| --- | --- | --- | --- |
| MCP session setup timeout or transient server error | Up to two retries with bounded backoff | Abort before processing cases if setup remains unavailable | No fabricated case result |
| MCP tool transport timeout | One retry with identical idempotent arguments | Return partial report; coordinator may select insufficient evidence | `handoff` / `MCP_TRANSIENT_FAILURE` |
| MCP tool-level error/no optional timeline | Cross-verification requests that exact tool once | Preserve the gap, lower confidence, and continue with independent evidence | `handoff` / `TARGETED_REFETCH_REQUIRED` |
| Authentication 401/403 | No per-case retry; abort the run | Operator must repair team credentials | No fabricated case result |
| Discovered tool missing | Refresh discovery once, no name guessing | Abort required branch or mark insufficient evidence | `handoff` / `TOOL_UNAVAILABLE` |
| Entity not found | No retry unless a distinct verified identifier exists | Empty entity set and insufficient evidence | `handoff` / `ENTITY_NOT_FOUND` |
| MCP response violates evidence schema | No retry as usable evidence | Reject response and branch | `handoff` / `INVALID_EVIDENCE_ENVELOPE` |
| Two authoritative sources conflict | No blind retry | Preserve both sources in `data_conflicts` and apply policy precedence | `policy_decided` / `SOURCE_CONFLICT_RESOLVED` |
| Specialist report is invalid | One correction request | Verifier rejects finalization | `handoff` / `SPECIALIST_REPORT_REJECTED` |
| Cross-verification finds a missing fact | One targeted refetch round through the original authorized specialist | Preserve the gap and lower confidence or select insufficient evidence | `task_assigned` / `TARGETED_REFETCH` |
| Output violates public schema | No fallback serialization | Abort that case; never write an invalid output | No `case_finalized` |

Retries are case-local and idempotent. Retrying must not mutate arguments, replace a missing ID
with a guessed value, or silently select a different tool.

## 6. Public contract boundary

Public schemas are not customized by the agent implementation:

- `l3a-output-v2.schema.json` defines the exact per-case output. Root and nested
  `additionalProperties: false` constraints prohibit debug fields, agent notes, reasoning,
  or raw evidence in the output.
- `trace-event-v1.schema.json` defines the only observable workflow event shape.
- `mcp-evidence-response-v1.schema.json` defines the server evidence envelope; the client
  preserves it and validates it before consumption.
- `submission-manifest-v2.schema.json` is generated by the packaging layer, not by agents.
- `l3b-output-v2.schema.json` is registered but is not used while `VARIANT_ID == "l3a"`.

`contracts/schema-lock.json` pins the complete schema inventory with SHA-256 digests computed
from canonical JSON. `Contracts` verifies this lock before loading validators, so an accidental
schema edit, deletion, or unregistered addition fails before any case is processed. Canonical
JSON hashing makes the lock independent of indentation and LF/CRLF line endings. Updating the
lock is an explicit contract-versioning operation, never an implementation workaround.

Dataclasses or Pydantic models may validate internal A2A messages, but they are never an
independent definition of public output. Final validation always calls the repository's
`Contracts.validate_output`, `validate_trace`, `validate_evidence`, and `validate_manifest`
methods against the checked-in JSON Schemas.

Precedence is absolute: JSON Schema, then registry/variant constants, then implementation,
then this document. Any implementation field that is not accepted by the selected schema is
an implementation defect, not a reason to extend the public contract.

## 7. Verification invariants

Before returning a result, the verifier enforces all of the following:

- output has exactly the L3A fields allowed by the schema and validates with
  `Contracts.validate_output`;
- output `case_id`, trace `case_id`, ledger scope, and input `case_id` match;
- every submitted evidence ref exists in the case ledger, was consumed by an observable
  specialist event, and supports at least one submitted claim or decision;
- affected entities were confirmed by evidence and contain no duplicates;
- claim verdicts and confidence values agree with their linked evidence;
- root-cause ranks are unique and contiguous, and responsible parties are supported by facts;
- `recommended_refund_brl` equals the sum of refund lines using deterministic BRL rounding;
- refund arithmetic uses the selected authoritative policy rule and deterministic Python BRL
  rounding, never an LLM-created number;
- case status, primary issue, refund, responsible party, and resolution actions do not
  contradict each other;
- an unsupported or insufficient claim is not assigned unjustifiably high confidence;
- no secret, raw prompt, chain-of-thought, or non-contract field appears in output or trace.

## 8. Reproducibility and packaging

- Runtime dependencies remain pinned by the ranges in `pyproject.toml`; LangGraph provides the
  graph runtime while public JSON Schema validation remains framework-independent.
- Specialists are routed dynamically and executed in deterministic dependency order. Cases
  are processed in `case-set.json` order so output and trace behavior remain understandable.
- Any randomness is limited to trace/message IDs and never affects the business decision.
- Commands are `day09 validate-inputs`, `day09 run`, `day09 validate`, and
  `day09 package --output dist/submission.zip`.
- Packaging includes only `manifest.json`, `trace.jsonl`, and per-case output JSON files.
- The released `case-set.json` and 100 L3A inputs are validated before a run. Tool arguments
  are bound from structured fields such as `claimed_order_id` and `policy_version` against
  runtime-discovered MCP schemas; identifiers are never parsed from customer prose.
