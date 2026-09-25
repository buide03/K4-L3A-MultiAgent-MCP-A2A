from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from . import VARIANT_ID


class ContractError(ValueError):
    pass


LOCK_VERSION = "day09-public-contract-lock-v1"
LOCK_ALGORITHM = "sha256-canonical-json"


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_lock(root: Path) -> dict[str, str]:
    path = root.parent / "schema-lock.json"
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"public contract lock is missing or invalid: {path}") from exc
    if not isinstance(lock, dict) or set(lock) != {"lock_version", "algorithm", "schemas"}:
        raise ContractError("public contract lock has unexpected or missing fields")
    if lock["lock_version"] != LOCK_VERSION or lock["algorithm"] != LOCK_ALGORITHM:
        raise ContractError("public contract lock version or algorithm is unsupported")
    digests = lock["schemas"]
    if not isinstance(digests, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in digests.items()
    ):
        raise ContractError("public contract lock schemas must map names to digests")
    return digests


class Contracts:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        expected_digests = _load_lock(self.root)
        paths = {path.name: path for path in self.root.glob("*.schema.json")}
        if set(paths) != set(expected_digests):
            missing = sorted(set(expected_digests) - set(paths))
            extra = sorted(set(paths) - set(expected_digests))
            message = f"public contract inventory mismatch; missing={missing}, extra={extra}"
            raise ContractError(message)
        schemas: dict[str, dict[str, Any]] = {}
        registry = Registry()
        for name, path in sorted(paths.items()):
            try:
                schema = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(f"public contract is not valid UTF-8 JSON: {name}") from exc
            actual_digest = _canonical_digest(schema)
            if actual_digest != expected_digests[name]:
                raise ContractError(f"public contract checksum mismatch: {name}")
            schemas[name] = schema
            resource = Resource.from_contents(schema)
            registry = registry.with_resource(schema["$id"], resource)
        self._schemas = schemas
        self._registry = registry

    def validate(self, schema_name: str, value: Any, label: str) -> None:
        schema = self._schemas.get(schema_name)
        if schema is None:
            raise ContractError(f"contract not found: {schema_name}")
        validator = Draft202012Validator(
            schema, registry=self._registry, format_checker=FormatChecker()
        )
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "$"
            raise ContractError(f"{label}:{location}: {error.message}")

    def validate_output(self, value: Any, label: str) -> None:
        self.validate(f"{VARIANT_ID}-output-v2.schema.json", value, label)

    def validate_trace(self, value: Any, label: str) -> None:
        self.validate("trace-event-v1.schema.json", value, label)

    def validate_manifest(self, value: Any, label: str = "manifest.json") -> None:
        self.validate("submission-manifest-v2.schema.json", value, label)

    def validate_evidence(self, value: Any, label: str = "MCP response") -> None:
        self.validate("mcp-evidence-response-v1.schema.json", value, label)
