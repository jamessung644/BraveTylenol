"""Strict, deterministic bindings between L2 functions and MCP tools.

MCP discovery returns transport names and raw JSON Schemas.  Those values are
not sent to L2 directly: each discovered tool is compiled into a closed strict
function schema and an explicit model-name-to-transport-name binding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import SchemaError, validators

from lunit_hackathon.schemas import MCPTool

PROJECTION_VERSION = "strict-null-optionals-v1"

_CODEX_EXPOSED_PREFIX = "mcp__"
_LOCAL_TOOL_NAMES = frozenset(
    {
        "finalize_retrieval",
        "finalize_retrieval_rich_v2",
        "retrieve_relevant_content",
    }
)
_MODEL_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TRANSPORT_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_UNSUPPORTED_PROJECTION_KEYWORDS = frozenset(
    {
        "allOf",
        "contains",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
        "$dynamicRef",
        "$recursiveRef",
        "if",
        "maxContains",
        "maxProperties",
        "minContains",
        "minProperties",
        "not",
        "patternProperties",
        "propertyNames",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ToolBindingError(ValueError):
    """A discovered MCP tool cannot be exposed through a safe strict binding."""

    def __init__(self, message: str, *, code: str = "tool_binding_invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolBinding:
    """One approved, immutable model-function-to-MCP-tool projection."""

    model_function_name: str
    transport_tool_name: str
    description: str
    raw_input_schema: dict[str, Any]
    strict_input_schema: dict[str, Any]
    raw_schema_sha256: str
    strict_schema_sha256: str
    projection_version: str = PROJECTION_VERSION

    @property
    def model_schema_sha256(self) -> str:
        """Alias matching the terminology used by release manifests."""

        return self.strict_schema_sha256

    @property
    def argument_projection_version(self) -> str:
        return self.projection_version

    def as_chat_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.model_function_name,
                "description": self.description,
                "strict": True,
                "parameters": copy.deepcopy(self.strict_input_schema),
            },
        }

    def manifest_entry(self) -> dict[str, str]:
        return {
            "logical_alias": self.model_function_name,
            "transport_tool_name": self.transport_tool_name,
            "model_function_name": self.model_function_name,
            "raw_schema_sha256": self.raw_schema_sha256,
            "strict_wrapper_schema_sha256": self.strict_schema_sha256,
            "argument_projection_version": self.projection_version,
        }

    def project_arguments(self, model_arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate strict model args, omit null optionals, then validate raw args."""

        if not isinstance(model_arguments, dict):
            raise ToolBindingError(
                "Model tool arguments must be an object",
                code="model_arguments_invalid",
            )
        _validate_instance(
            model_arguments,
            self.strict_input_schema,
            message="Model tool arguments do not match the strict wrapper schema",
            code="model_arguments_invalid",
        )
        projected = _project_value(
            copy.deepcopy(model_arguments),
            self.raw_input_schema,
            self.raw_input_schema,
        )
        if not isinstance(projected, dict):
            raise ToolBindingError(
                "Projected MCP arguments must be an object",
                code="projected_arguments_invalid",
            )
        self.validate_transport_arguments(projected)
        return projected

    def validate_transport_arguments(self, arguments: dict[str, Any]) -> None:
        _validate_instance(
            arguments,
            self.raw_input_schema,
            message="Projected arguments do not match the raw MCP schema",
            code="projected_arguments_invalid",
        )


def compile_tool_binding(
    tool: MCPTool,
    *,
    model_function_name: str | None = None,
) -> ToolBinding:
    """Compile and fingerprint a discovered MCP tool's strict L2 wrapper."""

    transport_name = tool.name
    resolved_model_name = model_function_name or tool.model_function_name or transport_name
    _validate_tool_name(transport_name, model_facing=False)
    _validate_tool_name(resolved_model_name, model_facing=True)

    raw_schema = copy.deepcopy(tool.input_schema)
    _check_schema(raw_schema, code="raw_schema_invalid")
    if not _is_object_schema(raw_schema):
        raise ToolBindingError(
            "MCP input schema root must be an object",
            code="raw_schema_not_object",
        )

    strict_schema = _compile_schema(raw_schema, raw_schema, path="$", root=True)
    _check_schema(strict_schema, code="strict_schema_invalid")
    return ToolBinding(
        model_function_name=resolved_model_name,
        transport_tool_name=transport_name,
        description=tool.description,
        raw_input_schema=raw_schema,
        strict_input_schema=strict_schema,
        raw_schema_sha256=schema_sha256(raw_schema),
        strict_schema_sha256=schema_sha256(strict_schema),
    )


def compile_tool_bindings(
    tools: list[MCPTool],
    *,
    approved_aliases: Collection[str] | None = None,
    binding_seeds: Iterable[Mapping[str, Any]] | None = None,
    require_exact: bool = False,
) -> list[ToolBinding]:
    """Compile an immutable approved registry from live MCP discovery.

    Unknown discovered tools are quarantined by omission when an approved alias
    registry is supplied.  Every approved alias must resolve exactly once when
    ``require_exact`` is true.  A seed may pin a transport name and/or schema
    fingerprints after an in-network approval trial.
    """

    aliases = _approved_aliases(approved_aliases)
    seeds = _binding_seed_map(binding_seeds, aliases)
    if require_exact and aliases is None:
        raise ToolBindingError(
            "Exact binding compilation requires an approved alias registry",
            code="approved_registry_missing",
        )

    bindings: list[ToolBinding] = []
    model_names: set[str] = set()
    transport_names: set[str] = set()
    for tool in tools:
        model_name = _resolve_model_name(tool, aliases, seeds)
        if model_name is None:
            continue
        binding = compile_tool_binding(tool, model_function_name=model_name)
        if binding.model_function_name in model_names:
            raise ToolBindingError(
                "Duplicate model function binding",
                code="duplicate_model_function_name",
            )
        if binding.transport_tool_name in transport_names:
            raise ToolBindingError(
                "Duplicate MCP transport tool binding",
                code="duplicate_transport_tool_name",
            )
        _validate_seed_fingerprints(binding, seeds.get(binding.model_function_name))
        model_names.add(binding.model_function_name)
        transport_names.add(binding.transport_tool_name)
        bindings.append(binding)

    if require_exact and aliases is not None and model_names != aliases:
        raise ToolBindingError(
            "Live MCP discovery does not match the approved model-function registry",
            code="approved_registry_mismatch",
        )
    return sorted(bindings, key=lambda binding: binding.model_function_name)


def _approved_aliases(aliases: Collection[str] | None) -> set[str] | None:
    if aliases is None:
        return None
    values = list(aliases)
    if not values or len(values) != len(set(values)):
        raise ToolBindingError(
            "Approved aliases must be a non-empty unique registry",
            code="approved_registry_invalid",
        )
    for alias in values:
        if not isinstance(alias, str):
            raise ToolBindingError(
                "Approved aliases must be strings",
                code="approved_registry_invalid",
            )
        _validate_tool_name(alias, model_facing=True)
    return set(values)


def _binding_seed_map(
    seeds: Iterable[Mapping[str, Any]] | None,
    aliases: set[str] | None,
) -> dict[str, Mapping[str, Any]]:
    if seeds is None:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for seed in seeds:
        if not isinstance(seed, Mapping):
            raise ToolBindingError(
                "Binding seeds must be objects",
                code="binding_seed_invalid",
            )
        alias = seed.get("logical_alias")
        model_name = seed.get("model_function_name")
        if not isinstance(alias, str) or model_name != alias:
            raise ToolBindingError(
                "Binding seed logical and model names must match",
                code="binding_seed_invalid",
            )
        _validate_tool_name(alias, model_facing=True)
        if alias in result or (aliases is not None and alias not in aliases):
            raise ToolBindingError(
                "Binding seed registry is ambiguous",
                code="binding_seed_invalid",
            )
        transport_name = seed.get("transport_tool_name")
        if transport_name is not None:
            if not isinstance(transport_name, str):
                raise ToolBindingError(
                    "Pinned transport name must be a string",
                    code="binding_seed_invalid",
                )
            _validate_tool_name(transport_name, model_facing=False)
        for hash_key in ("raw_schema_sha256", "strict_wrapper_schema_sha256"):
            digest = seed.get(hash_key)
            if digest is not None and (
                not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
            ):
                raise ToolBindingError(
                    "Pinned schema fingerprint must be a lowercase SHA-256",
                    code="binding_seed_invalid",
                )
        if seed.get("status") not in {"requires_live_discovery", "approved"}:
            raise ToolBindingError(
                "Binding seed status is invalid",
                code="binding_seed_invalid",
            )
        result[alias] = seed
    if aliases is not None and set(result) != aliases:
        raise ToolBindingError(
            "Binding seeds do not match the approved alias registry",
            code="binding_seed_invalid",
        )
    return result


def _resolve_model_name(
    tool: MCPTool,
    aliases: set[str] | None,
    seeds: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if aliases is None:
        return tool.model_function_name or tool.name

    if tool.model_function_name is not None:
        alias = tool.model_function_name
        if alias not in aliases:
            return None
        seed = seeds.get(alias)
        pinned = seed.get("transport_tool_name") if seed is not None else None
        if pinned is not None and pinned != tool.name:
            raise ToolBindingError(
                "Discovered transport name drifted from its approved binding",
                code="transport_name_drift",
            )
        return alias

    candidates: list[str] = []
    if tool.name in aliases:
        seed = seeds.get(tool.name)
        pinned = seed.get("transport_tool_name") if seed is not None else None
        if pinned is None or pinned == tool.name:
            candidates.append(tool.name)
    candidates.extend(
        alias
        for alias, seed in seeds.items()
        if seed.get("transport_tool_name") == tool.name and alias not in candidates
    )
    if len(candidates) > 1:
        raise ToolBindingError(
            "One MCP transport name maps to multiple approved aliases",
            code="transport_binding_ambiguous",
        )
    return candidates[0] if candidates else None


def _validate_seed_fingerprints(
    binding: ToolBinding,
    seed: Mapping[str, Any] | None,
) -> None:
    if seed is None:
        return
    expected_raw = seed.get("raw_schema_sha256")
    if expected_raw is not None and expected_raw != binding.raw_schema_sha256:
        raise ToolBindingError(
            "Raw MCP schema fingerprint drifted",
            code="raw_schema_drift",
        )
    expected_strict = seed.get("strict_wrapper_schema_sha256")
    if expected_strict is not None and expected_strict != binding.strict_schema_sha256:
        raise ToolBindingError(
            "Strict wrapper schema fingerprint drifted",
            code="strict_schema_drift",
        )


def schema_sha256(schema: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ToolBindingError(
            "JSON Schema is not canonically serializable",
            code="schema_not_serializable",
        ) from error
    return hashlib.sha256(canonical).hexdigest()


def _validate_tool_name(name: str, *, model_facing: bool) -> None:
    if name.startswith(_CODEX_EXPOSED_PREFIX):
        raise ToolBindingError(
            "Codex-exposed MCP names are not runtime tool names",
            code="codex_tool_name_rejected",
        )
    if name in _LOCAL_TOOL_NAMES:
        raise ToolBindingError(
            "Harness-local tool names cannot be bound to MCP",
            code="local_tool_name_rejected",
        )
    pattern = _MODEL_FUNCTION_NAME if model_facing else _TRANSPORT_TOOL_NAME
    if not pattern.fullmatch(name):
        plane = "model function" if model_facing else "MCP transport tool"
        raise ToolBindingError(
            f"Invalid {plane} name",
            code="tool_name_invalid",
        )


def _check_schema(schema: dict[str, Any], *, code: str) -> None:
    if not isinstance(schema, dict):
        raise ToolBindingError("JSON Schema must be an object", code=code)
    try:
        validator = validators.validator_for(schema)
        validator.check_schema(schema)
    except SchemaError as error:
        raise ToolBindingError("JSON Schema is invalid", code=code) from error


def _compile_schema(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    path: str,
    root: bool = False,
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ToolBindingError(
            f"Boolean or non-object schema at {path} cannot be projected",
            code="strict_projection_unsupported",
        )
    unsupported = _UNSUPPORTED_PROJECTION_KEYWORDS.intersection(schema)
    if unsupported:
        raise ToolBindingError(
            f"Schema at {path} uses unsupported strict keywords",
            code="strict_projection_unsupported",
        )
    compiled = copy.deepcopy(schema)
    # These annotations either are unsupported by strict function schemas or
    # could make the model apply a server-side default before projection.  The
    # raw copy retains them and null omission lets the MCP server own defaults.
    for annotation in (
        "$comment",
        "$id",
        "$schema",
        "default",
        "deprecated",
        "examples",
        "readOnly",
        "writeOnly",
    ):
        compiled.pop(annotation, None)

    for definitions_key in ("$defs", "definitions"):
        definitions = compiled.get(definitions_key)
        if definitions is not None:
            if not isinstance(definitions, dict):
                raise ToolBindingError(
                    f"Invalid definitions at {path}",
                    code="strict_projection_unsupported",
                )
            compiled[definitions_key] = {
                name: _compile_schema(
                    definition,
                    root_schema,
                    path=f"{path}/{definitions_key}/{name}",
                )
                for name, definition in definitions.items()
            }

    for combinator in ("anyOf", "oneOf"):
        branches = compiled.get(combinator)
        if branches is not None:
            if not isinstance(branches, list) or not branches:
                raise ToolBindingError(
                    f"Invalid {combinator} at {path}",
                    code="strict_projection_unsupported",
                )
            compiled[combinator] = [
                _compile_schema(branch, root_schema, path=f"{path}/{combinator}/{index}")
                for index, branch in enumerate(branches)
            ]

    if _is_object_schema(schema):
        explicit_additional = schema.get("additionalProperties")
        if isinstance(explicit_additional, dict) or explicit_additional is True:
            raise ToolBindingError(
                f"Open object schema at {path} cannot be losslessly closed",
                code="strict_projection_not_lossless",
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ToolBindingError(
                f"Object properties at {path} must be an object",
                code="strict_projection_unsupported",
            )
        raw_required = schema.get("required", [])
        if not isinstance(raw_required, list) or any(
            not isinstance(name, str) for name in raw_required
        ):
            raise ToolBindingError(
                f"Object required list at {path} is invalid",
                code="strict_projection_unsupported",
            )
        if not set(raw_required).issubset(properties):
            raise ToolBindingError(
                f"Required property is not declared at {path}",
                code="strict_projection_not_lossless",
            )
        strict_properties: dict[str, Any] = {}
        required = set(raw_required)
        for name, property_schema in properties.items():
            if not isinstance(property_schema, dict):
                raise ToolBindingError(
                    f"Property schema at {path}/properties/{name} is invalid",
                    code="strict_projection_unsupported",
                )
            strict_property = _compile_schema(
                property_schema,
                root_schema,
                path=f"{path}/properties/{name}",
            )
            resolved_property = _resolve_schema(property_schema, root_schema)
            accepts_null = _schema_accepts_null(property_schema, root_schema)
            if name not in required:
                if accepts_null and resolved_property.get("default") is not None:
                    raise ToolBindingError(
                        f"Optional nullable property at {path}/properties/{name} has an "
                        "ambiguous non-null default",
                        code="strict_projection_not_lossless",
                    )
                if not accepts_null:
                    strict_property = {"anyOf": [strict_property, {"type": "null"}]}
            strict_properties[name] = strict_property
        compiled["type"] = "object"
        compiled["properties"] = strict_properties
        compiled["required"] = sorted(properties)
        compiled["additionalProperties"] = False
    elif root:
        raise ToolBindingError(
            "MCP input schema root must be an object",
            code="raw_schema_not_object",
        )

    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            compiled["items"] = _compile_schema(items, root_schema, path=f"{path}/items")
        elif isinstance(items, list):
            compiled["items"] = [
                _compile_schema(item, root_schema, path=f"{path}/items/{index}")
                for index, item in enumerate(items)
            ]
        else:
            raise ToolBindingError(
                f"Array items at {path} are unsupported",
                code="strict_projection_unsupported",
            )
    if "prefixItems" in schema:
        prefix_items = schema["prefixItems"]
        if not isinstance(prefix_items, list):
            raise ToolBindingError(
                f"Array prefixItems at {path} are unsupported",
                code="strict_projection_unsupported",
            )
        compiled["prefixItems"] = [
            _compile_schema(item, root_schema, path=f"{path}/prefixItems/{index}")
            for index, item in enumerate(prefix_items)
        ]
    return compiled


def _project_value(value: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> Any:
    resolved = _resolve_schema(schema, root_schema)

    for combinator in ("anyOf", "oneOf"):
        branches = resolved.get(combinator)
        if isinstance(branches, list):
            for branch in branches:
                try:
                    projected = _project_value(value, branch, root_schema)
                    _validate_instance(
                        projected,
                        branch,
                        message="Projected branch is invalid",
                        code="projected_arguments_invalid",
                        root_schema=root_schema,
                    )
                    return projected
                except ToolBindingError:
                    continue

    if isinstance(value, dict) and _is_object_schema(resolved):
        properties = resolved.get("properties", {})
        required = set(resolved.get("required", []))
        projected: dict[str, Any] = {}
        for key, item in value.items():
            property_schema = properties.get(key)
            if property_schema is None:
                projected[key] = item
            elif (
                item is None
                and key not in required
                and not _schema_accepts_null(property_schema, root_schema)
            ):
                continue
            else:
                projected[key] = _project_value(item, property_schema, root_schema)
        return projected

    if isinstance(value, list):
        items = resolved.get("items")
        if isinstance(items, dict):
            return [_project_value(item, items, root_schema) for item in value]
        prefix_items = resolved.get("prefixItems")
        if isinstance(prefix_items, list):
            return [
                _project_value(item, prefix_items[index], root_schema)
                if index < len(prefix_items)
                else item
                for index, item in enumerate(value)
            ]
    return value


def _resolve_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    if not reference.startswith("#/"):
        raise ToolBindingError(
            "External JSON Schema references are unsupported",
            code="strict_projection_unsupported",
        )
    if len(schema) != 1:
        raise ToolBindingError(
            "JSON Schema reference siblings are unsupported",
            code="strict_projection_unsupported",
        )
    resolved: Any = root_schema
    for token in reference[2:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, dict) or key not in resolved:
            raise ToolBindingError(
                "JSON Schema reference cannot be resolved",
                code="strict_projection_unsupported",
            )
        resolved = resolved[key]
    if not isinstance(resolved, dict):
        raise ToolBindingError(
            "JSON Schema reference does not resolve to an object",
            code="strict_projection_unsupported",
        )
    return resolved


def _schema_accepts_null(schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    resolved = _resolve_schema(schema, root_schema)
    schema_type = resolved.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if resolved.get("const", object()) is None:
        return True
    enum = resolved.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    for combinator in ("anyOf", "oneOf"):
        branches = resolved.get(combinator)
        if isinstance(branches, list) and any(
            _schema_accepts_null(branch, root_schema) for branch in branches
        ):
            return True
    if not any(
        key in resolved
        for key in ("type", "enum", "const", "anyOf", "oneOf", "allOf", "$ref")
    ):
        return True
    return False


def _is_object_schema(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    return schema_type == "object" or "properties" in schema


def _validate_instance(
    instance: Any,
    schema: dict[str, Any],
    *,
    message: str,
    code: str,
    root_schema: dict[str, Any] | None = None,
) -> None:
    validation_schema = root_schema or schema
    validator = validators.validator_for(validation_schema)(validation_schema)
    if root_schema is not None:
        validator = validator.evolve(schema=schema)
    if next(validator.iter_errors(instance), None) is not None:
        raise ToolBindingError(message, code=code)
