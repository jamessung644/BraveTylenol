import copy

import pytest

from lunit_hackathon.schemas import MCPTool
from lunit_hackathon.tool_bindings import (
    PROJECTION_VERSION,
    ToolBindingError,
    compile_tool_binding,
    compile_tool_bindings,
    schema_sha256,
)


def tool(
    name: str = "lookup",
    *,
    model_function_name: str | None = None,
    schema: dict | None = None,
) -> MCPTool:
    return MCPTool(
        name=name,
        model_function_name=model_function_name,
        description="lookup evidence",
        input_schema=schema
        or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def seed(
    alias: str,
    *,
    transport: str | None = None,
    raw_hash: str | None = None,
    strict_hash: str | None = None,
) -> dict:
    return {
        "logical_alias": alias,
        "model_function_name": alias,
        "transport_tool_name": transport,
        "raw_schema_sha256": raw_hash,
        "strict_wrapper_schema_sha256": strict_hash,
        "status": "requires_live_discovery",
    }


def test_strict_wrapper_requires_every_field_and_projects_null_optionals():
    raw_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "default": 10},
            "cursor": {"type": ["string", "null"]},
            "filters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "include_archived": {"type": "boolean"},
                },
                "required": ["region"],
                "additionalProperties": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    original = copy.deepcopy(raw_schema)

    binding = compile_tool_binding(tool(schema=raw_schema))
    chat_tool = binding.as_chat_tool()
    strict = chat_tool["function"]["parameters"]

    assert raw_schema == original
    assert chat_tool["function"]["strict"] is True
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["cursor", "filters", "limit", "query"]
    assert strict["properties"]["limit"]["anyOf"][-1] == {"type": "null"}
    assert "default" not in strict["properties"]["limit"]["anyOf"][0]
    assert strict["properties"]["filters"]["anyOf"][-1] == {"type": "null"}
    nested = strict["properties"]["filters"]["anyOf"][0]
    assert nested["required"] == ["include_archived", "region"]
    assert nested["additionalProperties"] is False

    projected = binding.project_arguments(
        {
            "query": "hypertension",
            "limit": None,
            "cursor": None,
            "filters": {"region": "KR", "include_archived": None},
        }
    )

    assert projected == {
        "query": "hypertension",
        "cursor": None,
        "filters": {"region": "KR"},
    }


def test_strict_wrapper_rejects_omitted_wrapper_fields_and_extra_arguments():
    binding = compile_tool_binding(
        tool(
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        )
    )

    with pytest.raises(ToolBindingError) as missing:
        binding.project_arguments({"query": "x"})
    with pytest.raises(ToolBindingError) as extra:
        binding.project_arguments({"query": "x", "limit": None, "extra": True})

    assert missing.value.code == "model_arguments_invalid"
    assert extra.value.code == "model_arguments_invalid"


def test_projection_resolves_local_refs_and_revalidates_raw_schema():
    binding = compile_tool_binding(
        tool(
            schema={
                "$defs": {
                    "filter": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "exact": {"type": "boolean"},
                        },
                        "required": ["term"],
                        "additionalProperties": False,
                    }
                },
                "type": "object",
                "properties": {"filter": {"$ref": "#/$defs/filter"}},
                "required": [],
                "additionalProperties": False,
            }
        )
    )

    assert binding.project_arguments(
        {"filter": {"term": "I10", "exact": None}}
    ) == {"filter": {"term": "I10"}}

    with pytest.raises(ToolBindingError) as caught:
        binding.validate_transport_arguments({"filter": {"term": 10}})
    assert caught.value.code == "projected_arguments_invalid"


def test_binding_keeps_model_and_transport_name_planes_separate():
    binding = compile_tool_binding(
        tool("server/v2/drug-search", model_function_name="adr_retrieve_drug_info")
    )

    assert binding.model_function_name == "adr_retrieve_drug_info"
    assert binding.transport_tool_name == "server/v2/drug-search"
    assert binding.as_chat_tool()["function"]["name"] == "adr_retrieve_drug_info"
    assert binding.projection_version == PROJECTION_VERSION
    assert binding.manifest_entry() == {
        "logical_alias": "adr_retrieve_drug_info",
        "transport_tool_name": "server/v2/drug-search",
        "model_function_name": "adr_retrieve_drug_info",
        "raw_schema_sha256": binding.raw_schema_sha256,
        "strict_wrapper_schema_sha256": binding.strict_schema_sha256,
        "argument_projection_version": PROJECTION_VERSION,
    }


@pytest.mark.parametrize(
    "name",
    [
        "mcp__lunit_mcp__kcd_get_name",
        "finalize_retrieval",
        "finalize_retrieval_rich_v2",
        "retrieve_relevant_content",
    ],
)
def test_binding_rejects_codex_and_harness_local_names(name):
    with pytest.raises(ToolBindingError) as caught:
        compile_tool_binding(tool(name))

    assert caught.value.code in {"codex_tool_name_rejected", "local_tool_name_rejected"}


def test_exact_registry_uses_seeds_and_quarantines_unknown_tools():
    aliases = ("alpha", "beta")
    seeds = (seed("alpha", transport="server/alpha"), seed("beta"))
    bindings = compile_tool_bindings(
        [
            tool("unknown_extra"),
            tool("server/alpha", model_function_name="alpha"),
            tool("beta"),
        ],
        approved_aliases=aliases,
        binding_seeds=seeds,
        require_exact=True,
    )

    assert [binding.model_function_name for binding in bindings] == ["alpha", "beta"]
    assert [binding.transport_tool_name for binding in bindings] == [
        "server/alpha",
        "beta",
    ]


def test_exact_registry_rejects_missing_or_duplicate_aliases():
    aliases = ("alpha", "beta")

    with pytest.raises(ToolBindingError) as missing:
        compile_tool_bindings(
            [tool("alpha")],
            approved_aliases=aliases,
            require_exact=True,
        )
    with pytest.raises(ToolBindingError) as duplicate:
        compile_tool_bindings(
            [
                tool("one", model_function_name="alpha"),
                tool("two", model_function_name="alpha"),
                tool("beta"),
            ],
            approved_aliases=aliases,
            require_exact=True,
        )

    assert missing.value.code == "approved_registry_mismatch"
    assert duplicate.value.code == "duplicate_model_function_name"


def test_pinned_schema_fingerprint_detects_live_drift():
    alias = "alpha"
    discovered = tool(alias)
    approved = compile_tool_binding(discovered)
    good_seed = seed(
        alias,
        raw_hash=approved.raw_schema_sha256,
        strict_hash=approved.strict_schema_sha256,
    )

    assert compile_tool_bindings(
        [discovered],
        approved_aliases=(alias,),
        binding_seeds=(good_seed,),
        require_exact=True,
    )[0].raw_schema_sha256 == approved.raw_schema_sha256

    drifted_seed = {**good_seed, "raw_schema_sha256": "0" * 64}
    with pytest.raises(ToolBindingError) as caught:
        compile_tool_bindings(
            [discovered],
            approved_aliases=(alias,),
            binding_seeds=(drifted_seed,),
            require_exact=True,
        )
    assert caught.value.code == "raw_schema_drift"


@pytest.mark.parametrize(
    ("schema", "code"),
    [
        (
            {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            "strict_projection_not_lossless",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"allOf": [{"type": "string"}]}},
                "additionalProperties": False,
            },
            "strict_projection_unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {"value": True},
                "additionalProperties": False,
            },
            "strict_projection_unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                        "default": "server-default",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            "strict_projection_not_lossless",
        ),
    ],
)
def test_binding_quarantines_non_lossless_or_unsupported_schemas(schema, code):
    with pytest.raises(ToolBindingError) as caught:
        compile_tool_binding(tool(schema=schema))

    assert caught.value.code == code


def test_schema_fingerprint_is_canonical_and_rejects_non_json_numbers():
    first = {
        "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
        "type": "object",
    }
    second = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }

    assert schema_sha256(first) == schema_sha256(second)

    with pytest.raises(ToolBindingError) as caught:
        schema_sha256({"default": float("nan")})
    assert caught.value.code == "schema_not_serializable"
