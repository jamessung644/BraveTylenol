"""Explicit, read-only live canary for the organizer Model and MCP endpoints.

Importing this module and running it without both opt-in flags performs no network I/O.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from jsonschema import ValidationError, validators

from lunit_hackathon.artifacts import (
    DEFAULT_ARTIFACT_PATH,
    FINALIZE_RETRIEVAL_TOOL,
    MCP_BINDING_SEEDS,
    MCP_ENDPOINT,
    MCP_TOOL_ALIASES,
    MODEL_ENDPOINT,
    MODEL_NAME,
    RETRIEVE_RELEVANT_CONTENT_TOOL,
    RUNTIME_ARTIFACTS,
)
from lunit_hackathon.config import Settings
from lunit_hackathon.errors import LunitHackathonError
from lunit_hackathon.l2_client import L2Client
from lunit_hackathon.mcp_client import MCPClient
from lunit_hackathon.schemas import TOOL_CALL_FINISH_REASONS
from lunit_hackathon.tool_bindings import ToolBinding, ToolBindingError, compile_tool_bindings

_MCP_CANARY_ALIAS = "rag_get_all_data_sources"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")


class LiveCanaryError(RuntimeError):
    """The opt-in canary could not prove a required runtime contract."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ModelClientProtocol(Protocol):
    async def complete(self, **kwargs: Any) -> Any: ...


class ModelContextProtocol(Protocol):
    async def __aenter__(self) -> ModelClientProtocol: ...

    async def __aexit__(self, *args: object) -> None: ...


class MCPFactoryResultProtocol(Protocol):
    def connect(self) -> Any: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BraveTylenol Lunit-only live compatibility canary"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="permit calls to the two organizer endpoints",
    )
    parser.add_argument(
        "--confirm-lunit-network",
        action="store_true",
        help="assert that this process is running on the Lunit network",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.getenv("GIT_COMMIT_SHA"),
        help="40-character submission commit SHA to include in the report",
    )
    parser.add_argument(
        "--image-digest",
        default=os.getenv("SUBMISSION_IMAGE_DIGEST"),
        help="optional sha256 Docker image digest to include in the report",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def _validate_release_identifiers(
    commit_sha: str | None,
    image_digest: str | None,
) -> None:
    if commit_sha is not None and _SHA.fullmatch(commit_sha) is None:
        raise LiveCanaryError(
            "commit SHA must be a full lowercase 40-character SHA",
            code="commit_sha_invalid",
        )
    if image_digest is not None and _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise LiveCanaryError(
            "image digest must be a sha256 digest",
            code="image_digest_invalid",
        )


def _validate_live_target(settings: Settings) -> None:
    """Fail closed before network I/O if any target differs from the official manifest."""

    if settings.chat_completions_url != MODEL_ENDPOINT:
        raise LiveCanaryError(
            "Model endpoint differs from the official endpoint",
            code="model_endpoint_mismatch",
        )
    if settings.lunit_fm_model != MODEL_NAME:
        raise LiveCanaryError(
            "Model name differs from the official L2 preview model",
            code="model_name_mismatch",
        )
    if settings.mcp_url != MCP_ENDPOINT:
        raise LiveCanaryError(
            "MCP endpoint differs from the verified runtime manifest",
            code="mcp_endpoint_mismatch",
        )
    if not settings.api_key:
        raise LiveCanaryError(
            "LUNIT_FM_API_KEY is required for the live canary",
            code="credential_missing",
        )


def _decode_arguments(raw: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveCanaryError(
                    "Model tool arguments contain duplicate keys",
                    code="model_tool_arguments_invalid",
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise LiveCanaryError(
            f"Model tool arguments contain non-finite JSON number {token}",
            code="model_tool_arguments_invalid",
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise LiveCanaryError(
            "Model tool arguments are not valid JSON",
            code="model_tool_arguments_invalid",
        ) from error
    if not isinstance(value, dict):
        raise LiveCanaryError(
            "Model tool arguments are not an object",
            code="model_tool_arguments_invalid",
        )
    return value


async def _forced_tool_call(
    client: ModelClientProtocol,
    tool: Mapping[str, Any],
    *,
    prompt: str,
    offered_tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise LiveCanaryError("Canary tool is malformed", code="canary_tool_invalid")
    name = function["name"]
    tools = list(offered_tools) if offered_tools is not None else [tool]
    offered_names = [
        item.get("function", {}).get("name")
        for item in tools
        if isinstance(item, Mapping)
        and isinstance(item.get("function"), Mapping)
    ]
    if offered_names.count(name) != 1:
        raise LiveCanaryError(
            "Forced canary function must appear exactly once in the offered registry",
            code="canary_tool_invalid",
        )
    completion = await client.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "This is a transport compatibility canary. Call the forced function once. "
                    "Do not answer the medical question and do not claim that the tool ran."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": name}},
    )
    if (
        (isinstance(completion.content, str) and completion.content.strip())
        or len(completion.tool_calls) != 1
        or completion.tool_calls[0].function.name != name
        or completion.finish_reason not in TOOL_CALL_FINISH_REASONS
    ):
        raise LiveCanaryError(
            "L2 did not return a tool-only single forced tool call",
            code="model_strict_tool_rejected",
        )
    return _decode_arguments(completion.tool_calls[0].function.arguments)


def _validate_strict_arguments(tool: Mapping[str, Any], arguments: dict[str, Any]) -> None:
    function = tool["function"]
    schema = function["parameters"]
    try:
        validator_type = validators.validator_for(schema)
        validator_type.check_schema(schema)
        validator_type(schema).validate(arguments)
    except ValidationError as error:
        raise LiveCanaryError(
            "L2 returned arguments outside the strict local schema",
            code="model_strict_tool_arguments_invalid",
        ) from error


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _discover_bindings(connection: Any) -> list[ToolBinding]:
    discovered = await connection.list_tools()
    try:
        return compile_tool_bindings(
            discovered,
            approved_aliases=MCP_TOOL_ALIASES,
            binding_seeds=MCP_BINDING_SEEDS,
            require_exact=True,
        )
    except ToolBindingError as error:
        raise LiveCanaryError(str(error), code=error.code) from error


async def run_live_canary(
    settings: Settings,
    *,
    commit_sha: str | None = None,
    image_digest: str | None = None,
    model_factory: Callable[[Settings], ModelContextProtocol] = L2Client,
    mcp_factory: Callable[[Settings], MCPFactoryResultProtocol] = MCPClient,
) -> dict[str, Any]:
    """Run read-only compatibility checks after the CLI has obtained explicit opt-in."""

    _validate_release_identifiers(commit_sha, image_digest)
    _validate_live_target(settings)

    mcp = mcp_factory(settings)
    async with mcp.connect() as connection:
        bindings = await _discover_bindings(connection)
        binding_by_name = {binding.model_function_name: binding for binding in bindings}
        canary_binding = binding_by_name.get(_MCP_CANARY_ALIAS)
        if canary_binding is None:
            raise LiveCanaryError(
                "The read-only MCP canary alias was not discovered",
                code="mcp_canary_alias_missing",
            )

        async with model_factory(settings) as model:
            local_arguments = await _forced_tool_call(
                model,
                RETRIEVE_RELEVANT_CONTENT_TOOL,
                prompt=(
                    "Call retrieve_relevant_content with a short self-contained query that asks "
                    "which organizer MCP data sources are available."
                ),
            )
            _validate_strict_arguments(RETRIEVE_RELEVANT_CONTENT_TOOL, local_arguments)
            mcp_arguments = await _forced_tool_call(
                model,
                canary_binding.as_chat_tool(),
                prompt=(
                    f"Call {canary_binding.model_function_name} with schema-valid benign "
                    "arguments for a read-only live transport check."
                ),
                offered_tools=[
                    *(binding.as_chat_tool() for binding in bindings),
                    FINALIZE_RETRIEVAL_TOOL,
                ],
            )
            transport_arguments = canary_binding.project_arguments(mcp_arguments)

        mcp_result = await connection.call_tool(
            canary_binding.transport_tool_name,
            transport_arguments,
        )
        result_content = getattr(mcp_result, "content", None)
        if getattr(mcp_result, "is_error", True):
            raise LiveCanaryError(
                "The read-only MCP canary tool returned an error",
                code="mcp_canary_call_failed",
            )
        if (
            not isinstance(result_content, str)
            or not result_content
            or len(result_content) > settings.max_tool_result_chars
        ):
            raise LiveCanaryError(
                "The read-only MCP canary result was empty or unbounded",
                code="mcp_canary_result_invalid",
            )
        result_content_chars = len(result_content)

    manifest = [binding.manifest_entry() for binding in bindings]
    bundle = RUNTIME_ARTIFACTS.bundle
    prompts = bundle["prompts"]
    generation_phase_hashes = {
        phase: entry["sha256"]
        for phase, entry in prompts["generation"]["phases"].items()
    }
    generation_phase_registry_sha256 = _canonical_sha256(generation_phase_hashes)
    return {
        "schema_version": "lunit-live-canary-v1",
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "release": {
            "commit_sha": commit_sha or "not_recorded",
            "image_digest": image_digest or "not_recorded",
            "artifact_revision": bundle["artifact_revision"],
            "artifact_bundle_sha256": hashlib.sha256(
                DEFAULT_ARTIFACT_PATH.read_bytes()
            ).hexdigest(),
            "generation_prompt_sha256": generation_phase_registry_sha256,
            "generation_phase_registry_sha256": generation_phase_registry_sha256,
            "generation_phase_prompt_sha256": generation_phase_hashes,
            "retrieval_prompt_sha256": prompts["retrieval_dashboard_v1"]["sha256"],
            "live_binding_registry_sha256": _canonical_sha256(manifest),
        },
        "target": {
            "model_endpoint": settings.chat_completions_url,
            "model": settings.lunit_fm_model,
            "mcp_endpoint": settings.mcp_url,
        },
        "checks": {
            "mcp_exact_discovery_and_strict_binding": {
                "status": "passed",
                "approved_alias_count": len(MCP_TOOL_ALIASES),
                "bound_tool_count": len(bindings),
            },
            "generation_local_strict_tool": {
                "status": "passed",
                "function": RETRIEVE_RELEVANT_CONTENT_TOOL["function"]["name"],
            },
            "retrieval_mcp_strict_wrapper": {
                "status": "passed",
                "function": canary_binding.model_function_name,
                "transport_tool": canary_binding.transport_tool_name,
                "executed": True,
                "result_content_chars": result_content_chars,
            },
        },
    }


def _error_report(error: BaseException) -> dict[str, str]:
    if isinstance(error, LiveCanaryError):
        code = error.code
    elif isinstance(error, ToolBindingError):
        code = error.code
    elif isinstance(error, LunitHackathonError):
        code = type(error).__name__
    else:
        code = "unexpected_canary_error"
    return {
        "schema_version": "lunit-live-canary-v1",
        "status": "failed",
        "error_code": code,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.live and args.confirm_lunit_network):
        print(
            "No network request was made. Re-run on the Lunit network with both "
            "--live and --confirm-lunit-network."
        )
        return 2

    try:
        report = asyncio.run(
            run_live_canary(
                Settings(),
                commit_sha=args.commit_sha,
                image_digest=args.image_digest,
            )
        )
    except Exception as error:  # The CLI emits only a credential-safe error code.
        print(json.dumps(_error_report(error), ensure_ascii=False, sort_keys=True))
        return 1
    indent = None if args.compact else 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
