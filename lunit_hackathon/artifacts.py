"""Verified access to the generated Lunit L2/MCP runtime artifact bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = "lunit-runtime-artifacts-v1"
DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("runtime_artifacts") / "runtime_bundle_v1.json"
_RUNTIME_TOKENS = (
    "USER_LOCALE_OR_UNKNOWN",
    "CURRENT_DATE",
)
_GENERATION_PHASES = (
    "tool_decision",
    "direct_final",
    "post_retrieval_final",
    "mcp_failure_final",
    "emergency_final",
    "clean_recovery_final",
    "safe_completion_final",
)
_UPPER_TOKEN = re.compile(r"\[([A-Z][A-Z0-9_]+)\]")
_LOCALE = re.compile(r"(?:unknown|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)\Z")
_FORBIDDEN_PROMPT_SCAFFOLDING = (
    "의료법 담당자가 작성할 영역",
    "플레이스홀더가 채워지기 전",
    "상세 placeholder에서 담당자가 확정",
)
_FORBIDDEN_FINAL_PROMPT_PROTOCOL = (
    "retrieve_relevant_content",
    "finalize_retrieval",
    "<tool_call",
    "<tool_calls",
    "<function_call",
    '"tool_call"',
    '"tool_calls"',
    '"function_call"',
    "generation-reroute-v2",
    "reroute nonce",
    "retrieval mode:",
    "emergency_no_tool",
)


class RuntimeArtifactError(RuntimeError):
    """A generated artifact is missing, stale, malformed, or unsafe to use."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeArtifactError(f"duplicate artifact JSON key: {key}")
        result[key] = value
    return result


def _expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeArtifactError(f"{label} must be an object")
    return value


def _expect_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeArtifactError(f"{label} must be a non-empty string")
    return value


def _tool_hash(tool: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        tool,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _validate_sidecar(path: Path, payload: bytes) -> None:
    sidecar = path.with_suffix(".sha256")
    try:
        line = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise RuntimeArtifactError(f"artifact digest sidecar is unavailable: {sidecar}") from error
    expected = f"{_sha256(payload)}  {path.name}\n"
    if line != expected:
        raise RuntimeArtifactError("artifact bundle SHA-256 verification failed")


def _validate_local_tool(container: Mapping[str, Any], expected_name: str) -> dict[str, Any]:
    entry = _expect_object(container.get("entry"), f"local_tools.{expected_name}.entry")
    digest = _expect_string(container.get("sha256"), f"local_tools.{expected_name}.sha256")
    if _tool_hash(entry) != digest:
        raise RuntimeArtifactError(f"local tool hash mismatch: {expected_name}")
    if entry.get("type") != "function":
        raise RuntimeArtifactError(f"local tool type mismatch: {expected_name}")
    function = _expect_object(entry.get("function"), f"{expected_name}.function")
    if function.get("name") != expected_name or function.get("strict") is not True:
        raise RuntimeArtifactError(f"local tool contract mismatch: {expected_name}")
    parameters = _expect_object(function.get("parameters"), f"{expected_name}.parameters")
    if parameters.get("additionalProperties") is not False:
        raise RuntimeArtifactError(f"local tool is not closed: {expected_name}")
    properties = _expect_object(parameters.get("properties"), f"{expected_name}.properties")
    required = parameters.get("required")
    if not isinstance(required, list) or set(required) != set(properties):
        raise RuntimeArtifactError(f"local tool required fields mismatch: {expected_name}")
    return entry


@dataclass(frozen=True)
class RuntimeArtifacts:
    """A verified immutable view of model-facing build artifacts."""

    bundle: Mapping[str, Any]
    generation_template: str
    generation_phase_templates: Mapping[str, str]
    retrieval_prompt: str
    retrieve_tool: Mapping[str, Any]
    finalize_tool: Mapping[str, Any]
    model: str
    model_endpoint: str
    mcp_endpoint: str
    mcp_tool_aliases: tuple[str, ...]
    mcp_binding_seeds: tuple[Mapping[str, Any], ...]

    def local_tool(self, name: str) -> dict[str, Any]:
        if name == "retrieve_relevant_content":
            return copy.deepcopy(dict(self.retrieve_tool))
        if name == "finalize_retrieval":
            return copy.deepcopy(dict(self.finalize_tool))
        raise KeyError(name)

    def manifest_copy(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.bundle))


def load_runtime_artifacts(path: Path | str | None = None) -> RuntimeArtifacts:
    resolved = Path(path) if path is not None else DEFAULT_ARTIFACT_PATH
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise RuntimeArtifactError(f"runtime artifact is unavailable: {resolved}") from error
    _validate_sidecar(resolved, payload)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeArtifactError("runtime artifact is not valid UTF-8 JSON") from error
    bundle = _expect_object(value, "artifact")
    if bundle.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeArtifactError("unsupported runtime artifact schema")
    if bundle.get("default_agent_mode") != "hybrid":
        raise RuntimeArtifactError("runtime artifact default agent mode must be hybrid")

    prompts = _expect_object(bundle.get("prompts"), "prompts")
    generation = _expect_object(prompts.get("generation"), "prompts.generation")
    phases = _expect_object(generation.get("phases"), "prompts.generation.phases")
    if set(phases) != set(_GENERATION_PHASES):
        raise RuntimeArtifactError("Generation phase prompt registry mismatch")
    generation_phase_templates: dict[str, str] = {}
    generation_phase_hashes: set[str] = set()
    for phase in _GENERATION_PHASES:
        entry = _expect_object(phases.get(phase), f"Generation {phase} prompt")
        template = _expect_string(entry.get("template"), f"Generation {phase} template")
        digest = _expect_string(entry.get("sha256"), f"Generation {phase} hash")
        if _sha256(template.encode("utf-8")) != digest:
            raise RuntimeArtifactError(f"Generation {phase} prompt hash mismatch")
        if digest in generation_phase_hashes:
            raise RuntimeArtifactError("Generation phase prompts must be independently distinct")
        generation_phase_hashes.add(digest)
        placeholders = entry.get("runtime_placeholders")
        if placeholders != list(_RUNTIME_TOKENS):
            raise RuntimeArtifactError(
                f"Generation {phase} runtime placeholder registry mismatch"
            )
        counts = _expect_object(
            entry.get("runtime_placeholder_counts"),
            f"Generation {phase} runtime placeholder counts",
        )
        for token in _RUNTIME_TOKENS:
            count = counts.get(token)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise RuntimeArtifactError(
                    f"invalid Generation {phase} placeholder count: {token}"
                )
            if template.count(f"[{token}]") != count:
                raise RuntimeArtifactError(
                    f"Generation {phase} placeholder count drift: {token}"
                )
        remaining = set(_UPPER_TOKEN.findall(template))
        if remaining != set(_RUNTIME_TOKENS):
            raise RuntimeArtifactError(
                f"Generation {phase} template contains an unapproved placeholder"
            )
        if any(marker in template for marker in _FORBIDDEN_PROMPT_SCAFFOLDING):
            raise RuntimeArtifactError(
                f"Generation {phase} template contains build-only scaffolding"
            )
        if phase != "tool_decision" and any(
            marker in template.casefold() for marker in _FORBIDDEN_FINAL_PROMPT_PROTOCOL
        ):
            raise RuntimeArtifactError(
                f"Generation {phase} template contains tool protocol"
            )
        generation_phase_templates[phase] = template
    generation_template = generation_phase_templates["tool_decision"]

    retrieval = _expect_object(
        prompts.get("retrieval_dashboard_v1"),
        "prompts.retrieval_dashboard_v1",
    )
    retrieval_prompt = _expect_string(retrieval.get("text"), "Retrieval prompt")
    if _sha256(retrieval_prompt.encode("utf-8")) != retrieval.get("sha256"):
        raise RuntimeArtifactError("Retrieval prompt hash mismatch")
    if _UPPER_TOKEN.search(retrieval_prompt):
        raise RuntimeArtifactError("Retrieval prompt contains an unresolved placeholder")
    if any(marker in retrieval_prompt for marker in _FORBIDDEN_PROMPT_SCAFFOLDING):
        raise RuntimeArtifactError("Retrieval prompt contains build-only scaffolding")

    local_tools = _expect_object(bundle.get("local_tools"), "local_tools")
    retrieve_tool = _validate_local_tool(
        _expect_object(local_tools.get("retrieve_relevant_content"), "retrieve tool"),
        "retrieve_relevant_content",
    )
    finalize_tool = _validate_local_tool(
        _expect_object(local_tools.get("finalize_retrieval"), "finalize tool"),
        "finalize_retrieval",
    )

    model = _expect_string(bundle.get("model"), "model")
    model_endpoint = _expect_string(bundle.get("model_endpoint"), "model_endpoint")
    if model != "Lunit/L2-preview":
        raise RuntimeArtifactError("runtime artifact model differs from the official L2 model")
    if model_endpoint != "https://model.hackathon.lunit.io/v1/chat/completions":
        raise RuntimeArtifactError(
            "runtime artifact Model endpoint differs from the official endpoint"
        )

    mcp = _expect_object(bundle.get("mcp"), "mcp")
    endpoint = _expect_string(mcp.get("endpoint"), "mcp.endpoint")
    aliases = mcp.get("logical_aliases")
    if (
        not isinstance(aliases, list)
        or not aliases
        or not all(isinstance(alias, str) and alias for alias in aliases)
        or aliases != sorted(set(aliases))
    ):
        raise RuntimeArtifactError("MCP logical aliases must be a sorted unique string list")
    if any(alias.startswith("mcp__") for alias in aliases):
        raise RuntimeArtifactError("Codex-exposed MCP names are forbidden in the runtime registry")
    registered_function_names = {
        "retrieve_relevant_content",
        "finalize_retrieval",
        *aliases,
    }
    for phase, template in generation_phase_templates.items():
        if phase == "tool_decision":
            continue
        exposed = sorted(
            name
            for name in registered_function_names
            if name.casefold() in template.casefold()
        )
        if exposed:
            raise RuntimeArtifactError(
                f"Generation {phase} template exposes registered function names"
            )
    seeds = mcp.get("binding_seeds")
    if not isinstance(seeds, list) or len(seeds) != len(aliases):
        raise RuntimeArtifactError("MCP binding seed registry mismatch")
    for alias, seed_value in zip(aliases, seeds, strict=True):
        seed = _expect_object(seed_value, f"MCP binding seed {alias}")
        if seed.get("logical_alias") != alias or seed.get("model_function_name") != alias:
            raise RuntimeArtifactError(f"MCP binding seed name mismatch: {alias}")
        if seed.get("status") != "requires_live_discovery":
            raise RuntimeArtifactError(f"MCP binding seed status mismatch: {alias}")

    return RuntimeArtifacts(
        bundle=bundle,
        generation_template=generation_template,
        generation_phase_templates=generation_phase_templates,
        retrieval_prompt=retrieval_prompt,
        retrieve_tool=retrieve_tool,
        finalize_tool=finalize_tool,
        model=model,
        model_endpoint=model_endpoint,
        mcp_endpoint=endpoint,
        mcp_tool_aliases=tuple(aliases),
        mcp_binding_seeds=tuple(seeds),
    )


def render_generation_phase_prompt(
    phase: str,
    *,
    user_locale: str = "unknown",
    current_date: str | date,
) -> str:
    """Render trusted runtime values without interpreting user-controlled format syntax."""

    if phase not in _GENERATION_PHASES:
        raise RuntimeArtifactError("invalid Generation phase")
    if not isinstance(user_locale, str) or not _LOCALE.fullmatch(user_locale):
        raise RuntimeArtifactError("invalid Generation locale")
    if isinstance(current_date, date):
        date_value = current_date.isoformat()
    elif isinstance(current_date, str):
        try:
            parsed = date.fromisoformat(current_date)
        except ValueError as error:
            raise RuntimeArtifactError("invalid Generation current date") from error
        date_value = parsed.isoformat()
        if date_value != current_date:
            raise RuntimeArtifactError("Generation current date must be canonical ISO-8601")
    else:
        raise RuntimeArtifactError("invalid Generation current date")
    values = {
        "USER_LOCALE_OR_UNKNOWN": user_locale,
        "CURRENT_DATE": date_value,
    }
    rendered = GENERATION_PHASE_PROMPT_TEMPLATES[phase]
    for token, replacement in values.items():
        rendered = rendered.replace(f"[{token}]", replacement)
    if _UPPER_TOKEN.search(rendered):
        raise RuntimeArtifactError("rendered Generation prompt contains an unresolved placeholder")
    return rendered


def render_generation_system_prompt(
    *,
    user_locale: str = "unknown",
    current_date: str | date,
) -> str:
    """Compatibility renderer for the tool-decision phase."""

    return render_generation_phase_prompt(
        "tool_decision",
        user_locale=user_locale,
        current_date=current_date,
    )


RUNTIME_ARTIFACTS = load_runtime_artifacts()
GENERATION_PHASE_PROMPT_TEMPLATES = RUNTIME_ARTIFACTS.generation_phase_templates
GENERATION_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES["tool_decision"]
DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES["direct_final"]
POST_RETRIEVAL_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES[
    "post_retrieval_final"
]
MCP_FAILURE_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES[
    "mcp_failure_final"
]
EMERGENCY_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES["emergency_final"]
CLEAN_RECOVERY_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES[
    "clean_recovery_final"
]
SAFE_COMPLETION_FINAL_SYSTEM_PROMPT_TEMPLATE = GENERATION_PHASE_PROMPT_TEMPLATES[
    "safe_completion_final"
]
RETRIEVAL_SYSTEM_PROMPT = RUNTIME_ARTIFACTS.retrieval_prompt
RETRIEVE_RELEVANT_CONTENT_TOOL = RUNTIME_ARTIFACTS.local_tool("retrieve_relevant_content")
FINALIZE_RETRIEVAL_TOOL = RUNTIME_ARTIFACTS.local_tool("finalize_retrieval")
MCP_ENDPOINT = RUNTIME_ARTIFACTS.mcp_endpoint
MODEL_NAME = RUNTIME_ARTIFACTS.model
MODEL_ENDPOINT = RUNTIME_ARTIFACTS.model_endpoint
MCP_TOOL_ALIASES = RUNTIME_ARTIFACTS.mcp_tool_aliases
MCP_BINDING_SEEDS = RUNTIME_ARTIFACTS.mcp_binding_seeds


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CLEAN_RECOVERY_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "EMERGENCY_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "FINALIZE_RETRIEVAL_TOOL",
    "GENERATION_SYSTEM_PROMPT_TEMPLATE",
    "GENERATION_PHASE_PROMPT_TEMPLATES",
    "MCP_BINDING_SEEDS",
    "MCP_ENDPOINT",
    "MCP_TOOL_ALIASES",
    "MODEL_ENDPOINT",
    "MODEL_NAME",
    "MCP_FAILURE_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "POST_RETRIEVAL_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "RETRIEVAL_SYSTEM_PROMPT",
    "RETRIEVE_RELEVANT_CONTENT_TOOL",
    "SAFE_COMPLETION_FINAL_SYSTEM_PROMPT_TEMPLATE",
    "RUNTIME_ARTIFACTS",
    "RuntimeArtifactError",
    "RuntimeArtifacts",
    "load_runtime_artifacts",
    "render_generation_phase_prompt",
    "render_generation_system_prompt",
]
