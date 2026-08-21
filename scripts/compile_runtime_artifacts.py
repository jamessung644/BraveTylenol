#!/usr/bin/env python3
"""Compile canonical L2/MCP Markdown into a deterministic runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "lunit-runtime-artifacts-v1"
OFFICIAL_MODEL = "Lunit/L2-preview"
OFFICIAL_MODEL_ENDPOINT = "https://model.hackathon.lunit.io/v1/chat/completions"
OFFICIAL_MCP_ENDPOINT = "https://mcp.hackathon.lunit.io/mcp"
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "canonical_sources"
DEFAULT_RUNTIME_SOURCES = Path(__file__).resolve().parents[1] / "runtime_sources"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "lunit_hackathon"
    / "runtime_artifacts"
    / "runtime_bundle_v1.json"
)

CANONICAL_DOCUMENTS = (
    "10_MCP_CATALOG.md",
    "14_LUNIT_RUNTIME_APPLICATION_BLUEPRINT.md",
    "20_GENERATION_PROMPT.md",
    "30_RETRIEVAL_PROMPT.md",
    "45_RUNTIME_RESILIENCE.md",
)
RUNTIME_PLACEHOLDERS = (
    "USER_LOCALE_OR_UNKNOWN",
    "CURRENT_DATE",
)
BUILD_PLACEHOLDERS = (
    "FINAL_LEGAL_POLICY_PLACEHOLDER",
    "MCP_ROUTING_PROMPT_FRAGMENT",
    "MAX_RETRIEVAL_CALLS",
    "LEGAL_POLICY_PLACEHOLDER",
    "LEGAL_CITATION_POLICY_PLACEHOLDER",
    "LEGAL_MCP_ROUTING_PLACEHOLDER",
    "LEGAL_JURISDICTION_PLACEHOLDER",
    "LEGAL_SOURCE_PRIORITY_PLACEHOLDER",
    "LEGAL_EFFECTIVE_DATE_PLACEHOLDER",
)
_UPPER_TOKEN = re.compile(r"\[([A-Z][A-Z0-9_]+)\]")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TOOL_NAME = re.compile(r"`([a-z][a-z0-9_]+)`")
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
MAX_FINAL_ANSWER_POLICY_CHARS = 1_200
MAX_FINAL_GENERATION_PROMPT_CHARS = 3_000


class CompileError(RuntimeError):
    """The canonical sources cannot produce a safe, deterministic artifact."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompileError(f"cannot read JSON build input: {path}") from error
    if not isinstance(value, dict):
        raise CompileError(f"JSON build input must be an object: {path}")
    return value


def _read_document(source_root: Path, name: str) -> tuple[str, bytes]:
    path = source_root / "docs" / "model-instructions" / name
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise CompileError(f"cannot read canonical source: {path}") from error
    return text, raw


def _extract_unique_fence(text: str, language: str, label: str) -> str:
    pattern = re.compile(
        rf"^```{re.escape(language)}[ \t]*\n(.*?)^```[ \t]*$",
        re.MULTILINE | re.DOTALL,
    )
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise CompileError(f"{label} must contain exactly one {language} fence")
    return matches[0].rstrip("\n")


def _heading_section(text: str, heading: str) -> str:
    lines = text.splitlines(keepends=True)
    start: int | None = None
    level: int | None = None
    for index, line in enumerate(lines):
        match = _HEADING.match(line.rstrip("\n"))
        if match and match.group(2) == heading:
            if start is not None:
                raise CompileError(f"duplicate heading: {heading}")
            start = index + 1
            level = len(match.group(1))
    if start is None or level is None:
        raise CompileError(f"missing heading: {heading}")
    end = len(lines)
    for index in range(start, len(lines)):
        match = _HEADING.match(lines[index].rstrip("\n"))
        if match and len(match.group(1)) <= level:
            end = index
            break
    return "".join(lines[start:end])


def _replace_required(text: str, token: str, replacement: str) -> str:
    marker = f"[{token}]"
    count = text.count(marker)
    if count != 1:
        raise CompileError(f"{marker} must appear exactly once; found {count}")
    if _UPPER_TOKEN.search(replacement):
        raise CompileError(f"replacement for {marker} contains an unresolved token")
    return text.replace(marker, replacement)


def _replace_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise CompileError(f"{label} must appear exactly once; found {count}")
    return text.replace(old, new)


def _insert_after_exact_once(text: str, marker: str, addition: str, label: str) -> str:
    count = text.count(marker)
    if count != 1:
        raise CompileError(f"{label} anchor must appear exactly once; found {count}")
    if not addition.strip():
        raise CompileError(f"{label} must not be empty")
    if _UPPER_TOKEN.search(addition):
        raise CompileError(f"{label} contains an unresolved token")
    return text.replace(marker, f"{marker}\n\n{addition.strip()}", 1)


def _compile_legal_generation(text: str, legal: Mapping[str, str]) -> str:
    text = _replace_required(
        text,
        "LEGAL_POLICY_PLACEHOLDER",
        legal["LEGAL_POLICY_PLACEHOLDER"],
    )
    text = _replace_exact_once(
        text,
        "의료법 담당자가 작성할 영역이다. 의료법 관련 허용·제한 동작, 표현 경계, "
        "고지 방식, 전문가 연결 기준을 여기에 삽입한다.\n",
        "",
        "Generation legal authoring scaffold",
    )
    text = _replace_exact_once(
        text,
        "플레이스홀더가 채워지기 전에는 이 블록을 production system prompt로 배포하지 "
        "않는다.\n",
        "",
        "Generation legal deployment scaffold",
    )
    citation_marker = "[LEGAL_CITATION_POLICY_PLACEHOLDER]"
    citation_line = f"- 법령 인용 규칙은 {citation_marker}에 정의한다."
    text = _replace_exact_once(
        text,
        citation_line,
        f"- {legal['LEGAL_CITATION_POLICY_PLACEHOLDER']}",
        "Generation legal citation placeholder sentence",
    )
    return text


def _compile_final_generation(
    text: str,
    legal: Mapping[str, str],
    natural_language_policy: str,
) -> str:
    text = _replace_required(
        text,
        "FINAL_LEGAL_POLICY_PLACEHOLDER",
        legal["FINAL_LEGAL_POLICY_PLACEHOLDER"],
    )
    return _insert_after_exact_once(
        text,
        "</runtime_context>",
        natural_language_policy,
        "Generation final natural-language policy",
    )


def _compile_legal_routing(text: str, legal: Mapping[str, str]) -> str:
    routing_marker = "[LEGAL_MCP_ROUTING_PLACEHOLDER]"
    text = _replace_exact_once(
        text,
        f"세부 관할·현행성·fallback 규칙은 {routing_marker}에 정의한다.",
        legal["LEGAL_MCP_ROUTING_PLACEHOLDER"],
        "MCP legal routing placeholder sentence",
    )
    jurisdiction_marker = "[LEGAL_JURISDICTION_PLACEHOLDER]"
    text = _replace_exact_once(
        text,
        f"법률 질문의 지역·관할 처리 규칙은 {jurisdiction_marker}에 정의한다.",
        legal["LEGAL_JURISDICTION_PLACEHOLDER"],
        "MCP legal jurisdiction placeholder sentence",
    )
    return text


def _compile_legal_retrieval(text: str, legal: Mapping[str, str]) -> str:
    priority_marker = "[LEGAL_SOURCE_PRIORITY_PLACEHOLDER]"
    text = _replace_exact_once(
        text,
        f"법령 및 규제 자료의 우선순위는 {priority_marker}에 별도로 정의한다.",
        legal["LEGAL_SOURCE_PRIORITY_PLACEHOLDER"],
        "Retrieval legal priority placeholder sentence",
    )
    text = _replace_exact_once(
        text,
        "의료법의 확인된 기본 순서는 Lunit 법령 MCP의 조문 원문, 제공 의료법 PDF "
        "스냅샷, 2019년 비의료 건강관리서비스 가이드라인, 보도자료다. "
        "판례·하위법령·행정해석을 포함한 최종 우선순위는 상세 placeholder에서 "
        "담당자가 확정한다.\n\n",
        "",
        "Retrieval obsolete legal placeholder hierarchy",
    )
    effective_marker = "[LEGAL_EFFECTIVE_DATE_PLACEHOLDER]"
    text = _replace_exact_once(
        text,
        f"- 법률 자료의 날짜·버전 규칙은 {effective_marker}에 정의한다.",
        f"- {legal['LEGAL_EFFECTIVE_DATE_PLACEHOLDER']}",
        "Retrieval legal date placeholder sentence",
    )
    return text


def _extract_catalog_aliases(catalog: str) -> list[str]:
    section = _heading_section(catalog, "기본 도구 목록")
    aliases: list[str] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        columns = line.split("|")
        if len(columns) < 4:
            continue
        aliases.extend(_TOOL_NAME.findall(columns[2]))
    aliases = sorted(set(aliases))
    if not aliases:
        raise CompileError("MCP catalog contains no logical tool aliases")
    for alias in aliases:
        if alias.startswith("mcp__") or alias in {
            "finalize_retrieval",
            "retrieve_relevant_content",
        }:
            raise CompileError(f"invalid MCP logical alias: {alias}")
    return aliases


def _validate_strict_schema(schema: Mapping[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict):
            raise CompileError(f"{path}.properties must be an object")
        if schema.get("additionalProperties") is not False:
            raise CompileError(f"{path}.additionalProperties must be false")
        if not isinstance(required, list) or set(required) != set(properties):
            raise CompileError(f"{path}.required must exactly match properties")
        if len(required) != len(set(required)):
            raise CompileError(f"{path}.required contains duplicates")
        for name, child in properties.items():
            if not isinstance(child, dict):
                raise CompileError(f"{path}.properties.{name} must be an object")
            _validate_strict_schema(child, f"{path}.properties.{name}")
    elif schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            raise CompileError(f"{path}.items must be an object")
        _validate_strict_schema(items, f"{path}.items")


def _validate_local_tool(tool: Mapping[str, Any], expected_name: str) -> None:
    if set(tool) != {"type", "function"} or tool.get("type") != "function":
        raise CompileError(f"{expected_name} must be an exact function tool entry")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise CompileError(f"{expected_name}.function must be an object")
    if function.get("name") != expected_name:
        raise CompileError(f"unexpected local tool name for {expected_name}")
    if function.get("strict") is not True:
        raise CompileError(f"{expected_name} must set function.strict=true")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        raise CompileError(f"{expected_name}.parameters must be an object")
    _validate_strict_schema(parameters, f"{expected_name}.parameters")


def _extract_tool(document: str, heading: str, expected_name: str) -> dict[str, Any]:
    section = _heading_section(document, heading)
    raw = _extract_unique_fence(section, "json", expected_name)
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_object)
    except json.JSONDecodeError as error:
        raise CompileError(f"invalid canonical JSON for {expected_name}") from error
    if not isinstance(value, dict):
        raise CompileError(f"canonical {expected_name} tool must be an object")
    _validate_local_tool(value, expected_name)
    return value


def _tool_hash(tool: Mapping[str, Any]) -> str:
    compact = json.dumps(
        tool,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(compact)


def compile_bundle(source_root: Path, runtime_sources: Path) -> dict[str, Any]:
    config_path = runtime_sources / "release_config_v1.json"
    config = _load_json(config_path)
    expected_config = {
        "artifact_revision",
        "dashboard_verified_date",
        "default_agent_mode",
        "finalizer_contract",
        "legal_policy_file",
        "max_retrieval_calls",
        "mcp_endpoint",
        "mcp_transport",
        "model",
        "model_endpoint",
        "natural_language_policy_file",
        "schema_version",
    }
    if set(config) != expected_config:
        missing = sorted(expected_config - set(config))
        extra = sorted(set(config) - expected_config)
        raise CompileError(
            f"runtime source config mismatch; missing={missing}, extra={extra}"
        )
    if config.get("schema_version") != "runtime-source-config-v1":
        raise CompileError("unsupported runtime source config schema")
    if config.get("finalizer_contract") != "dashboard_v1":
        raise CompileError("unsupported Retrieval finalizer contract")
    if config.get("default_agent_mode") != "hybrid":
        raise CompileError("submission runtime must default to hybrid mode")
    if config.get("mcp_transport") != "streamable_http":
        raise CompileError("unsupported MCP transport")
    if config.get("model") != OFFICIAL_MODEL:
        raise CompileError("release model differs from the official L2 model")
    if config.get("model_endpoint") != OFFICIAL_MODEL_ENDPOINT:
        raise CompileError("release Model endpoint differs from the official endpoint")
    if config.get("mcp_endpoint") != OFFICIAL_MCP_ENDPOINT:
        raise CompileError("release MCP endpoint differs from the official endpoint")
    max_calls = config.get("max_retrieval_calls")
    if not isinstance(max_calls, int) or isinstance(max_calls, bool) or not 1 <= max_calls <= 12:
        raise CompileError("max_retrieval_calls must be an integer from 1 to 12")

    legal_path = runtime_sources / str(config.get("legal_policy_file", ""))
    legal = _load_json(legal_path)
    expected_legal = {
        token
        for token in BUILD_PLACEHOLDERS
        if token.startswith("LEGAL_") or token == "FINAL_LEGAL_POLICY_PLACEHOLDER"
    }
    if set(legal) != expected_legal:
        missing = sorted(expected_legal - set(legal))
        extra = sorted(set(legal) - expected_legal)
        raise CompileError(f"legal placeholder map mismatch; missing={missing}, extra={extra}")
    if not all(isinstance(value, str) and value.strip() for value in legal.values()):
        raise CompileError("all legal placeholder replacements must be non-empty strings")

    natural_language_path = runtime_sources / str(
        config.get("natural_language_policy_file", "")
    )
    natural_language = _load_json(natural_language_path)
    expected_natural_language = {
        "final_answer_policy",
        "schema_version",
        "generation_policy",
        "retrieval_policy",
    }
    if set(natural_language) != expected_natural_language:
        missing = sorted(expected_natural_language - set(natural_language))
        extra = sorted(set(natural_language) - expected_natural_language)
        raise CompileError(
            "natural-language policy mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if natural_language["schema_version"] != "natural-language-policy-ko-v1":
        raise CompileError("unsupported natural-language policy schema")
    if not all(
        isinstance(natural_language[field], str) and natural_language[field].strip()
        for field in ("final_answer_policy", "generation_policy", "retrieval_policy")
    ):
        raise CompileError("natural-language prompt policies must be non-empty strings")
    if len(natural_language["final_answer_policy"]) > MAX_FINAL_ANSWER_POLICY_CHARS:
        raise CompileError(
            "final-answer natural-language policy exceeds compact character budget"
        )

    documents: dict[str, str] = {}
    source_manifest: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_DOCUMENTS:
        text, raw = _read_document(source_root, name)
        documents[name] = text
        source_manifest[name] = {
            "path": f"docs/model-instructions/{name}",
            "sha256": _sha256_bytes(raw),
        }
    for path in (config_path, legal_path, natural_language_path):
        source_manifest[f"runtime_sources/{path.name}"] = {
            "path": f"runtime_sources/{path.name}",
            "sha256": _sha256_bytes(path.read_bytes()),
        }

    phase_headings = {
        "tool_decision": "Phase artifact: tool decision",
        "direct_final": "Phase artifact: direct final",
        "post_retrieval_final": "Phase artifact: post retrieval final",
        "mcp_failure_final": "Phase artifact: evidence failure final",
        "emergency_final": "Phase artifact: emergency final",
        "clean_recovery_final": "Phase artifact: clean final recovery",
        "safe_completion_final": "Phase artifact: safe completion final",
    }
    generation_phases: dict[str, str] = {}
    for phase, heading in phase_headings.items():
        generation_phases[phase] = _extract_unique_fence(
            _heading_section(documents["20_GENERATION_PROMPT.md"], heading),
            "text",
            f"Generation {phase} prompt",
        )
    for phase in phase_headings:
        if phase in {"tool_decision", "safe_completion_final"}:
            continue
        generation_phases[phase] = _compile_final_generation(
            generation_phases[phase],
            legal,
            natural_language["final_answer_policy"],
        )

    routing = _extract_unique_fence(
        documents["10_MCP_CATALOG.md"], "text", "MCP routing fragment"
    )
    routing = _compile_legal_routing(routing, legal)

    retrieval = _extract_unique_fence(
        documents["30_RETRIEVAL_PROMPT.md"], "text", "Retrieval prompt"
    )
    retrieval = _replace_required(retrieval, "MCP_ROUTING_PROMPT_FRAGMENT", routing)
    retrieval = _compile_legal_retrieval(retrieval, legal)
    retrieval = _replace_required(retrieval, "MAX_RETRIEVAL_CALLS", str(max_calls))
    retrieval = _insert_after_exact_once(
        retrieval,
        "</input_contract>",
        natural_language["retrieval_policy"],
        "Retrieval natural-language policy",
    )

    for phase, prompt in generation_phases.items():
        remaining_generation = set(_UPPER_TOKEN.findall(prompt))
        if remaining_generation != set(RUNTIME_PLACEHOLDERS):
            raise CompileError(
                f"Generation {phase} runtime placeholder mismatch: "
                f"expected={sorted(RUNTIME_PLACEHOLDERS)}, "
                f"found={sorted(remaining_generation)}"
            )
        if phase != "tool_decision" and any(
            marker in prompt.casefold() for marker in _FORBIDDEN_FINAL_PROMPT_PROTOCOL
        ):
            raise CompileError(f"Generation {phase} prompt contains tool protocol")
        if phase != "tool_decision" and len(prompt) > MAX_FINAL_GENERATION_PROMPT_CHARS:
            raise CompileError(f"Generation {phase} prompt exceeds compact character budget")
    if _UPPER_TOKEN.findall(retrieval) or _UPPER_TOKEN.findall(routing):
        raise CompileError("Retrieval prompt contains unresolved build placeholders")
    for token in BUILD_PLACEHOLDERS:
        marker = f"[{token}]"
        if any(marker in prompt for prompt in generation_phases.values()) or marker in retrieval:
            raise CompileError(f"unresolved build placeholder: {marker}")

    retrieve_tool = _extract_tool(
        documents["20_GENERATION_PROMPT.md"],
        "Generation application tool 입력 계약",
        "retrieve_relevant_content",
    )
    finalize_tool = _extract_tool(
        documents["30_RETRIEVAL_PROMPT.md"],
        "기본 model-facing 계약: Dashboard-v1",
        "finalize_retrieval",
    )
    aliases = _extract_catalog_aliases(documents["10_MCP_CATALOG.md"])
    registered_function_names = {
        "retrieve_relevant_content",
        "finalize_retrieval",
        *aliases,
    }
    for phase, prompt in generation_phases.items():
        if phase == "tool_decision":
            continue
        exposed = sorted(
            name for name in registered_function_names if name.casefold() in prompt.casefold()
        )
        if exposed:
            raise CompileError(
                f"Generation {phase} prompt exposes registered function names: {exposed}"
            )
    missing_in_routing = [alias for alias in aliases if alias not in routing]
    if missing_in_routing:
        raise CompileError(f"MCP aliases missing from routing prompt: {missing_in_routing}")

    compiled_phase_entries: dict[str, dict[str, Any]] = {}
    for phase, prompt in generation_phases.items():
        runtime_counts = {
            token: prompt.count(f"[{token}]") for token in RUNTIME_PLACEHOLDERS
        }
        if any(count < 1 for count in runtime_counts.values()):
            raise CompileError(
                f"every Generation {phase} runtime placeholder must occur at least once"
            )
        compiled_phase_entries[phase] = {
            "runtime_placeholder_counts": runtime_counts,
            "runtime_placeholders": list(RUNTIME_PLACEHOLDERS),
            "sha256": _sha256_text(prompt),
            "template": prompt,
        }
    phase_hashes = [entry["sha256"] for entry in compiled_phase_entries.values()]
    if len(set(phase_hashes)) != len(phase_hashes):
        raise CompileError("Generation phase prompts must be independently distinct")

    binding_seeds = [
        {
            "logical_alias": alias,
            "model_function_name": alias,
            "raw_schema_sha256": None,
            "status": "requires_live_discovery",
            "strict_wrapper_schema_sha256": None,
            "transport_tool_name": None,
        }
        for alias in aliases
    ]

    return {
        "artifact_revision": config["artifact_revision"],
        "dashboard_verified_date": config["dashboard_verified_date"],
        "default_agent_mode": config["default_agent_mode"],
        "feature_flags": {
            "dashboard_v1_finalizer": True,
            "rich_v2": False,
        },
        "local_tools": {
            "finalize_retrieval": {
                "entry": finalize_tool,
                "sha256": _tool_hash(finalize_tool),
            },
            "retrieve_relevant_content": {
                "entry": retrieve_tool,
                "sha256": _tool_hash(retrieve_tool),
            },
        },
        "mcp": {
            "binding_contract": {
                "codex_exposed_prefix_allowed": False,
                "discovery_required": True,
                "drift_action": "quarantine_and_fail_rag_readiness",
                "exact_prompt_registry_equality_required": True,
                "raw_schema_hash_required_for_ready": True,
                "strict_wrapper_and_projection_required_for_ready": True,
            },
            "binding_seeds": binding_seeds,
            "endpoint": config["mcp_endpoint"],
            "logical_aliases": aliases,
            "transport": config["mcp_transport"],
        },
        "model": config["model"],
        "model_endpoint": config["model_endpoint"],
        "prompts": {
            "generation": {
                "phases": compiled_phase_entries,
            },
            "retrieval_dashboard_v1": {
                "max_retrieval_calls": max_calls,
                "sha256": _sha256_text(retrieval),
                "text": retrieval,
            },
        },
        "schema_version": SCHEMA_VERSION,
        "sources": source_manifest,
    }


def _sidecar_path(output: Path) -> Path:
    return output.with_suffix(".sha256")


def _write_or_check(bundle: Mapping[str, Any], output: Path, check: bool) -> bool:
    payload = _canonical_json_bytes(bundle)
    digest = _sha256_bytes(payload)
    sidecar = _sidecar_path(output)
    expected_sidecar = f"{digest}  {output.name}\n".encode("ascii")
    if check:
        try:
            current_payload = output.read_bytes()
            current_sidecar = sidecar.read_bytes()
        except OSError:
            return False
        return current_payload == payload and current_sidecar == expected_sidecar
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    sidecar.write_bytes(expected_sidecar)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--runtime-sources", type=Path, default=DEFAULT_RUNTIME_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero unless committed artifact bytes exactly match a fresh compile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = compile_bundle(args.source_root.resolve(), args.runtime_sources.resolve())
        matched = _write_or_check(bundle, args.output.resolve(), args.check)
    except CompileError as error:
        print(f"artifact compile failed: {error}", file=sys.stderr)
        return 2
    if args.check and not matched:
        print("runtime artifact is stale; regenerate it", file=sys.stderr)
        return 1
    if not args.check:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
