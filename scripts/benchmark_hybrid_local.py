#!/usr/bin/env python3
"""Run the secret-free deterministic routing and cost-accounting gate.

This benchmark deliberately does not call L2, MCP, a judge, or the network.  Its
latencies are virtual-clock fixture values used to make route/call regressions
comparable across local runs; they are not runtime performance measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Route = Literal["direct", "emergency", "rag"]
MCPProfile = Literal["none", "structured", "hierarchical"]

COEVAL_CONTRACT = {
    "repository": "https://github.com/lunit-io/CoEval",
    "pinned_commit": "741263cfafba687f8baeb7422c747ef9557df1c4",
    "suite": "conquer_val",
    "public_examples": 301,
    "judge_calls_reported_by_pinned_config": 3_337,
    "judge_calls_reported_by_published_ids_manifest": 3_410,
    "cost_basis_judge_calls_per_arm": 3_410,
    "concurrency": 16,
    "inference_attempts": 2,
    "inference_delay_seconds": 2,
    "max_tokens": 6_144,
    "timeout_seconds": 180,
    "inference_failure_score": 0,
    "held_out_policy": "do_not_access_or_tune_conquer_test_ids_or_judge",
    "public_split_policy": {
        "generation": "coeval-741263-dev-v2",
        "dev_count": 120,
        "prospective_score_holdout_count": 181,
        "known_prior_judge_touched_forced_into_dev": 3,
        "strict_prompt_holdout": False,
        "prior_full_val_inference_artifact_known": True,
        "split_generator": "scripts/prepare_conquer_val_split.py",
        "runtime_item_specific_rules_forbidden": True,
    },
}

COEVAL_PUBLIC_REQUEST_VECTORS = (
    {
        "id": "single_turn",
        "payload": {
            "model": "team-chatbot",
            "messages": [{"role": "user", "content": "What should I monitor?"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 6_144,
            "stream": False,
        },
        "expected_latest_user": "What should I monitor?",
    },
    {
        "id": "multi_turn",
        "payload": {
            "model": "team-chatbot",
            "messages": [
                {"role": "system", "content": "You are a careful medical assistant."},
                {"role": "user", "content": "I have had a mild fever for two days."},
                {
                    "role": "assistant",
                    "content": "Do you have warning signs or relevant conditions?",
                },
                {
                    "role": "user",
                    "content": "No breathing difficulty or confusion. What should I monitor?",
                },
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 6_144,
            "stream": False,
        },
        "expected_latest_user": (
            "No breathing difficulty or confusion. What should I monitor?"
        ),
    },
    {
        "id": "text_content_parts",
        "payload": {
            "model": "team-chatbot",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "혈압이 "},
                        {"type": "input_text", "text": "높을 때 무엇을 확인할까?"},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 6_144,
            "stream": False,
        },
        "expected_latest_user": "혈압이 높을 때 무엇을 확인할까?",
    },
)

# Deterministic fake-clock costs.  They intentionally model accounting only.
_VIRTUAL_L2_CALL_MS = 3_000
_VIRTUAL_MCP_CALL_MS = 1_000
_VIRTUAL_REQUEST_OVERHEAD_MS = 100


@dataclass(frozen=True, slots=True)
class RouteVector:
    vector_id: str
    cohort: str
    messages: tuple[tuple[str, str], ...]
    expected_route: Route
    mcp_profile: MCPProfile = "none"
    rag_mcp_calls: int = 0
    fake_observed_cite_uids: tuple[str, ...] = ()


ROUTE_VECTORS: tuple[RouteVector, ...] = (
    RouteVector(
        "general_single_turn",
        "general",
        (("user", "가벼운 긴장성 두통에 집에서 해볼 수 있는 방법은?"),),
        "direct",
    ),
    RouteVector(
        "noisy_fragment",
        "noisy_korean",
        (("user", "머리아픔 어제부터 열없음 물 마심"),),
        "direct",
    ),
    RouteVector(
        "inverted_low_risk_question",
        "noisy_korean",
        (("user", "먹어도 될까, 감기 때 죽은?"),),
        "direct",
    ),
    RouteVector(
        "source_word_not_a_request",
        "router_precision",
        (("user", "감기 출처가 집인지 회사인지 모르겠어요"),),
        "direct",
    ),
    RouteVector(
        "stale_authority_then_thanks",
        "multi_turn",
        (
            ("user", "식약처 아세트아미노펜 허가사항을 알려줘"),
            ("assistant", "이전 답변입니다."),
            ("user", "고마워"),
        ),
        "direct",
    ),
    RouteVector(
        "source_topic_then_topic_switch",
        "multi_turn",
        (
            ("user", "고혈압 최신 진료지침 출처를 찾아줘"),
            ("assistant", "이전 답변입니다."),
            ("user", "그건 됐고 오늘 할 가벼운 스트레칭을 알려줘"),
        ),
        "direct",
    ),
    RouteVector(
        "current_emergency",
        "emergency",
        (("user", "지금 가슴 통증이 심하고 숨을 못 쉬겠어"),),
        "emergency",
    ),
    RouteVector(
        "negated_historical_emergency",
        "emergency_precision",
        (("user", "어제는 가슴 통증이 있었지만 지금은 통증도 호흡곤란도 없어"),),
        "direct",
    ),
    RouteVector(
        "assistant_emergency_text_then_source_request",
        "multi_turn",
        (
            ("user", "고혈압 기준을 설명해줘"),
            ("assistant", "가슴 통증이나 숨을 못 쉬면 즉시 진료를 받으세요."),
            ("user", "그 기준의 공식 출처를 알려줘"),
        ),
        "rag",
        "hierarchical",
        2,
        ("fixture:guideline:source",),
    ),
    RouteVector(
        "warfarin_pregnancy",
        "drug_safety",
        (("user", "와파린 임신 중 먹어도 돼?"),),
        "rag",
        "structured",
        1,
        ("fixture:mfds:warfarin",),
    ),
    RouteVector(
        "mfds_label_lookup",
        "structured_lookup",
        (("user", "식약처 타이레놀 임신 중 허가사항 알려줘"),),
        "rag",
        "structured",
        1,
        ("fixture:mfds:label",),
    ),
    RouteVector(
        "kcd_code_lookup",
        "structured_lookup",
        (("user", "KCD I10의 공식 명칭은?"),),
        "rag",
        "structured",
        1,
        ("fixture:kcd:i10",),
    ),
    RouteVector(
        "hira_benefit_lookup",
        "structured_lookup",
        (("user", "심평원 기준 와파린 보험급여를 확인해줘"),),
        "rag",
        "structured",
        1,
        ("fixture:hira:warfarin",),
    ),
    RouteVector(
        "guideline_hierarchy",
        "hierarchical_lookup",
        (("user", "한국 고혈압 최신 진료지침의 목표 혈압과 출처를 알려줘"),),
        "rag",
        "hierarchical",
        3,
        ("fixture:guideline:hypertension",),
    ),
    RouteVector(
        "research_source_no_match",
        "hierarchical_lookup",
        (("user", "생강이 감기에 효과 있는지 공식 근거를 찾아줘"),),
        "rag",
        "hierarchical",
        2,
    ),
    RouteVector(
        "coreference_drug_three_turns_back",
        "multi_turn",
        (
            ("user", "와파린을 복용 중이야"),
            ("assistant", "확인했습니다."),
            ("user", "복용 시간은 저녁이야"),
            ("assistant", "확인했습니다."),
            ("user", "그리고 임신을 준비하고 있어"),
            ("assistant", "확인했습니다."),
            ("user", "그 약을 임신 중에도 계속 먹어도 돼?"),
        ),
        "rag",
        "structured",
        1,
        ("fixture:mfds:warfarin-coreference",),
    ),
)


def _classify_route(vector: RouteVector) -> Route:
    # Lazy imports keep direct script execution independent of PYTHONPATH setup.
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from lunit_hackathon.generation import _is_emergency_turn, requires_retrieval
    from lunit_hackathon.schemas import ChatMessage

    messages = [ChatMessage(role=role, content=content) for role, content in vector.messages]
    if _is_emergency_turn(messages):
        return "emergency"
    if requires_retrieval(messages):
        return "rag"
    return "direct"


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 3)


def _latency_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0),
    }


def _local_public_contract_checks() -> dict[str, Any]:
    from lunit_hackathon.schemas import ChatCompletionRequest

    checks: list[dict[str, Any]] = []
    for vector in COEVAL_PUBLIC_REQUEST_VECTORS:
        request = ChatCompletionRequest.model_validate(vector["payload"])
        latest_user = next(
            message.content
            for message in reversed(request.messages)
            if message.role == "user"
        )
        checks.append(
            {
                "id": vector["id"],
                "request_schema_accepted": True,
                "message_count": len(request.messages),
                "latest_user_preserved": latest_user == vector["expected_latest_user"],
            }
        )
    return {
        "scope": "local request-schema parsing only",
        "checks": checks,
        "http_response_nonempty_or_explicit_error": "not_executed",
    }


def run_benchmark() -> dict[str, Any]:
    """Return a deterministic local routing and virtual-cost report."""

    rows: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    direct_latencies: list[int] = []
    rag_latencies: list[int] = []
    all_latencies: list[int] = []
    l2_total = 0
    mcp_total = 0
    rag_with_fake_observed_cite_uid = 0

    for vector in ROUTE_VECTORS:
        actual_route = _classify_route(vector)
        route_counts[actual_route] += 1
        mcp_calls = vector.rag_mcp_calls if actual_route == "rag" else 0
        # Successful RAG minimum: one Generation decision, one Retrieval planner
        # turn per remote MCP call, one local-finalizer planner turn, and one final
        # Generation answer. Forced retries and answer recovery are intentionally
        # excluded and must only increase this lower bound.
        l2_calls = mcp_calls + 3 if actual_route == "rag" else 1
        latency_ms = (
            _VIRTUAL_REQUEST_OVERHEAD_MS
            + l2_calls * _VIRTUAL_L2_CALL_MS
            + mcp_calls * _VIRTUAL_MCP_CALL_MS
        )
        fake_cite_uids = (
            list(vector.fake_observed_cite_uids) if actual_route == "rag" else []
        )
        if fake_cite_uids:
            rag_with_fake_observed_cite_uid += 1
        l2_total += l2_calls
        mcp_total += mcp_calls
        all_latencies.append(latency_ms)
        if actual_route == "rag":
            rag_latencies.append(latency_ms)
        else:
            direct_latencies.append(latency_ms)
        rows.append(
            {
                "id": vector.vector_id,
                "cohort": vector.cohort,
                "expected_route": vector.expected_route,
                "actual_route": actual_route,
                "route_match": actual_route == vector.expected_route,
                "mcp_profile": vector.mcp_profile,
                "simulated_l2_calls": l2_calls,
                "simulated_mcp_calls": mcp_calls,
                "simulated_latency_ms": latency_ms,
                "fake_observed_cite_uids": fake_cite_uids,
            }
        )

    request_count = len(rows)
    rag_count = route_counts["rag"]
    mismatches = [row["id"] for row in rows if not row["route_match"]]
    return {
        "schema_version": "hybrid-local-gate-v1",
        "measurement_kind": "secret_free_deterministic_simulation",
        "network_calls": 0,
        "coeval_contract": COEVAL_CONTRACT,
        "coeval_public_contract_checks": _local_public_contract_checks(),
        "execution_status": {
            "local_fake_gate": "executed",
            "official_coeval_score_ab": "not_measured",
            "live_l2_mcp": "not_measured",
            "official_dashboard": "not_measured",
        },
        "virtual_cost_model": {
            "l2_call_ms": _VIRTUAL_L2_CALL_MS,
            "mcp_call_ms": _VIRTUAL_MCP_CALL_MS,
            "request_overhead_ms": _VIRTUAL_REQUEST_OVERHEAD_MS,
            "scope": "successful-path minimum; retries and recovery excluded",
            "warning": "fixture accounting only; not observed latency",
        },
        "summary": {
            "request_count": request_count,
            "route_gate_passed": not mismatches,
            "route_mismatches": mismatches,
            "route_counts": {
                "direct": route_counts["direct"],
                "emergency": route_counts["emergency"],
                "rag": rag_count,
            },
            "route_share": {
                "direct_or_emergency": round(
                    (route_counts["direct"] + route_counts["emergency"])
                    / request_count,
                    4,
                ),
                "rag": round(rag_count / request_count, 4),
            },
            "simulated_latency": {
                "direct_or_emergency": _latency_summary(direct_latencies),
                "rag": _latency_summary(rag_latencies),
                "all": _latency_summary(all_latencies),
            },
            "simulated_batch": {
                "concurrency_shape": request_count,
                "wall_ms": max(all_latencies, default=0),
                "model": "all fixtures start at t=0 with no resource contention",
            },
            "simulated_calls": {
                "l2_total": l2_total,
                "l2_per_request": round(l2_total / request_count, 4),
                "mcp_total": mcp_total,
                "mcp_per_request": round(mcp_total / request_count, 4),
                "mcp_per_rag_request": round(mcp_total / rag_count, 4)
                if rag_count
                else 0.0,
            },
            "fake_fixture_cite_yield": {
                "rag_requests_with_observed_cite_uid": rag_with_fake_observed_cite_uid,
                "rag_requests": rag_count,
                "rate": round(rag_with_fake_observed_cite_uid / rag_count, 4)
                if rag_count
                else 0.0,
                "rule": "success only when fake MCP result contains an observed cite_uid",
            },
        },
        "vectors": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="indent the JSON report")
    parser.add_argument(
        "--allow-route-mismatch",
        action="store_true",
        help="return success even when a route fixture mismatches",
    )
    args = parser.parse_args(argv)
    report = run_benchmark()
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    if report["summary"]["route_gate_passed"] or args.allow_route_mismatch:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
