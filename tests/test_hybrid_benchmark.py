import json

from scripts.benchmark_hybrid_local import (
    COEVAL_CONTRACT,
    COEVAL_PUBLIC_REQUEST_VECTORS,
    ROUTE_VECTORS,
    main,
    run_benchmark,
)


def test_official_public_coeval_contract_is_pinned_without_held_out_tuning():
    assert COEVAL_CONTRACT == {
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


def test_fixed_sixteen_way_matrix_matches_router_policy():
    assert len(ROUTE_VECTORS) == 16
    assert len({vector.vector_id for vector in ROUTE_VECTORS}) == 16

    report = run_benchmark()

    assert report["summary"]["route_gate_passed"] is True
    assert report["summary"]["route_mismatches"] == []
    assert report["summary"]["route_counts"] == {
        "direct": 7,
        "emergency": 1,
        "rag": 8,
    }
    assert report["summary"]["route_share"] == {
        "direct_or_emergency": 0.5,
        "rag": 0.5,
    }


def test_domain_specific_fake_mcp_call_profiles_are_fixed_before_measurement():
    rows = {row["id"]: row for row in run_benchmark()["vectors"]}
    for vector in ROUTE_VECTORS:
        if vector.expected_route != "rag":
            assert vector.mcp_profile == "none"
            assert vector.rag_mcp_calls == 0
            assert rows[vector.vector_id]["simulated_l2_calls"] == 1
        elif vector.mcp_profile == "structured":
            assert vector.rag_mcp_calls == 1
            assert rows[vector.vector_id]["simulated_l2_calls"] == 4
        else:
            assert vector.mcp_profile == "hierarchical"
            assert vector.rag_mcp_calls in {2, 3}
            assert rows[vector.vector_id]["simulated_l2_calls"] == (
                vector.rag_mcp_calls + 3
            )


def test_known_high_risk_router_regressions_are_explicit_gate_vectors():
    report = run_benchmark()
    rows = {row["id"]: row for row in report["vectors"]}

    assert rows["warfarin_pregnancy"]["actual_route"] == "rag"
    assert rows["stale_authority_then_thanks"]["actual_route"] == "direct"
    assert rows["assistant_emergency_text_then_source_request"]["actual_route"] == "rag"
    assert rows["negated_historical_emergency"]["actual_route"] == "direct"
    assert rows["coreference_drug_three_turns_back"]["actual_route"] == "rag"


def test_virtual_latency_call_and_citation_accounting_is_deterministic():
    first = run_benchmark()
    second = run_benchmark()
    assert first == second

    summary = first["summary"]
    assert summary["simulated_latency"] == {
        "direct_or_emergency": {
            "count": 8,
            "p50_ms": 3100.0,
            "p95_ms": 3100.0,
            "max_ms": 3100,
        },
        "rag": {
            "count": 8,
            "p50_ms": 13100.0,
            "p95_ms": 19700.0,
            "max_ms": 21100,
        },
        "all": {
            "count": 16,
            "p50_ms": 8100.0,
            "p95_ms": 18100.0,
            "max_ms": 21100,
        },
    }
    assert summary["simulated_calls"] == {
        "l2_total": 44,
        "l2_per_request": 2.75,
        "mcp_total": 12,
        "mcp_per_request": 0.75,
        "mcp_per_rag_request": 1.5,
    }
    assert summary["simulated_batch"] == {
        "concurrency_shape": 16,
        "wall_ms": 21100,
        "model": "all fixtures start at t=0 with no resource contention",
    }
    assert summary["fake_fixture_cite_yield"] == {
        "rag_requests_with_observed_cite_uid": 7,
        "rag_requests": 8,
        "rate": 0.875,
        "rule": "success only when fake MCP result contains an observed cite_uid",
    }


def test_report_marks_every_nonlocal_result_as_unmeasured():
    report = run_benchmark()

    assert report["measurement_kind"] == "secret_free_deterministic_simulation"
    assert report["network_calls"] == 0
    assert report["execution_status"] == {
        "local_fake_gate": "executed",
        "official_coeval_score_ab": "not_measured",
        "live_l2_mcp": "not_measured",
        "official_dashboard": "not_measured",
    }
    assert report["virtual_cost_model"]["warning"] == (
        "fixture accounting only; not observed latency"
    )
    assert report["virtual_cost_model"]["scope"] == (
        "successful-path minimum; retries and recovery excluded"
    )


def test_public_contract_vectors_are_schema_only_and_preserve_text_parts():
    assert [vector["id"] for vector in COEVAL_PUBLIC_REQUEST_VECTORS] == [
        "single_turn",
        "multi_turn",
        "text_content_parts",
    ]

    checks = run_benchmark()["coeval_public_contract_checks"]
    assert checks["scope"] == "local request-schema parsing only"
    assert checks["http_response_nonempty_or_explicit_error"] == "not_executed"
    assert all(check["request_schema_accepted"] for check in checks["checks"])
    assert all(check["latest_user_preserved"] for check in checks["checks"])


def test_cli_emits_machine_readable_report(capsys):
    assert main([]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "hybrid-local-gate-v1"
    assert report["summary"]["route_gate_passed"] is True
