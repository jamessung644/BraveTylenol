import asyncio
import hashlib
import json
from collections import Counter

import httpx
import pytest

from scripts.synthetic_medical_gate import (
    DEFAULT_FIXTURE,
    MAX_CONCURRENCY,
    _check_json_object,
    _check_labeled_bullets,
    _validate_loopback_base_url,
    load_fixture,
    run_gate,
)

CATEGORY_CONTRACT = {
    "emergency": 5,
    "medication": 4,
    "routine_clinical_reasoning": 3,
    "uncertainty_clarification": 3,
    "multi_turn_context": 3,
    "structured_output": 2,
}
EXPECTED_IDS = [f"SMG-{number:03d}" for number in range(1, 21)]
# This locks the observed generation-1 artifact. Extensions go in a new v2
# fixture; changing this checksum to make an old observation pass is forbidden.
FROZEN_GENERATION_1_SHA256 = "4d515605f642e36005cc75660dafa304763d4e12a54dc106a1f5852a4e46b00d"


def test_generation_one_is_unique_complete_and_frozen():
    fixture = load_fixture()
    cases = fixture["cases"]

    assert fixture["generation"] == 1
    assert fixture["generation_status"] == "frozen_before_observation"
    assert fixture["append_only_policy"] == {
        "case_ids_are_permanent": True,
        "observed_case_semantics_must_not_change": True,
        "observed_category_assignments_must_not_change": True,
        "new_failures_create_new_case_ids": True,
        "extensions_require_a_new_frozen_generation": True,
    }
    assert [case["id"] for case in cases] == EXPECTED_IDS
    assert len({case["id"] for case in cases}) == 20
    assert len(
        {
            tuple(message["content"] for message in case["messages"] if message["role"] == "user")
            for case in cases
        }
    ) == 20
    assert Counter(case["primary_category"] for case in cases) == CATEGORY_CONTRACT
    assert fixture["category_contract"] == CATEGORY_CONTRACT
    assert Counter(case["expected_route"] for case in cases) == {
        "direct": 10,
        "emergency": 5,
        "rag": 5,
    }
    assert hashlib.sha256(DEFAULT_FIXTURE.read_bytes()).hexdigest() == (
        FROZEN_GENERATION_1_SHA256
    )


def test_fixture_uses_concept_categories_and_has_no_dataset_references():
    fixture = load_fixture()
    serialized = json.dumps(fixture, ensure_ascii=False).casefold()

    for prohibited_reference in ("healthbench", "coeval", "conquer", "held-out", "rubric"):
        assert prohibited_reference not in serialized
    for prohibited_answer_field in ("expected_answer", "reference_answer", "answer_text"):
        assert prohibited_answer_field not in serialized

    dimensions = {dimension for case in fixture["cases"] for dimension in case["dimensions"]}
    assert dimensions == {
        "accuracy",
        "completeness",
        "context",
        "communication",
        "instruction_adherence",
    }
    for case in fixture["cases"]:
        invariants = case["invariants"]
        assert invariants["required_concepts"]
        assert invariants["forbidden_concepts"]
        assert set(invariants["required_concepts"]).isdisjoint(
            invariants["forbidden_concepts"]
        )


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_runner_accepts_only_loopback_targets(url, normalized):
    assert _validate_loopback_base_url(url) == normalized


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://container.internal:8000",
        "http://127.0.0.1:8000/v1",
        "http://user:secret@127.0.0.1:8000",
    ],
)
def test_runner_rejects_external_path_or_credential_bearing_targets(url):
    with pytest.raises(ValueError):
        _validate_loopback_base_url(url)


async def test_report_redacts_request_response_and_secret_and_caps_concurrency():
    active = 0
    maximum_active = 0
    seen_requests = []
    seen_authorization = []
    secret = "SECRET_API_KEY_SENTINEL"
    response_sentinel = "PRIVATE_RESPONSE_BODY_SENTINEL"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            seen_authorization.append(request.headers.get("authorization"))
            seen_requests.append(json.loads(await request.aread()))
            await asyncio.sleep(0.005)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-synthetic",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "team-chatbot",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_sentinel},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        finally:
            active -= 1

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        transport=httpx.MockTransport(handler),
    ) as client:
        report = await run_gate(
            "http://127.0.0.1:8000",
            concurrency=MAX_CONCURRENCY,
            api_key=secret,
            client=client,
        )

    serialized = json.dumps(report, ensure_ascii=False)
    fixture = load_fixture()
    assert len(seen_requests) == len(fixture["cases"]) == 20
    assert maximum_active == MAX_CONCURRENCY == 8
    assert seen_authorization == [f"Bearer {secret}"] * 20
    assert secret not in serialized
    assert response_sentinel not in serialized
    assert fixture["cases"][0]["messages"][0]["content"] not in serialized
    assert report["retention"] == {
        "request_messages_in_report": False,
        "response_bodies_in_report": False,
        "credentials_in_report": False,
    }
    assert report["execution"]["actual_route_collection"] == (
        "not_collected_use_sanitized_container_log"
    )
    assert all(case["http_status"] == 200 for case in report["cases"])
    assert all(case["content_length"] == len(response_sentinel) for case in report["cases"])
    assert all("content" not in case for case in report["cases"])


def test_structured_format_checks_are_deterministic():
    json_spec = {"kind": "json_object", "exact_keys": ["a", "b"]}
    assert _check_json_object('{"a":"one","b":["two"]}', json_spec)
    assert not _check_json_object('```json\n{"a":"one","b":"two"}\n```', json_spec)
    assert not _check_json_object('{"a":"one","b":"two","c":"three"}', json_spec)

    bullet_spec = {"kind": "labeled_bullets", "labels": ["First", "Second"]}
    assert _check_labeled_bullets("- First: one\n- Second: two", bullet_spec)
    assert not _check_labeled_bullets("Heading\n- First: one\n- Second: two", bullet_spec)
    assert not _check_labeled_bullets("- Second: two\n- First: one", bullet_spec)


@pytest.mark.parametrize("concurrency", [0, 9])
async def test_runner_rejects_concurrency_outside_one_through_eight(concurrency):
    with pytest.raises(ValueError):
        await run_gate("http://127.0.0.1:8000", concurrency=concurrency)
