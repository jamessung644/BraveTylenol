import json

import pytest

from lunit_hackathon.artifacts import (
    DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE,
    RETRIEVAL_SYSTEM_PROMPT,
    load_runtime_artifacts,
)
from lunit_hackathon.generation import GenerationEngine
from lunit_hackathon.schemas import ChatMessage, L2Completion


class RecordingL2:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return L2Completion(content="안전한 범위에서 답변")


class UnusedRetrieval:
    async def retrieve(self, query):  # pragma: no cover - must not be reached
        raise AssertionError(f"unexpected retrieval: {query}")


def _generation_envelope(call: dict) -> dict:
    for message in call["messages"]:
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if payload.get("schema_version") == "generation-input-v1":
            return payload
    raise AssertionError("generation-input-v1 envelope was not sent to L2")


def test_generation_prompt_accepts_noisy_korean_without_clinical_autocorrection():
    prompt = DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE

    assert "<natural_language_final_contract>" in prompt
    assert "단어·증상·수치 나열" in prompt
    assert "문장 도치" in prompt
    assert "한국어·영어 혼용" in prompt
    assert "원문을 보존" in prompt
    assert "약물·성분·제품" in prompt
    assert "조용히 한 후보로 확정하지 않는다" in prompt
    assert "즉시 위험 신호가 있으면" in prompt
    assert "가장 중요한 1~3개만 짧게 확인" in prompt
    assert "문법을 평가" in prompt


def test_retrieval_prompt_separates_query_normalization_from_clinical_truth():
    prompt = RETRIEVAL_SYSTEM_PROMPT

    assert "<natural_language_query_contract>" in prompt
    assert "검증된 임상 사실이나 진단이 아니다" in prompt
    assert "원래 표현을 유지한 채 보수적으로 확장" in prompt
    assert "하나로 자동 교정하지 말고" in prompt
    assert "삭제·반전·단일값화" in prompt
    assert "가까이 있다는 이유만으로 약과 용량" in prompt
    assert "각 후보를 별도 검색 가설" in prompt
    assert "status=`partial`" in prompt


@pytest.mark.parametrize(
    "user_text",
    [
        "타이레놀 두알 먹엇는데 괜찬?",
        "머리아픔 3일 구토2번 뭐해야",
        "먹어도 되나요 임신 8주인데 이 약을",
        "BP 90? 어지럼 지금 약 암로디핀",
        "아까 말한건 엄마고 나는 안 아픔",
    ],
)
async def test_noisy_input_is_preserved_exactly_in_generation_envelope(user_text):
    l2 = RecordingL2()

    answer = await GenerationEngine(l2, UnusedRetrieval()).answer(
        [ChatMessage(role="user", content=user_text)]
    )

    assert answer == "안전한 범위에서 답변"
    envelope = _generation_envelope(l2.calls[0])
    assert envelope["latest_user_message"]["content"] == user_text
    assert envelope["application_context"]["normalization_status"] == (
        "degraded_raw_only"
    )


@pytest.mark.parametrize(
    "fragment",
    [
        "숨... 못쉼 지금",
        "버튼 전지 삼킴 아이",
        "약 많이먹음 방금",
        "chest-pain severe now",
    ],
)
async def test_obvious_noisy_emergency_fragments_skip_retrieval(fragment):
    l2 = RecordingL2()

    await GenerationEngine(l2, UnusedRetrieval()).answer(
        [ChatMessage(role="user", content=fragment)]
    )

    assert "tools" not in l2.calls[0]
    system_prompt = l2.calls[0]["messages"][0]["content"]
    assert "시간 민감한 건강 위험" in system_prompt
    assert "retrieve_relevant_content" not in system_prompt
    envelope = _generation_envelope(l2.calls[0])
    assert envelope["latest_user_message"]["content"] == fragment


def test_artifact_manifest_hashes_natural_language_policy_build_input():
    sources = load_runtime_artifacts().manifest_copy()["sources"]

    policy = sources["runtime_sources/natural_language_policy_ko_v1.json"]
    assert policy["path"] == "runtime_sources/natural_language_policy_ko_v1.json"
    assert len(policy["sha256"]) == 64
