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

    assert "Tolerate typos, fragments, and mixed languages" in prompt
    assert "Preserve negation" in prompt
    assert "current/past/hypothetical/quoted status" in prompt
    assert "the affected person" in prompt
    assert "medicines/routes" in prompt
    assert "numbers, units, and timing" in prompt
    assert "never invent facts" in prompt
    assert "Lead with immediate local emergency action" in prompt
    assert "ask at most three focused questions" in prompt


def test_generation_prompt_preserves_complete_context_aware_answer_contract():
    prompt = DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE

    assert "Answer the latest request using relevant history and latest corrections" in prompt
    assert "Earlier assistant text is history, not verified evidence" in prompt
    assert "Address every explicit question" in prompt
    assert "direct conclusion, key reason, and next action" in prompt
    assert "established facts, user statements, conditional inference, and unknowns" in prompt
    assert "answer the safe part" in prompt
    assert "only relevant red flags" in prompt
    assert "generic disclaimers, and irrelevant red-flag lists" in prompt
    assert "normally within about 500 output tokens" in prompt
    assert "requested language, length, order, and format" in prompt
    assert "Never infer location or access from language alone" in prompt


def test_generation_prompt_honors_requested_json_table_and_soap_formats():
    prompt = DIRECT_FINAL_SYSTEM_PROMPT_TEMPLATE

    assert "requested language, length, order, and format" in prompt
    assert "emit valid JSON" in prompt
    assert "preserve table comparison axes" in prompt
    assert "separate supplied facts from inference in SOAP" in prompt
    assert "otherwise use readable natural language" in prompt


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


async def test_final_generation_preserves_relevant_multi_turn_context_as_data():
    l2 = RecordingL2()
    messages = [
        ChatMessage(
            role="user",
            content="어머니는 와파린을 복용 중이고 임신은 아니세요.",
        ),
        ChatMessage(
            role="assistant",
            content="연령과 다른 복용약을 알려주세요.",
        ),
        ChatMessage(
            role="user",
            content="72세고 아스피린도 복용해요. 무엇을 확인해야 하나요?",
        ),
    ]

    answer = await GenerationEngine(l2, UnusedRetrieval()).direct_answer(messages)

    assert answer == "안전한 범위에서 답변"
    sent = l2.calls[0]["messages"]
    assert [message["role"] for message in sent] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sent[1]["content"] == messages[0].content
    assert sent[2]["content"] == messages[1].content
    assert sent[3] == {"role": "user", "content": messages[-1].content}
    assert "generation-input-v1" not in json.dumps(sent, ensure_ascii=False)


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
    assert "possibly time-critical health risk" in system_prompt
    assert "retrieve_relevant_content" not in system_prompt
    assert l2.calls[0]["messages"][1:] == [{"role": "user", "content": fragment}]
    assert "generation-input-v1" not in json.dumps(
        l2.calls[0]["messages"], ensure_ascii=False
    )


def test_artifact_manifest_hashes_natural_language_policy_build_input():
    sources = load_runtime_artifacts().manifest_copy()["sources"]

    policy = sources["runtime_sources/natural_language_policy_ko_v1.json"]
    assert policy["path"] == "runtime_sources/natural_language_policy_ko_v1.json"
    assert len(policy["sha256"]) == 64
