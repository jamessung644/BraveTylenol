import pytest

from lunit_hackathon.fast_policy import choose_generation_profile
from lunit_hackathon.schemas import ChatMessage


def test_unresponsive_patient_uses_emergency_profile():
    profile = choose_generation_profile(
        [
            ChatMessage(
                role="user",
                content="My neighbor is unresponsive and breathing slowly. What should I do?",
            )
        ]
    )

    assert profile.kind == "emergency"
    assert profile.max_tokens == 1280
    assert "urgent action first" in profile.system_prompt
    assert "at most 180 words" in profile.system_prompt


def test_short_routine_question_uses_smallest_profile():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="감기 때 물을 많이 마시면 도움이 되나요?")]
    )

    assert profile.kind == "routine"
    assert profile.max_tokens == 768
    assert "If an emergency is plausible" in profile.system_prompt
    assert "at most 120 words" in profile.system_prompt


def test_long_clinical_context_keeps_room_for_a_complete_answer():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="검사 결과를 해석해 주세요. " + "상세 병력 " * 100)]
    )

    assert profile.kind == "complex"
    assert profile.max_tokens == 1280


def test_korean_emergency_language_is_detected():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="갑자기 의식을 잃고 숨을 제대로 못 쉬어요")]
    )

    assert profile.kind == "emergency"


@pytest.mark.parametrize(
    "text",
    [
        "Sudden chest pressure is spreading into my left arm.",
        "One side of their face is drooping and their speech is slurred.",
        "I am having trouble breathing.",
        "I took a whole bottle of pills.",
        "가슴이 짓눌리듯 아프고 왼팔로 퍼져요.",
        "얼굴 한쪽이 처지고 말이 어눌해졌어요.",
        "숨쉬기가 너무 힘들어요.",
        "약 한 병을 전부 먹었어요.",
    ],
)
def test_common_emergency_paraphrases_get_the_larger_budget(text):
    profile = choose_generation_profile([ChatMessage(role="user", content=text)])

    assert profile.kind == "emergency"
    assert profile.max_tokens == 1280
