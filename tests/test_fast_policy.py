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
    assert profile.max_tokens == 1536
    assert "urgent action first" in profile.system_prompt


def test_short_routine_question_uses_smallest_profile():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="감기 때 물을 많이 마시면 도움이 되나요?")]
    )

    assert profile.kind == "routine"
    assert profile.max_tokens == 1024


def test_long_clinical_context_keeps_room_for_a_complete_answer():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="검사 결과를 해석해 주세요. " + "상세 병력 " * 100)]
    )

    assert profile.kind == "complex"
    assert profile.max_tokens == 1536


def test_korean_emergency_language_is_detected():
    profile = choose_generation_profile(
        [ChatMessage(role="user", content="갑자기 의식을 잃고 숨을 제대로 못 쉬어요")]
    )

    assert profile.kind == "emergency"

