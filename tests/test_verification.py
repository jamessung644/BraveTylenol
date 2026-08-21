import asyncio

import pytest

from lunit_hackathon.config import Settings
from lunit_hackathon.deadline import RequestDeadline
from lunit_hackathon.errors import RetrievalError, UpstreamTimeoutError
from lunit_hackathon.schemas import ChatMessage, L2Completion, RetrievalResult
from lunit_hackathon.verification import AnswerVerifier


class ScriptedL2:
    def __init__(self, completion=None, error=None):
        self.completion = completion
        self.error = error
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.completion


def settings() -> Settings:
    return Settings(_env_file=None)


async def test_verifier_returns_repaired_answer_for_high_risk_drug_claim():
    l2 = ScriptedL2(L2Completion(content="수정된 근거 기반 답변"))
    candidate = "근거 없는 2배 복용 권고"

    answer = await AnswerVerifier(l2, settings()).verify(
        [ChatMessage(role="user", content="이 약을 두 배 먹어도 되나요?")],
        candidate,
        RetrievalResult(status="partial", note="근거 부족"),
        RequestDeadline.start(),
    )

    assert answer == "수정된 근거 기반 답변"
    assert l2.calls[0]["max_tokens"] <= 2_048
    assert l2.calls[0]["reasoning_effort"] == "medium"
    assert l2.calls[0]["timeout_seconds"] <= 25
    assert "BEGIN UNTRUSTED CANDIDATE" in l2.calls[0]["messages"][-2]["content"]


async def test_verifier_skips_when_fewer_than_minimum_seconds_remain():
    l2 = ScriptedL2(L2Completion(content="수정본"))
    candidate = "복용량을 임의로 늘리지 마세요."
    clock = iter([0.0, 140.1])
    deadline = RequestDeadline.start(total_seconds=165.0, clock=lambda: next(clock))

    assert (
        await AnswerVerifier(l2, settings()).verify(
            [ChatMessage(role="user", content="용량을 늘려도 되나요?")],
            candidate,
            RetrievalResult(status="partial", note="근거 부족"),
            deadline,
        )
        == candidate
    )
    assert l2.calls == []


async def test_verifier_leaves_two_second_reserve_and_returns_candidate_on_expected_errors():
    l2 = ScriptedL2(error=UpstreamTimeoutError("timed out"))
    candidate = "후보 답변"
    clock = iter([0.0, 130.0, 130.0])
    deadline = RequestDeadline.start(total_seconds=165.0, clock=lambda: next(clock))

    assert (
        await AnswerVerifier(l2, settings()).verify(
            [ChatMessage(role="user", content="질문")],
            candidate,
            RetrievalResult(status="no_evidence"),
            deadline,
        )
        == candidate
    )
    assert l2.calls[0]["timeout_seconds"] == 25.0


async def test_verifier_timeout_leaves_two_second_reserve_below_twenty_five_seconds():
    l2 = ScriptedL2(error=UpstreamTimeoutError("timed out"))
    clock = iter([0.0, 139.0, 139.0])
    deadline = RequestDeadline.start(total_seconds=165.0, clock=lambda: next(clock))

    assert (
        await AnswerVerifier(l2, settings()).verify(
            [ChatMessage(role="user", content="질문")],
            "후보 답변",
            RetrievalResult(status="no_evidence"),
            deadline,
        )
        == "후보 답변"
    )
    assert l2.calls[0]["timeout_seconds"] == 24.0


async def test_verifier_propagates_cancellation_and_programmer_errors():
    candidate = "후보 답변"
    kwargs = dict(
        messages=[ChatMessage(role="user", content="질문")],
        candidate=candidate,
        retrieval=RetrievalResult(status="no_evidence"),
        deadline=RequestDeadline.start(),
    )
    with pytest.raises(asyncio.CancelledError):
        await AnswerVerifier(ScriptedL2(error=asyncio.CancelledError()), settings()).verify(
            **kwargs
        )
    with pytest.raises(ValueError):
        await AnswerVerifier(ScriptedL2(error=ValueError("bug")), settings()).verify(**kwargs)
    with pytest.raises(RetrievalError):
        await AnswerVerifier(
            ScriptedL2(error=RetrievalError("not an L2 failure")), settings()
        ).verify(**kwargs)
