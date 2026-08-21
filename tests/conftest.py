from collections.abc import Sequence

import pytest

from harness.config import Settings
from harness.schemas import ChatMessage


class FakeOrchestrator:
    def __init__(self, answer_text: str = "반갑습니다.") -> None:
        self.answer_text = answer_text
        self.calls: list[Sequence[ChatMessage]] = []

    async def answer(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.answer_text


@pytest.fixture
def settings_without_key() -> Settings:
    return Settings()


@pytest.fixture
def settings_with_key(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LUNIT_FM_API_KEY", "test-key")
    return Settings()


@pytest.fixture
def fake_orchestrator() -> FakeOrchestrator:
    return FakeOrchestrator()
