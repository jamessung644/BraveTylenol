from harness.config import Settings


def test_default_harness_mode_is_passthrough(monkeypatch):
    monkeypatch.delenv("HARNESS_MODE", raising=False)

    assert Settings().harness_mode == "passthrough"


def test_rag_mode_remains_selectable(monkeypatch):
    monkeypatch.setenv("HARNESS_MODE", "rag")

    assert Settings().harness_mode == "rag"
