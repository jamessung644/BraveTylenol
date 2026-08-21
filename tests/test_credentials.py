import pytest

import lunit_hackathon.credentials as credentials
from lunit_hackathon.credentials import resolve_lunit_api_key
from lunit_hackathon.errors import ConfigurationError


def test_resolver_rejects_evaluator_dummy_and_uses_embedded_key():
    value = resolve_lunit_api_key("Bearer evaluator-token", None)

    assert value.startswith("lunit_")


def test_environment_key_precedes_valid_bearer():
    assert resolve_lunit_api_key("Bearer lunit_request", "lunit_environment") == (
        "lunit_environment"
    )


def test_resolver_raises_when_every_candidate_is_malformed(monkeypatch):
    monkeypatch.setattr(credentials, "EMBEDDED_LUNIT_API_KEY", "not-a-lunit-key")

    with pytest.raises(ConfigurationError):
        resolve_lunit_api_key("Basic lunit_request", "lunit_environment\ninvalid")
