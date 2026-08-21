import json
from http.client import RemoteDisconnected
from urllib.error import HTTPError

import pytest

from scripts import verify_submission


class _Response:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._body = json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self) -> bytes:
        return self._body


def test_request_json_sends_key_only_as_bearer_header(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["authorization"] = request.get_header("Authorization")
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response({"object": "list", "data": []})

    monkeypatch.setattr(verify_submission, "urlopen", fake_urlopen)

    status, payload = verify_submission.request_json(
        "http://127.0.0.1:8000/v1/models",
        bearer_token="top-secret",
        timeout=3,
    )

    assert status == 200
    assert payload["object"] == "list"
    assert seen == {
        "authorization": "Bearer top-secret",
        "url": "http://127.0.0.1:8000/v1/models",
        "timeout": 3,
    }


def test_docker_run_command_does_not_contain_api_key():
    command = verify_submission.docker_run_command(
        image="brave-tylenol:verify",
        container_name="brave-tylenol-verify-123",
        port=8123,
        harness_mode="passthrough",
    )

    assert command == [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        "brave-tylenol-verify-123",
        "--publish",
        "127.0.0.1:8123:8000",
        "--env",
        "HARNESS_MODE=passthrough",
        "brave-tylenol:verify",
    ]
    assert all("LUNIT" not in argument for argument in command)


def test_validate_chat_completion_accepts_openai_shape():
    summary = verify_submission.validate_chat_completion(
        {
            "object": "chat.completion",
            "model": "team-chatbot",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "정상 응답"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )

    assert summary == {
        "model": "team-chatbot",
        "finish_reason": "stop",
        "content_chars": 5,
        "total_tokens": 5,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"object": "chat.completion", "choices": []},
        {
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        },
    ],
)
def test_validate_chat_completion_rejects_invalid_or_empty_response(payload):
    with pytest.raises(verify_submission.VerificationError):
        verify_submission.validate_chat_completion(payload)


def test_request_json_maps_http_error_without_leaking_authorization(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise HTTPError(
            "http://127.0.0.1:8000/v1/chat/completions",
            503,
            "Unavailable",
            {},
            None,
        )

    monkeypatch.setattr(verify_submission, "urlopen", fake_urlopen)

    with pytest.raises(verify_submission.HttpStatusError) as caught:
        verify_submission.request_json(
            "http://127.0.0.1:8000/v1/chat/completions",
            bearer_token="top-secret",
            timeout=3,
        )

    assert caught.value.status == 503
    assert "top-secret" not in str(caught.value)


def test_request_json_maps_startup_disconnect_to_verification_error(monkeypatch):
    def fake_urlopen(request, timeout):
        del request, timeout
        raise RemoteDisconnected("server is still starting")

    monkeypatch.setattr(verify_submission, "urlopen", fake_urlopen)

    with pytest.raises(verify_submission.VerificationError, match="RemoteDisconnected"):
        verify_submission.request_json("http://127.0.0.1:8000/v1/models", timeout=2)
