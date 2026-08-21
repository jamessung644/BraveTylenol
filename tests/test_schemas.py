import pytest
from pydantic import ValidationError

from lunit_hackathon.schemas import ChatCompletionRequest


def test_request_normalizes_modern_text_only_openai_fields():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": "안전 지침"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "첫 문장"},
                        {"type": "text", "text": "과 둘째 문장"},
                    ],
                },
            ],
            "stream": None,
            "max_tokens": 700,
            "max_completion_tokens": 600,
        }
    )

    assert [(message.role, message.content) for message in request.messages] == [
        ("system", "안전 지침"),
        ("user", "첫 문장과 둘째 문장"),
    ]
    assert request.stream is None
    assert request.requested_max_tokens == 600


def test_request_rejects_unsupported_non_text_content_parts():
    with pytest.raises(ValidationError, match="Only text content parts are supported"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://example.test"}}
                        ],
                    }
                ]
            }
        )


def test_request_without_token_limit_has_no_requested_cap():
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "질문"}]}
    )

    assert request.requested_max_tokens is None
