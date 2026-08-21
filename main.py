"""Minimal, zero-dependency Lunit L2 conversation driver."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MODEL_ID = "team-chatbot"
UPSTREAM_MODEL_ID = "Lunit/L2-preview"
UPSTREAM_CHAT_COMPLETIONS_URL = "https://model.hackathon.lunit.io/v1/chat/completions"
L2_TIMEOUT_SECONDS = 18.0
L2_MAX_TOKENS = 4_096
KOREAN_BASELINE_RESPONSE = (
    "질문을 확인했습니다. 증상이 심하거나 갑자기 악화되면 즉시 119 또는 "
    "응급실의 도움을 받고, 정확한 판단을 위해 의료 전문가와 상담해 주세요."
)
MEDICAL_SYSTEM_PROMPT = (
    "Answer the user's health question directly in the user's language. Give accurate, "
    "practical medical information, important red flags, and clear next steps. Do not "
    "invent patient facts or expose reasoning. Return only the final user-facing answer."
)
CompletionProvider = Callable[[dict[str, Any], str | None], dict[str, Any]]


class ClientRequestError(ValueError):
    """The evaluator request cannot be forwarded safely."""


class ConfigurationError(RuntimeError):
    """The L2 endpoint or credential is not configured."""


class L2ResponseError(RuntimeError):
    """L2 could not provide a usable final answer."""


class L2TimeoutError(L2ResponseError):
    """The bounded L2 attempts exhausted their deadline."""


def _is_timeout_error(error: object) -> bool:
    # socket.timeout became an alias of TimeoutError only in newer Python.
    return isinstance(error, TimeoutError) or isinstance(error, socket.timeout)  # noqa: UP041


CompletionProvider = Callable[..., dict[str, Any]]

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        completion_provider: CompletionProvider | None = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.completion_provider = completion_provider or request_l2_or_fallback


def models_payload() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "brave-tylenol",
            }
        ],
    }


def completion_payload(
    content: str = KOREAN_BASELINE_RESPONSE,
    *,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_usage = usage if isinstance(usage, Mapping) else {}

    def token_count(name: str) -> int:
        value = normalized_usage.get(name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    prompt_tokens = max(0, token_count("prompt_tokens"))
    completion_tokens = max(0, token_count("completion_tokens"))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def request_l2_or_fallback(
    request_payload: dict[str, Any],
    authorization: str | None,
    *,
    opener: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Try L2 once within the inference budget, otherwise return the baseline."""

    environment = os.environ if environ is None else environ
    api_key = _bearer_token(authorization) or environment.get("LUNIT_FM_API_KEY", "").strip()
    messages = _normalized_messages(request_payload.get("messages"))
    if not api_key or not messages:
        return completion_payload()

    upstream_payload = {
        "model": UPSTREAM_MODEL_ID,
        "messages": [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            *messages,
        ],
        "max_tokens": min(
            L2_MAX_TOKENS,
            request_payload.get("max_tokens", L2_MAX_TOKENS)
            if isinstance(request_payload.get("max_tokens"), int)
            else L2_MAX_TOKENS,
        ),
        "reasoning_effort": "low",
        "temperature": 0.0,
        "stream": False,
    }
    encoded = json.dumps(upstream_payload, ensure_ascii=False).encode("utf-8")
    upstream_request = Request(
        UPSTREAM_CHAT_COMPLETIONS_URL,
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with (opener or urlopen)(
            upstream_request,
            timeout=L2_TIMEOUT_SECONDS,
        ) as response:
            if not 200 <= int(getattr(response, "status", 200)) < 300:
                return completion_payload()
            upstream = json.loads(response.read(4_000_001).decode("utf-8"))
        message = upstream["choices"][0]["message"]
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            return completion_payload()
        return completion_payload(content.strip(), usage=upstream.get("usage"))
    except Exception:
        return completion_payload()


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return None
    return token.strip() or None


def _normalized_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for message in value:
        if not isinstance(message, Mapping):
            return []
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            return []
        content = message.get("content")
        if not isinstance(content, str):
            return []
        normalized.append({"role": role, "content": content})
    return normalized


class BaselineHandler(BaseHTTPRequestHandler):
    """OpenAI-shaped endpoint with one bounded L2 attempt and a static fallback."""


def request_l2_completion(
    request_payload: dict[str, Any],
    authorization: str | None,
    *,
    opener: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate one answer with L2, allowing one recovery only for blank text."""

    environment = os.environ if environ is None else environ
    api_key = _bearer_token(authorization) or _configured_api_key(environment)
    if api_key is None:
        raise ConfigurationError("L2 API credential is unavailable")

    messages = _normalized_messages(request_payload.get("messages"))
    if not messages:
        raise ClientRequestError("messages must contain at least one text message")

    upstream_payload: dict[str, Any] = {
        "model": UPSTREAM_MODEL_ID,
        "messages": [
            {"role": "system", "content": DIRECT_MEDICAL_SYSTEM_PROMPT},
            *messages,
        ],
        # CoEval supplies 6,144. L2 reasoning consumes this same budget, so the
        # integration branch's accidental 1,024 cap could end before final text.
        "max_tokens": _completion_token_budget(request_payload),
        "reasoning_effort": "low",
        "temperature": 0.0,
        "stream": False,
    }
    open_request = opener or urlopen
    # The evaluator allows 180 seconds. Keep the deadline independent from
    # stale integration environment values such as REQUEST_TIMEOUT_SECONDS=3.
    deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_SECONDS

    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise L2TimeoutError("L2 request deadline exhausted")
        try:
            status, raw_response = _post_json(
                url=UPSTREAM_CHAT_COMPLETIONS_URL,
                api_key=api_key,
                payload=upstream_payload,
                timeout=remaining,
                opener=open_request,
            )
        except HTTPError as error:
            raise L2ResponseError("L2 returned an unsuccessful status") from error
        except URLError as error:
            if _is_timeout_error(error.reason):
                raise L2TimeoutError("L2 request timed out") from error
            raise L2ResponseError("L2 transport failed") from error
        except OSError as error:
            if _is_timeout_error(error):
                raise L2TimeoutError("L2 request timed out") from error
            raise L2ResponseError("L2 transport failed") from error

        if not 200 <= status < 300:
            raise L2ResponseError("L2 returned an unsuccessful status")

        try:
            content, finish_reason, usage = _parse_l2_completion(raw_response)
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            if attempt == 0:
                upstream_payload = _recovery_payload(upstream_payload)
                continue
            raise L2ResponseError("L2 returned no usable final text") from error
        return completion_payload(
            content,
            finish_reason=finish_reason,
            usage=usage,
        )

    raise L2ResponseError("L2 returned no usable final text")


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
    opener: Callable[..., Any],
) -> tuple[int, bytes]:
    encoded_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    upstream_request = Request(
        url,
        data=encoded_payload,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "BraveTylenol-Minimal/1.0",
        },
        method="POST",
    )
    with opener(upstream_request, timeout=timeout) as response:
        status = int(getattr(response, "status", HTTPStatus.OK))
        raw_response = response.read(MAX_UPSTREAM_RESPONSE_BYTES + 1)
    if len(raw_response) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise L2ResponseError("L2 response exceeded the size limit")
    return status, raw_response


def _parse_l2_completion(
    raw_response: bytes,
) -> tuple[str, str, Mapping[str, Any] | None]:
    upstream = json.loads(raw_response.decode("utf-8"))
    choice = upstream["choices"][0]
    message = choice["message"]
    if not isinstance(message, Mapping):
        raise ValueError("invalid message")
    content = _text_content(message.get("content"))
    if content is None or not content.strip():
        raise ValueError("blank final text")
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    if not isinstance(finish_reason, str):
        finish_reason = "stop"
    usage = upstream.get("usage") if isinstance(upstream, Mapping) else None
    return content, finish_reason, usage if isinstance(usage, Mapping) else None


def _recovery_payload(original: Mapping[str, Any]) -> dict[str, Any]:
    recovered = dict(original)
    recovered["messages"] = [
        *original["messages"],
        {
            "role": "system",
            "content": (
                "The previous attempt did not return usable final text. Return only the "
                "complete user-facing answer now, with no analysis, tool calls, XML, or preamble."
            ),
        },
    ]
    return recovered


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    token = token.strip()
    if not separator or scheme.casefold() != "bearer" or not token:
        return None
    if len(token) > 4_096 or any(character in token for character in "\r\n"):
        return None
    return token


def _configured_api_key(environment: Mapping[str, str]) -> str | None:
    api_key = environment.get("LUNIT_FM_API_KEY", "").strip()
    return None if api_key in _PLACEHOLDER_API_KEYS else api_key


def _completion_token_budget(
    request_payload: Mapping[str, Any],
) -> int:
    requested = [
        value
        for value in (
            request_payload.get("max_tokens"),
            request_payload.get("max_completion_tokens"),
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    return min(DEFAULT_MAX_COMPLETION_TOKENS, *requested) if requested else (
        DEFAULT_MAX_COMPLETION_TOKENS
    )


def _normalized_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ClientRequestError("each message must be an object")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise ClientRequestError("unsupported message role")
        content = _text_content(message.get("content"))
        if content is None:
            raise ClientRequestError("message content must be text")
        normalized.append({"role": role, "content": content})
    return normalized


def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    text_parts: list[str] = []
    for part in value:
        if not isinstance(part, Mapping) or part.get("type") not in {"text", "input_text"}:
            return None
        text = part.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    return "".join(text_parts)


def _normalized_usage(value: Mapping[str, Any] | None) -> dict[str, int]:
    usage = value if isinstance(value, Mapping) else {}

    def token_count(name: str) -> int:
        count = usage.get(name, 0)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return 0
        return count

    prompt_tokens = token_count("prompt_tokens")
    completion_tokens = token_count("completion_tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class MinimalL2Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        completion_provider: CompletionProvider | None = None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.completion_provider = completion_provider or request_l2_completion


class MinimalL2Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BraveTylenolMinimal/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Never log medical text or evaluator credentials.
        del format, args

    def _path(self) -> str:
        path = urlsplit(self.path).path.rstrip("/")
        return path or "/"

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        *,
        include_body: bool = True,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        if include_body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        self.close_connection = True

    def _get_payload(self) -> tuple[HTTPStatus, dict[str, Any]]:
        path = self._path()
        if path in {"/health", "/healthz"}:
            return HTTPStatus.OK, {"status": "ok"}
        if path == "/v1/models":
            return HTTPStatus.OK, models_payload()
        if path == "/":
            return HTTPStatus.OK, {"status": "ok", "mode": "minimal-l2-direct"}
        return HTTPStatus.NOT_FOUND, {"detail": "Not found"}

    def _read_json_payload(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else 0
        except ValueError as error:
            raise ClientRequestError("Content-Length is invalid") from error
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ClientRequestError("request body size is invalid")
        try:
            self.connection.settimeout(5.0)
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ClientRequestError("request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ClientRequestError("request body must be an object")
        return payload

    def do_GET(self) -> None:
        status, payload = self._get_payload()
        self._send_json(status, payload)

    def do_HEAD(self) -> None:
        status, payload = self._get_payload()
        self._send_json(status, payload, include_body=False)

    def _read_request_payload(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if content_length <= 0 or content_length > 1_000_000:
            return {}
        try:
            self.connection.settimeout(3.0)
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_POST(self) -> None:
        if self._path() == "/v1/chat/completions":
            request_payload = self._read_request_payload()
            try:
                provider = self.server.completion_provider
                response_payload = provider(
                    request_payload,
                    self.headers.get("Authorization"),
                )
            except Exception:
                response_payload = completion_payload()
            self._send_json(HTTPStatus.OK, response_payload)
            return
        try:
            request_payload = self._read_json_payload()
            if request_payload.get("stream") is True:
                raise ClientRequestError("streaming is not supported")
            response_payload = self.server.completion_provider(
                request_payload,
                self.headers.get("Authorization"),
            )
        except ClientRequestError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                error_payload(
                    "Invalid chat completion request",
                    "invalid_request_error",
                    "invalid_request",
                ),
            )
            return
        except ConfigurationError:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                error_payload("L2 is not configured", "server_error", "l2_not_configured"),
            )
            return
        except L2TimeoutError:
            self._send_json(
                HTTPStatus.GATEWAY_TIMEOUT,
                error_payload("L2 request timed out", "server_error", "l2_timeout"),
            )
            return
        except Exception:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                error_payload("L2 request failed", "server_error", "l2_failure"),
            )
            return
        self._send_json(HTTPStatus.OK, response_payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


def create_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    completion_provider: CompletionProvider | None = None,
) -> BaselineServer:
    return BaselineServer(
        (host, port),
        BaselineHandler,
        completion_provider=completion_provider,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded L2 Korean baseline")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = create_server(args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
