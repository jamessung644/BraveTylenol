"""Zero-dependency, no-upstream Korean response baseline."""

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
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MODEL_ID = "team-chatbot"
UPSTREAM_MODEL_ID = "Lunit/L2-preview"
UPSTREAM_CHAT_COMPLETIONS_URL = "https://model.hackathon.lunit.io/v1/chat/completions"
L2_TIMEOUT_SECONDS = 30.0
L2_MAX_TOKENS = 4_096
MAX_API_KEY_LENGTH = 4_096
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


class BaselineServer(ThreadingHTTPServer):
    """Small concurrent server with no network or model dependencies."""

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
        "id": f"chatcmpl-baseline-{uuid.uuid4().hex}",
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
    api_key = _resolve_lunit_api_key(authorization, environment)
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
    return token or None


def _is_valid_lunit_key(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    key = value.strip()
    return (
        key.startswith("lunit_")
        and len(value) <= MAX_API_KEY_LENGTH
        and not any(character in value for character in "\r\n")
    )


def _resolve_lunit_api_key(
    authorization: str | None,
    environment: Mapping[str, str],
) -> str | None:
    environment_key = environment.get("LUNIT_FM_API_KEY")
    if _is_valid_lunit_key(environment_key):
        return environment_key.strip()
    bearer_key = _bearer_token(authorization)
    return bearer_key.strip() if _is_valid_lunit_key(bearer_key) else None


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

    protocol_version = "HTTP/1.1"
    server_version = "BraveTylenolBaseline/1.0"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # Deliberately avoid logging request headers or evaluator credentials.
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
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
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
            return HTTPStatus.OK, {
                "status": "ok",
                "mode": "static-korean-baseline",
            }
        return HTTPStatus.NOT_FOUND, {"detail": "Not found"}

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
        self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

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
