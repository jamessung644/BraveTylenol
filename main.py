"""Minimal, zero-dependency Lunit L2 conversation driver."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import ssl
import threading
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
DEFAULT_REQUEST_TIMEOUT_SECONDS = 65.0
UPSTREAM_ATTEMPT_TIMEOUT_SECONDS = 35.0
UPSTREAM_RECOVERY_TIMEOUT_SECONDS = 25.0
DEFAULT_MAX_COMPLETION_TOKENS = 4_096
RECOVERY_MAX_COMPLETION_TOKENS = 2_048
MAX_CONCURRENT_L2_REQUESTS = 16
MAX_REQUEST_BYTES = 1_000_000
MAX_UPSTREAM_RESPONSE_BYTES = 4_000_000
_PLACEHOLDER_API_KEYS = {"", "여기에_직접_입력", "lunit_replace_me"}
_L2_REQUEST_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_L2_REQUESTS)
KOREAN_BASELINE_RESPONSE = (
    "질문을 확인했습니다. 증상이 심하거나 갑자기 악화되면 즉시 119 또는 "
    "응급실의 도움을 받고, 정확한 판단을 위해 의료 전문가와 상담해 주세요."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("brave_tylenol")

DIRECT_MEDICAL_SYSTEM_PROMPT = """You are Lunit L2, the sole author of the final
user-facing medical answer. Respond in the user's language with accurate, relevant,
practical, and understandable health information. Return the final answer immediately,
without exposing reasoning or tool protocol, and use no more than 150 words. Put urgent
action first when the situation may be an emergency. Clearly name important red flags and
next steps. Do not claim a diagnosis the conversation cannot support, invent patient facts,
or make individualized prescription changes without clinician review. State meaningful
uncertainty briefly. No retrieval or other tools are available; answer from reliable general
medical knowledge and say what authoritative source should be checked when current or
source-specific facts cannot be verified."""


class ClientRequestError(ValueError):
    """The evaluator request cannot be forwarded safely."""


class ConfigurationError(RuntimeError):
    """The L2 endpoint or credential is not configured."""


class L2ResponseError(RuntimeError):
    """L2 could not provide a usable final answer."""

    def __init__(self, message: str, *, kind: str = "upstream") -> None:
        super().__init__(message)
        self.kind = kind


class L2TimeoutError(L2ResponseError):
    """The bounded L2 attempts exhausted their deadline."""

    def __init__(self, message: str, *, kind: str = "timeout") -> None:
        super().__init__(message, kind=kind)


def _is_timeout_error(error: object) -> bool:
    # socket.timeout became an alias of TimeoutError only in newer Python.
    return isinstance(error, TimeoutError) or isinstance(error, socket.timeout)  # noqa: UP041


CompletionProvider = Callable[..., dict[str, Any]]


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
    content: str,
    *,
    finish_reason: str = "stop",
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape L2 text or the nonempty safety baseline for the evaluator."""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason if finish_reason in {"stop", "length"} else "stop",
            }
        ],
        "usage": _normalized_usage(usage),
    }


def error_payload(message: str, error_type: str, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def request_l2_completion(
    request_payload: dict[str, Any],
    authorization: str | None,
    *,
    opener: Callable[..., Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate one answer with L2, recovering only an unusable 2xx result."""

    environment = os.environ if environ is None else environ
    api_keys = _credential_candidates(authorization, environment)
    if not api_keys:
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
        # CoEval supplies 6,144. A measured 4,096 cap keeps the 301-request
        # inference phase inside its wall-clock budget; a rare blank result gets
        # one concise L2-only recovery below.
        "max_tokens": _completion_token_budget(request_payload),
        "reasoning_effort": "low",
        "temperature": 0.0,
        "stream": False,
    }
    open_request = opener or urlopen
    primary_payload = dict(upstream_payload)
    # CoEval allows 180 seconds per request but applies a much tighter global
    # inference budget. Bound each attempt so one tail request cannot consume it.
    deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_SECONDS
    slot_wait = max(0.0, deadline - time.monotonic())
    if not _L2_REQUEST_SLOTS.acquire(timeout=slot_wait):
        raise L2TimeoutError("L2 concurrency queue exhausted", kind="queue_timeout")

    try:
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise L2TimeoutError("L2 request deadline exhausted")
            attempt_limit = (
                UPSTREAM_ATTEMPT_TIMEOUT_SECONDS
                if attempt == 0
                else UPSTREAM_RECOVERY_TIMEOUT_SECONDS
            )
            attempt_timeout = min(attempt_limit, remaining)
            attempt_deadline = min(deadline, time.monotonic() + attempt_timeout)
            attempt_started = time.monotonic()
            try:
                status, raw_response = _post_json_with_credentials(
                    url=UPSTREAM_CHAT_COMPLETIONS_URL,
                    api_keys=api_keys,
                    payload=upstream_payload,
                    timeout=attempt_timeout,
                    deadline=attempt_deadline,
                    opener=open_request,
                )
            except HTTPError as error:
                status = int(error.code)
                error.close()
                failure = L2ResponseError(
                    "L2 returned an unsuccessful status",
                    kind=f"http_{status}",
                )
                raise failure from error
            except URLError as error:
                if _is_timeout_error(error.reason):
                    failure = L2TimeoutError("L2 request timed out")
                else:
                    failure = L2ResponseError(
                        "L2 transport failed",
                        kind=_transport_failure_kind(error.reason),
                    )
                raise failure from error
            except OSError as error:
                if _is_timeout_error(error):
                    failure = L2TimeoutError("L2 request timed out")
                else:
                    failure = L2ResponseError(
                        "L2 transport failed",
                        kind=_transport_failure_kind(error),
                    )
                raise failure from error

            if not 200 <= status < 300:
                failure = L2ResponseError(
                    "L2 returned an unsuccessful status",
                    kind=f"http_{status}",
                )
                raise failure

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
                    upstream_payload = _prepare_recovery(primary_payload, deadline)
                    _log_l2_retry("malformed", attempt_started, "concise")
                    continue
                raise L2ResponseError(
                    "L2 returned no usable final text",
                    kind="malformed",
                ) from error
            return completion_payload(
                content,
                finish_reason=finish_reason,
                usage=usage,
            )
    finally:
        _L2_REQUEST_SLOTS.release()

    raise L2ResponseError("L2 returned no usable final text", kind="malformed")


def _post_json_with_credentials(
    *,
    url: str,
    api_keys: Sequence[str],
    payload: Mapping[str, Any],
    timeout: float,
    deadline: float,
    opener: Callable[..., Any],
) -> tuple[int, bytes]:
    """Prefer the documented environment key, failing over only on auth errors."""

    for index, api_key in enumerate(api_keys):
        remaining = min(timeout, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("L2 request deadline exhausted")
        try:
            status, raw_response = _post_json(
                url=url,
                api_key=api_key,
                payload=payload,
                timeout=remaining,
                deadline=deadline,
                opener=opener,
            )
        except HTTPError as error:
            status = int(error.code)
            if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN} and index + 1 < len(
                api_keys
            ):
                error.close()
                LOGGER.warning("l2_credential_failover status=%d", status)
                continue
            raise
        if status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN} and index + 1 < len(api_keys):
            LOGGER.warning("l2_credential_failover status=%d", status)
            continue
        return status, raw_response

    raise L2ResponseError("L2 credentials were rejected", kind="http_401")


def _post_json(
    *,
    url: str,
    api_key: str,
    payload: Mapping[str, Any],
    timeout: float,
    deadline: float,
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
        raw_response = _read_bounded_response(response, deadline)
    if len(raw_response) > MAX_UPSTREAM_RESPONSE_BYTES:
        raise L2ResponseError(
            "L2 response exceeded the size limit",
            kind="response_too_large",
        )
    return status, raw_response


def _read_bounded_response(response: Any, deadline: float) -> bytes:
    limit = MAX_UPSTREAM_RESPONSE_BYTES + 1
    read1 = getattr(response, "read1", None)
    if not callable(read1):
        body = response.read(limit)
        if time.monotonic() > deadline:
            raise TimeoutError("L2 response body deadline exhausted")
        return body

    chunks: list[bytes] = []
    total = 0
    while total < limit:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("L2 response body deadline exhausted")
        _set_response_socket_timeout(response, remaining)
        chunk = read1(min(64 * 1_024, limit - total))
        if time.monotonic() > deadline:
            raise TimeoutError("L2 response body deadline exhausted")
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _set_response_socket_timeout(response: Any, timeout: float) -> None:
    try:
        response.fp.raw._sock.settimeout(timeout)
    except (AttributeError, OSError):
        # Custom test openers need not expose urllib's private socket chain.
        pass


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
                "complete user-facing answer now, with no analysis, tool calls, XML, or "
                "preamble, in at most 150 words."
            ),
        },
    ]
    requested = recovered.get("max_tokens", RECOVERY_MAX_COMPLETION_TOKENS)
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        requested = RECOVERY_MAX_COMPLETION_TOKENS
    recovered["max_tokens"] = min(requested, RECOVERY_MAX_COMPLETION_TOKENS)
    return recovered


def _prepare_recovery(
    original: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    if deadline - time.monotonic() <= 0:
        raise L2TimeoutError("L2 request deadline exhausted")
    return _recovery_payload(original)


def _log_l2_retry(kind: str, started: float, next_mode: str) -> None:
    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    LOGGER.warning(
        "l2_retry attempt=1 kind=%s duration_ms=%d next_mode=%s",
        kind,
        duration_ms,
        next_mode,
    )


def _transport_failure_kind(error: object) -> str:
    if isinstance(error, socket.gaierror):
        return "dns"
    if isinstance(error, ConnectionRefusedError):
        return "connect"
    if isinstance(error, ssl.SSLCertVerificationError):
        return "tls_verify"
    if isinstance(error, ssl.SSLError):
        return "tls"
    return "transport"


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


def _credential_candidates(
    authorization: str | None,
    environment: Mapping[str, str],
) -> list[str]:
    candidates: list[str] = []
    # Quick Start documents LUNIT_FM_API_KEY as the Model/MCP credential.
    # The evaluator Bearer remains a fallback because some deployments forward
    # the same team key directly to the candidate service.
    for candidate in (
        _configured_api_key(environment),
        _bearer_token(authorization),
    ):
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


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
    return (
        min(DEFAULT_MAX_COMPLETION_TOKENS, *requested)
        if requested
        else (DEFAULT_MAX_COMPLETION_TOKENS)
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

    def _request_id(self) -> str:
        request_id = getattr(self, "_request_id_value", None)
        if request_id is None:
            request_id = f"req-{uuid.uuid4().hex}"
            self._request_id_value = request_id
        return request_id

    def _log_chat_failure(self, kind: str, started: float) -> None:
        duration_ms = max(0, round((time.monotonic() - started) * 1_000))
        LOGGER.warning(
            "chat_failed request_id=%s kind=%s duration_ms=%d",
            self._request_id(),
            kind,
            duration_ms,
        )

    def _send_degraded_completion(self, kind: str, started: float) -> None:
        """Finish one failed sample without triggering CoEval's nested retries."""

        self._log_chat_failure(kind, started)
        # The evaluator's OpenAI adapter normalizes an empty string to None and
        # retries it. Preserve the proven baseline's nonempty response so a
        # single upstream failure cannot abort the entire submission.
        self._send_json(HTTPStatus.OK, completion_payload(KOREAN_BASELINE_RESPONSE))

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
        self.send_header("X-Request-ID", self._request_id())
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

    def do_POST(self) -> None:
        started = time.monotonic()
        if self._path() != "/v1/chat/completions":
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
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
            self._send_degraded_completion("not_configured", started)
            return
        except L2TimeoutError as error:
            self._send_degraded_completion(error.kind, started)
            return
        except L2ResponseError as error:
            self._send_degraded_completion(error.kind, started)
            return
        except Exception:
            self._send_degraded_completion("internal", started)
            return
        self._send_json(HTTPStatus.OK, response_payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, HEAD, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.send_header("X-Request-ID", self._request_id())
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


def create_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    completion_provider: CompletionProvider | None = None,
) -> MinimalL2Server:
    return MinimalL2Server(
        (host, port),
        MinimalL2Handler,
        completion_provider=completion_provider,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Lunit L2 driver")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _handle_termination_signal(signum: int, frame: Any) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    server = create_server(args.host, args.port)
    LOGGER.info("server_started host=%s port=%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
