"""Killable, aggregate-safe JSON HTTP requests for local verification scripts."""

from __future__ import annotations

import json
import multiprocessing
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_PROCESS_JOIN_SECONDS = 0.1


def _post_json_worker(
    connection: Any,
    url: Any,
    payload: Any,
    headers: Any,
    timeout_seconds: float,
) -> None:
    """Perform all serialisation and I/O in a process the parent can terminate."""
    result: tuple[int, Any | None] = (0, None)
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - supplied CLI URL.
            status = response.status
            try:
                result = status, json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                result = status, None
    except HTTPError as error:
        result = error.code, None
    except Exception:
        result = 0, None
    try:
        connection.send(result)
    except Exception:
        pass
    finally:
        try:
            connection.close()
        except Exception:
            pass


def post_json(
    url: Any,
    payload: Any,
    headers: Any,
    timeout_seconds: float,
) -> tuple[int, Any | None]:
    """Return a sanitized result under a hard wall-clock deadline with no zombie child."""
    if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds:
        return 0, None
    started = time.perf_counter()
    process = None
    receiver = None
    sender = None
    process_started = False
    cleanup_failed = False
    result: tuple[int, Any | None] = (0, None)
    try:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_post_json_worker,
            args=(sender, url, payload, headers, timeout_seconds),
            daemon=True,
        )
        process.start()
        process_started = True
        sender.close()
        sender = None
        remaining = max(0.0, timeout_seconds - (time.perf_counter() - started))
        if receiver.poll(remaining):
            try:
                status, response = receiver.recv()
                if isinstance(status, int):
                    result = status, response
            except (EOFError, OSError, ValueError):
                pass
    except Exception:
        result = 0, None
    finally:
        if sender is not None:
            try:
                sender.close()
            except Exception:
                cleanup_failed = True
        if receiver is not None:
            try:
                receiver.close()
            except Exception:
                cleanup_failed = True
        if process is not None and process_started:
            alive = _is_alive(process)
            if alive is not False:
                cleanup_failed = not _terminate(process) or cleanup_failed
            cleanup_failed = not _join(process) or cleanup_failed
            alive_after = _is_alive(process)
            if alive_after is not False:
                cleanup_failed = not _kill(process) or cleanup_failed
            cleanup_failed = not _join(process) or cleanup_failed
            if _is_alive(process) is not False:
                cleanup_failed = True
    return (0, None) if cleanup_failed else result


def _is_alive(process: Any) -> bool | None:
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return None


def _terminate(process: Any) -> bool:
    try:
        process.terminate()
        return True
    except (AssertionError, OSError, ValueError):
        return False


def _kill(process: Any) -> bool:
    try:
        process.kill()
        return True
    except (AssertionError, OSError, ValueError):
        return False


def _join(process: Any) -> bool:
    try:
        process.join(_PROCESS_JOIN_SECONDS)
        return True
    except (AssertionError, OSError, ValueError):
        return False
