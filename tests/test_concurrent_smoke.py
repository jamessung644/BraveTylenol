import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lunit_hackathon.orchestrator import MEDICAL_SAFETY_FALLBACK


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "concurrent_smoke.py"
    spec = importlib.util.spec_from_file_location("concurrent_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_completion_accepts_a_valid_fallback_envelope():
    """Dropping valid safety fallbacks would turn expected upstream failures into smoke failures."""
    smoke = _load_smoke_module()
    payload = {
        "model": "team-chatbot",
        "choices": [
            {
                "message": {"role": "assistant", "content": smoke.SAFETY_FALLBACK},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    assert smoke.validate_completion(payload, latency_seconds=1.5) == (True, True)


def test_fallback_literal_matches_the_server_recovery_envelope():
    """Changing only one fallback literal would make fallback accounting silently drift."""
    smoke = _load_smoke_module()

    assert smoke.SAFETY_FALLBACK == MEDICAL_SAFETY_FALLBACK


def test_validate_completion_rejects_bad_envelopes_and_slow_responses():
    """Removing a response-shape or deadline check must fail this test."""
    smoke = _load_smoke_module()
    valid = {
        "model": "team-chatbot",
        "choices": [
            {"message": {"role": "assistant", "content": "안내입니다."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    assert smoke.validate_completion({**valid, "model": "other"}, 1.0) == (False, False)
    assert smoke.validate_completion({**valid, "usage": {"prompt_tokens": -1}}, 1.0) == (
        False,
        False,
    )
    assert smoke.validate_completion(valid, latency_seconds=165.01) == (False, False)


def test_aggregate_counts_structural_fallbacks_without_exposing_response_text():
    """A regression that loses fallback or HTTP count accounting must fail this test."""
    smoke = _load_smoke_module()
    results = [
        smoke.RequestResult(status=200, latency_seconds=1.0, success=True, fallback=False),
        smoke.RequestResult(status=200, latency_seconds=3.0, success=True, fallback=True),
        smoke.RequestResult(status=503, latency_seconds=2.0, success=False, fallback=False),
    ]

    assert smoke.aggregate(results) == {
        "requests": 3,
        "success": 2,
        "fallback": 1,
        "failure": 1,
        "status_counts": {200: 2, 503: 1},
        "latencies": [1.0, 3.0, 2.0],
    }


def test_exit_code_is_nonzero_when_any_request_failed():
    """Returning success despite an invalid completion must fail this test."""
    smoke = _load_smoke_module()
    summary = {
        "requests": 2,
        "success": 1,
        "fallback": 0,
        "failure": 1,
        "status_counts": {200: 1, 500: 1},
        "latencies": [0.1, 0.2],
    }

    assert smoke.exit_code(summary) == 1


class _DripHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json({"data": [{"id": "team-chatbot"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100000")
        self.end_headers()
        self.wfile.write(b"{")
        self.wfile.flush()
        time.sleep(2.0)

    def _send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def _drip_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DripHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _run_smoke_cli(*args):
    path = Path(__file__).resolve().parents[1] / "scripts" / "concurrent_smoke.py"
    return subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_drip_fed_response_hits_absolute_deadline_and_exits_promptly():
    """A peer that continually drips bytes must not hold a worker beyond the absolute deadline."""
    server = _drip_server()
    started = time.perf_counter()
    try:
        result = _run_smoke_cli(
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--requests",
            "1",
            "--concurrency",
            "1",
            "--deadline-seconds",
            "0.2",
        )
    finally:
        server.shutdown()
        server.server_close()
    elapsed = time.perf_counter() - started

    assert result.returncode == 1
    assert elapsed < 2.0
    assert result.stdout.startswith("requests=1 success=0 fallback=0 failure=1\n")
    assert "Traceback" not in result.stderr


def test_cli_normalizes_malformed_base_url_to_aggregate_failure():
    """A malformed URL must not escape before smoke output is sanitized."""
    result = _run_smoke_cli(
        "--base-url",
        "",
        "--requests",
        "1",
        "--concurrency",
        "1",
        "--deadline-seconds",
        "0.2",
    )

    assert result.returncode == 1
    assert result.stdout.startswith("requests=0 success=0 fallback=0 failure=1\n")
    assert result.stderr == ""
    assert "Traceback" not in result.stdout


def test_process_start_failure_is_sanitized_for_completion_and_preflight(monkeypatch):
    """Resource exhaustion while spawning a child must not escape cleanup as an assertion error."""
    smoke = _load_smoke_module()

    class Connection:
        def close(self):
            return None

    class Process:
        def start(self):
            raise OSError("resource exhausted")

        def is_alive(self):
            raise AssertionError("unstarted process inspected")

        def join(self, timeout):
            raise AssertionError("unstarted process joined")

    class Context:
        def Pipe(self, duplex):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda _: Context())

    assert smoke.send_completion("http://127.0.0.1:1", deadline_seconds=0.1).status == 0
    assert smoke.check_preflight("http://127.0.0.1:1", timeout_seconds=0.1) == [0, 0]


def test_cli_rejects_hostile_counts_without_allocating_workers_or_leaking_stderr():
    """Oversized, zero, and negative runtime limits must fail before pool/list allocation."""
    started = time.perf_counter()
    result = _run_smoke_cli(
        "--base-url",
        "http://127.0.0.1:1",
        "--requests",
        "999999999",
        "--concurrency",
        "999999999",
        "--deadline-seconds",
        "0",
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 1
    assert elapsed < 2.0
    assert result.stdout.startswith("requests=0 success=0 fallback=0 failure=1\n")
    assert result.stderr == ""
    assert "Traceback" not in result.stdout


def test_effective_workers_preserve_required_16x16_gate():
    """The production gate must retain all sixteen workers while never exceeding requests."""
    smoke = _load_smoke_module()

    assert smoke.effective_workers(requests=16, concurrency=16) == 16
    assert smoke.effective_workers(requests=3, concurrency=16) == 3


def test_executor_construction_and_submit_failures_return_only_failed_results(monkeypatch):
    """Thread/process resource exhaustion must not escape before aggregate accounting begins."""
    smoke = _load_smoke_module()

    def construction_failure(*args, **kwargs):
        raise OSError("resource exhausted")

    monkeypatch.setattr(smoke, "ThreadPoolExecutor", construction_failure)
    failed = smoke.run_completions("http://127.0.0.1:1", 2, 2, 0.1)
    assert [result.status for result in failed] == [0, 0]
    assert not any(result.success for result in failed)

    class Executor:
        def __init__(self, *args, **kwargs):
            self.shutdown_calls = []

        def submit(self, *args, **kwargs):
            raise RuntimeError("no workers")

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(smoke, "ThreadPoolExecutor", Executor)
    failed = smoke.run_completions("http://127.0.0.1:1", 2, 2, 0.1)
    assert [result.status for result in failed] == [0, 0]
    assert not any(result.success for result in failed)


def test_cleanup_faults_force_status_zero_and_attempt_each_safe_step(monkeypatch):
    """A failed cleanup after a received 200 must not be reported as a successful response."""
    smoke = _load_smoke_module()
    calls = []

    class Connection:
        def close(self):
            return None

        def poll(self, timeout):
            return True

        def recv(self):
            return 200, {"ok": True}

    class Process:
        def start(self):
            calls.append("start")

        def is_alive(self):
            calls.append("is_alive")
            raise OSError("state unavailable")

        def terminate(self):
            calls.append("terminate")
            raise OSError("terminate unavailable")

        def join(self, timeout):
            calls.append("join")
            raise OSError("join unavailable")

        def kill(self):
            calls.append("kill")
            raise OSError("kill unavailable")

    class Context:
        def Pipe(self, duplex):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda _: Context())

    assert smoke._read_json("http://example.test", "/", None, "GET", 0.1) == (0, None)
    assert {"start", "is_alive", "terminate", "join", "kill"} <= set(calls)


def test_worker_pipe_close_failure_is_suppressed_without_false_success():
    """A child-side pipe close error must not print a traceback or become an unhandled success."""
    smoke = _load_smoke_module()

    class Connection:
        def send(self, value):
            raise OSError("send unavailable")

        def close(self):
            raise RuntimeError("close unavailable")

    smoke._read_json_worker(Connection(), "", "/", None, "GET", 0.1)


def test_shutdown_failure_overrides_successful_futures(monkeypatch):
    """A failed executor shutdown must fail closed even when every future returned 200."""
    smoke = _load_smoke_module()

    class Future:
        def result(self, timeout):
            return smoke.RequestResult(
                status=200,
                latency_seconds=0.0,
                success=True,
                fallback=False,
            )

    class Executor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            return Future()

        def shutdown(self, **kwargs):
            raise RuntimeError("shutdown unavailable")

    monkeypatch.setattr(smoke, "ThreadPoolExecutor", Executor)

    results = smoke.run_completions("http://127.0.0.1:1", 2, 2, 0.1)
    assert [result.status for result in results] == [0, 0]
    assert not any(result.success for result in results)


def test_process_still_alive_after_kill_fails_closed(monkeypatch):
    """An unverified dead child must never permit a 200 result."""
    smoke = _load_smoke_module()

    class Connection:
        def close(self):
            return None

        def poll(self, timeout):
            return True

        def recv(self):
            return 200, {"ok": True}

    class Process:
        def start(self):
            return None

        def is_alive(self):
            return True

        def terminate(self):
            return None

        def join(self, timeout):
            return None

        def kill(self):
            return None

    class Context:
        def Pipe(self, duplex):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda _: Context())

    assert smoke._read_json("http://example.test", "/", None, "GET", 0.1) == (0, None)


def test_unexpected_parent_pipe_close_failure_fails_closed(monkeypatch):
    """Unexpected ordinary close exceptions must be sanitized in the parent supervisor."""
    smoke = _load_smoke_module()

    class Connection:
        def close(self):
            raise RuntimeError("close unavailable")

        def poll(self, timeout):
            return True

        def recv(self):
            return 200, {"ok": True}

    class Process:
        def start(self):
            return None

        def is_alive(self):
            return False

        def join(self, timeout):
            return None

    class Context:
        def Pipe(self, duplex):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(smoke.multiprocessing, "get_context", lambda _: Context())

    assert smoke._read_json("http://example.test", "/", None, "GET", 0.1) == (0, None)
