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
