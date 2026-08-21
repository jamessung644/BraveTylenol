import hashlib
import importlib.util
import json
import multiprocessing
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_paired_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paired_quality_check.py"
    spec = importlib.util.spec_from_file_location("paired_quality_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _completion(content: str) -> dict[str, object]:
    return {
        "model": "team-chatbot",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _handler(answer: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps(_completion(answer)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def _server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class _DripHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    interval_seconds = 0.02
    bytes_sent = 0
    sent_multiple_bytes = threading.Event()

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100000")
        self.end_headers()
        for _ in range(500):
            try:
                self.wfile.write(b" ")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            type(self).bytes_sent += 1
            if type(self).bytes_sent >= 3:
                type(self).sent_multiple_bytes.set()
            time.sleep(type(self).interval_seconds)

    def log_message(self, format, *args):
        return


def test_cli_writes_only_paired_answers_to_jsonl_and_prints_aggregate_metrics(tmp_path):
    """Printing individual answers or omitting a fixed scenario must fail this check."""
    direct = _server(_handler("direct fixture response"))
    score = _server(_handler("score fixture response"))
    output = tmp_path / "paired.jsonl"
    script = Path(__file__).resolve().parents[1] / "scripts" / "paired_quality_check.py"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--direct-url",
                f"http://127.0.0.1:{direct.server_port}",
                "--score-url",
                f"http://127.0.0.1:{score.server_port}",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    finally:
        direct.shutdown()
        direct.server_close()
        score.shutdown()
        score.server_close()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result.returncode == 0
    assert len(records) == 7
    assert all(
        set(record) == {"scenario", "domain", "direct_answer", "score_answer"}
        and isinstance(record["direct_answer"], str)
        and bool(record["direct_answer"].strip())
        and isinstance(record["score_answer"], str)
        and bool(record["score_answer"].strip())
        for record in records
    )
    assert "paired_requests=7 direct_success=7 score_success=7" in result.stdout
    assert "completion_rate=" in result.stdout
    assert "fallback_rate=" in result.stdout
    assert "latency_seconds=" in result.stdout
    assert "domain_coverage=" in result.stdout
    assert "direct_answer" not in result.stdout
    assert "score_answer" not in result.stdout
    assert hashlib.sha256(result.stdout.encode("utf-8")).digest()
    assert result.stderr == ""


def test_output_path_must_be_outside_the_repository_unless_ignored(tmp_path):
    """Allowing an ordinary repository path would make paired answers commit candidates."""
    paired = _load_paired_module()
    repository = Path(__file__).resolve().parents[1]

    assert paired.output_path_allowed(tmp_path / "paired.jsonl", repository)
    assert not paired.output_path_allowed(repository / "paired.jsonl", repository)


def test_post_json_enforces_an_absolute_deadline_against_a_drip_peer(capfd):
    """Continuous bytes below the socket timeout cannot extend the hard deadline."""
    paired = _load_paired_module()
    _DripHandler.bytes_sent = 0
    _DripHandler.sent_multiple_bytes.clear()
    prior_child_pids = {child.pid for child in multiprocessing.active_children()}
    server = _server(_DripHandler)
    started = time.perf_counter()
    try:
        status, payload = paired._post_json(
            f"http://127.0.0.1:{server.server_port}",
            {"model": "team-chatbot", "messages": []},
            timeout_seconds=0.2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert (status, payload) == (0, None)
    assert time.perf_counter() - started < 1.0
    assert _DripHandler.sent_multiple_bytes.is_set()
    assert _DripHandler.bytes_sent >= 3
    assert _DripHandler.interval_seconds < 0.2
    assert all(child.pid in prior_child_pids for child in multiprocessing.active_children())
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_serialization_and_jsonl_encoding_fail_closed_without_output(tmp_path):
    """A lone surrogate must become an aggregate failure instead of a traceback or partial JSONL."""
    paired = _load_paired_module()
    output = tmp_path / "paired.jsonl"

    assert paired._post_json("http://127.0.0.1:1", {"content": "\ud800"}) == (0, None)
    assert not paired.write_jsonl(
        output,
        [
            {
                "scenario": "fixture",
                "domain": "fixture",
                "direct_answer": "\ud800",
                "score_answer": None,
            }
        ],
    )
    assert not output.exists()


def test_main_sanitizes_jsonl_write_errors(monkeypatch, capsys, tmp_path):
    """Letting a local output encoding failure reach stderr would leak unsafe diagnostics."""
    paired = _load_paired_module()
    arguments = type(
        "Arguments",
        (),
        {
            "direct_url": "http://direct.test",
            "score_url": "http://score.test",
            "output": tmp_path / "x",
        },
    )()
    monkeypatch.setattr(paired, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        paired,
        "run_check",
        lambda *_: (
            [
                {
                    "scenario": "fixture",
                    "domain": "fixture",
                    "direct_answer": "\ud800",
                    "score_answer": None,
                }
            ],
            {
                "requests": 1,
                "direct": paired._aggregate([]),
                "score": paired._aggregate([]),
                "coverage": {},
            },
        ),
    )

    assert paired.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("paired_requests=0")


def test_main_sanitizes_interrupts_without_stderr(monkeypatch, capsys, tmp_path):
    """An operator interrupt during a request must not emit a traceback or request details."""
    paired = _load_paired_module()
    arguments = type(
        "Arguments",
        (),
        {
            "direct_url": "http://direct.test",
            "score_url": "http://score.test",
            "output": tmp_path / "x",
        },
    )()
    monkeypatch.setattr(paired, "parse_args", lambda: arguments)
    monkeypatch.setattr(paired, "run_check", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert paired.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("paired_requests=0")
