import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
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
