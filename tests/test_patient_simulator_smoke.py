import importlib.util
import json
import os
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _load_simulator_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "patient_simulator_smoke.py"
    spec = importlib.util.spec_from_file_location("patient_simulator_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _completion(model: str, role: str, content: str) -> dict[str, object]:
    return {
        "model": model,
        "choices": [
            {
                "message": {"role": role, "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class _DripHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100000")
        self.end_headers()
        self.wfile.write(b"{")
        self.wfile.flush()
        time.sleep(2.0)

    def log_message(self, format, *args):
        return


def _patient_handler(history_lengths: list[int], statuses: list[int]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            history_lengths.append(len(request["messages"]))
            status = statuses.pop(0) if statuses else 200
            if status == 200:
                body = json.dumps(
                    _completion(
                        "patient-simulator-ko",
                        "assistant",
                        f"fixture user turn {len(history_lengths)}",
                    )
                ).encode("utf-8")
            else:
                body = b"{}"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def _harness_handler(history_lengths: list[int], role_sequences: list[tuple[str, ...]]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            history_lengths.append(len(request["messages"]))
            role_sequences.append(tuple(message["role"] for message in request["messages"]))
            completion = _completion("team-chatbot", "assistant", "fixture assistant turn")
            body = json.dumps(completion).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return Handler


def test_conversation_preserves_full_history_and_stops_at_assistant_turn_limit():
    """Dropping prior turns or requesting a fourth assistant response must fail this check."""
    simulator = _load_simulator_module()
    patient_lengths: list[int] = []
    harness_lengths: list[int] = []
    harness_roles: list[tuple[str, ...]] = []
    patient = _server(_patient_handler(patient_lengths, []))
    harness = _server(_harness_handler(harness_lengths, harness_roles))
    try:
        result = simulator.run_conversation(
            harness_url=f"http://127.0.0.1:{harness.server_port}",
            simulator_url=f"http://127.0.0.1:{patient.server_port}",
            api_key="test-key",
            max_turns=3,
        )
    finally:
        patient.shutdown()
        patient.server_close()
        harness.shutdown()
        harness.server_close()

    assert patient_lengths == [0, 2, 4]
    assert harness_lengths == [1, 3, 5]
    assert harness_roles == [
        ("user",),
        ("user", "assistant", "user"),
        ("user", "assistant", "user", "assistant", "user"),
    ]
    assert result.assistant_turns == 3
    assert result.status_failures == 0
    assert result.shape_failures == 0


def test_conversation_retries_one_502_and_restarts_once_after_404():
    """Skipping the bounded 502 retry or retaining history after a 404 must fail this check."""
    simulator = _load_simulator_module()
    patient_lengths: list[int] = []
    harness_lengths: list[int] = []
    harness_roles: list[tuple[str, ...]] = []
    patient = _server(_patient_handler(patient_lengths, [502, 404]))
    harness = _server(_harness_handler(harness_lengths, harness_roles))
    try:
        result = simulator.run_conversation(
            harness_url=f"http://127.0.0.1:{harness.server_port}",
            simulator_url=f"http://127.0.0.1:{patient.server_port}",
            api_key="test-key",
            max_turns=1,
        )
    finally:
        patient.shutdown()
        patient.server_close()
        harness.shutdown()
        harness.server_close()

    assert patient_lengths == [0, 0, 0]
    assert harness_lengths == [1]
    assert result.simulator_retries == 1
    assert result.simulator_restarts == 1
    assert result.assistant_turns == 1


def test_key_file_requires_owner_only_permissions(tmp_path):
    """Accepting a group-readable credential file would expose the runtime Lunit key."""
    simulator = _load_simulator_module()
    key_file = tmp_path / "lunit-key"
    key_file.write_text("test-key", encoding="utf-8")
    key_file.chmod(0o600)

    assert simulator.read_api_key_file(key_file)
    key_file.chmod(0o640)
    with pytest.raises(ValueError, match="credential file is not usable"):
        simulator.read_api_key_file(key_file)


def test_conversation_records_harness_latency_limit_violations():
    """A successful envelope arriving after 165 seconds must not pass the live gate."""
    simulator = _load_simulator_module()
    patient_lengths: list[int] = []
    harness_lengths: list[int] = []
    harness_roles: list[tuple[str, ...]] = []
    patient = _server(_patient_handler(patient_lengths, []))
    harness = _server(_harness_handler(harness_lengths, harness_roles))
    clock = iter([0.0, 0.0, 165.1, 165.1])
    try:
        result = simulator.run_conversation(
            harness_url=f"http://127.0.0.1:{harness.server_port}",
            simulator_url=f"http://127.0.0.1:{patient.server_port}",
            api_key="test-key",
            max_turns=1,
            clock=lambda: next(clock),
        )
    finally:
        patient.shutdown()
        patient.server_close()
        harness.shutdown()
        harness.server_close()

    assert result.harness_latencies == [165.1]
    assert result.harness_latency_failures == 1


def test_post_json_has_an_absolute_deadline_and_sanitizes_lone_surrogates():
    """A drip peer or invalid request text must produce only a failed aggregate result."""
    simulator = _load_simulator_module()
    server = _server(_DripHandler)
    started = time.perf_counter()
    try:
        status, payload = simulator._post_json(
            f"http://127.0.0.1:{server.server_port}",
            {"model": "team-chatbot", "messages": []},
            {},
            timeout_seconds=0.2,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert (status, payload) == (0, None)
    assert time.perf_counter() - started < 1.0
    assert simulator._post_json("http://127.0.0.1:1", {"content": "\ud800"}, {}) == (0, None)


def test_key_file_rejects_symlink_and_oversized_content(tmp_path):
    """Following a symlink or accepting unbounded key material permits key-path attacks."""
    simulator = _load_simulator_module()
    target = tmp_path / "target"
    target.write_text("test-key", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"x" * 65_537)
    oversized.chmod(0o600)

    with pytest.raises(ValueError, match="credential file is not usable"):
        simulator.read_api_key_file(link)
    with pytest.raises(ValueError, match="credential file is not usable"):
        simulator.read_api_key_file(oversized)


def test_key_file_uses_one_descriptor_across_a_path_swap(monkeypatch, tmp_path):
    """Reading the path after validation would accept attacker-replaced key material."""
    simulator = _load_simulator_module()
    key_file = tmp_path / "key"
    replacement = tmp_path / "replacement"
    key_file.write_text("original-key", encoding="utf-8")
    replacement.write_text("replacement-key", encoding="utf-8")
    key_file.chmod(0o600)
    replacement.chmod(0o600)
    original_stat = Path.stat

    def swap_after_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == key_file:
            os.replace(replacement, key_file)
        return result

    monkeypatch.setattr(Path, "stat", swap_after_stat)

    assert simulator.read_api_key_file(key_file) == "original-key"


@pytest.mark.parametrize("attribute", ["wrong_owner", "nonregular"])
def test_key_file_rejects_bad_fstat_metadata(monkeypatch, tmp_path, attribute):
    """Descriptor metadata must reject wrong owners and non-regular files before reading."""
    simulator = _load_simulator_module()
    key_file = tmp_path / "key"
    key_file.write_text("test-key", encoding="utf-8")
    key_file.chmod(0o600)
    uid = os.geteuid() + (1 if attribute == "wrong_owner" else 0)
    mode = stat.S_IFREG | 0o600 if attribute == "wrong_owner" else stat.S_IFIFO | 0o600
    metadata = type("Metadata", (), {"st_uid": uid, "st_mode": mode, "st_size": 8})()
    monkeypatch.setattr(os, "fstat", lambda _: metadata)

    with pytest.raises(ValueError, match="credential file is not usable"):
        simulator.read_api_key_file(key_file)


@pytest.mark.parametrize("failure", [UnicodeError, KeyboardInterrupt])
def test_main_sanitizes_runtime_and_summary_failures(monkeypatch, capsys, tmp_path, failure):
    """Unexpected local serialization/output faults must not print tracebacks or error details."""
    simulator = _load_simulator_module()
    arguments = type(
        "Arguments",
        (),
        {
            "harness_url": "http://harness.test",
            "api_key_file": tmp_path / "key",
            "conversations": 5,
            "max_turns": 3,
        },
    )()
    monkeypatch.setattr(simulator, "parse_args", lambda: arguments)
    monkeypatch.setattr(simulator, "read_api_key_file", lambda _: "test-key")
    monkeypatch.setattr(simulator, "run_smoke", lambda **_: (_ for _ in ()).throw(failure()))

    assert simulator.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("conversations=0")
