import http.client
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from main import (
    DEFAULT_L2_MODEL,
    L2_MAX_TOKENS,
    L2_TIMEOUT_SECONDS,
    MODEL_ID,
    ConfigurationError,
    L2ResponseError,
    L2TimeoutError,
    completion_payload,
    create_server,
    request_l2_completion,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]


class RecordingOpener:
    def __init__(self, payload=None, status=200, error=None):
        self.payload = payload or {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "L2 final answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        self.status = status
        self.error = error
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.payload, self.status)


class L2ClientTest(unittest.TestCase):
    def test_forwards_complete_history_and_incoming_bearer_once(self):
        opener = RecordingOpener()
        messages = [
            {"role": "system", "content": "Answer carefully."},
            {"role": "user", "content": "I have a fever."},
            {"role": "assistant", "content": "How high is it?"},
            {"role": "user", "content": "39 C"},
        ]

        result = request_l2_completion(
            {"messages": messages, "max_tokens": 6_144},
            "Bearer evaluator-l2-key",
            opener=opener,
            environ={},
        )

        self.assertEqual(len(opener.calls), 1)
        request, timeout = opener.calls[0]
        upstream = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://model.hackathon.lunit.io/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer evaluator-l2-key")
        self.assertEqual(timeout, L2_TIMEOUT_SECONDS)
        self.assertEqual(upstream["model"], DEFAULT_L2_MODEL)
        self.assertEqual(upstream["messages"], messages)
        self.assertEqual(upstream["max_tokens"], L2_MAX_TOKENS)
        self.assertEqual(upstream["temperature"], 0.0)
        self.assertFalse(upstream["stream"])
        self.assertEqual(result["choices"][0]["message"]["content"], "L2 final answer")
        self.assertEqual(result["usage"]["total_tokens"], 15)

    def test_uses_environment_key_when_bearer_is_absent(self):
        opener = RecordingOpener()
        request_l2_completion(
            {"messages": [{"role": "user", "content": "Question"}]},
            None,
            opener=opener,
            environ={"LUNIT_FM_API_KEY": "environment-l2-key"},
        )
        request, _ = opener.calls[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer environment-l2-key")

    def test_incoming_bearer_takes_priority_over_environment_key(self):
        opener = RecordingOpener()
        request_l2_completion(
            {"messages": [{"role": "user", "content": "Question"}]},
            "Bearer evaluator-l2-key",
            opener=opener,
            environ={"LUNIT_FM_API_KEY": "environment-l2-key"},
        )
        request, _ = opener.calls[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer evaluator-l2-key")

    def test_missing_key_fails_without_network(self):
        opener = RecordingOpener()
        with self.assertRaises(ConfigurationError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "Question"}]},
                None,
                opener=opener,
                environ={"LUNIT_FM_API_KEY": "여기에_직접_입력"},
            )
        self.assertEqual(opener.calls, [])

    def test_timeout_is_classified_and_not_retried(self):
        opener = RecordingOpener(error=TimeoutError("secret detail"))
        with self.assertRaises(L2TimeoutError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "Question"}]},
                "Bearer secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.calls), 1)

    def test_blank_l2_content_fails_and_is_not_retried(self):
        opener = RecordingOpener(
            payload={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "  "},
                        "finish_reason": "length",
                    }
                ]
            }
        )
        with self.assertRaises(L2ResponseError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "Question"}]},
                "Bearer secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.calls), 1)

    def test_rejects_unsupported_message_before_network(self):
        opener = RecordingOpener()
        with self.assertRaises(ValueError):
            request_l2_completion(
                {"messages": [{"role": "tool", "content": "not supported"}]},
                "Bearer secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(opener.calls, [])


class L2ProxyServerTest(unittest.TestCase):
    def start_server(self, provider):
        server = create_server("127.0.0.1", 0, completion_provider=provider)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

        self.addCleanup(cleanup)
        return server

    def request(self, server, method, path, body=None, headers=None, timeout=3):
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
            timeout=timeout,
        )
        started = time.perf_counter()
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw_body = response.read()
        elapsed = time.perf_counter() - started
        connection.close()
        payload = json.loads(raw_body) if raw_body else None
        return response.status, payload, elapsed

    def test_health_models_and_l2_chat_contract(self):
        captured = []

        def provider(payload, authorization):
            captured.append((payload, authorization))
            return completion_payload(
                "Answer generated by L2",
                usage={"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
            )

        server = self.start_server(provider)
        for path in ("/health", "/healthz"):
            status, payload, elapsed = self.request(server, "GET", path)
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"status": "ok"})
            self.assertLess(elapsed, 1)

        status, payload, _ = self.request(server, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"][0]["id"], MODEL_ID)

        request_payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Previous reply"},
                {"role": "user", "content": "Follow-up question"},
            ],
            "stream": False,
        }
        status, payload, _ = self.request(
            server,
            "POST",
            "/v1/chat/completions",
            json.dumps(request_payload).encode("utf-8"),
            {
                "Authorization": "Bearer evaluator-l2-key",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["choices"][0]["message"]["content"], "Answer generated by L2")
        self.assertEqual(captured, [(request_payload, "Bearer evaluator-l2-key")])

    def test_invalid_requests_are_400_without_calling_provider(self):
        calls = []

        def provider(payload, authorization):
            calls.append((payload, authorization))
            return completion_payload("should not run")

        server = self.start_server(provider)
        cases = [
            (b"not-json", {"Content-Type": "application/json"}),
            (b"", {}),
            (
                json.dumps(
                    {"messages": [{"role": "user", "content": "Q"}], "stream": True}
                ).encode(),
                {"Content-Type": "application/json"},
            ),
        ]
        for body, headers in cases:
            with self.subTest(body=body):
                status, payload, _ = self.request(
                    server,
                    "POST",
                    "/v1/chat/completions",
                    body,
                    headers,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "invalid_request")
        self.assertEqual(calls, [])

    def test_upstream_errors_are_sanitized(self):
        cases = [
            (ConfigurationError("credential secret"), 503, "l2_not_configured"),
            (L2TimeoutError("timeout secret"), 504, "l2_timeout"),
            (L2ResponseError("upstream secret"), 502, "l2_failure"),
        ]
        for error, expected_status, expected_code in cases:
            with self.subTest(error=type(error).__name__):

                def provider(payload, authorization, error=error):
                    raise error

                server = self.start_server(provider)
                status, payload, _ = self.request(
                    server,
                    "POST",
                    "/v1/chat/completions",
                    json.dumps({"messages": [{"role": "user", "content": "Q"}]}).encode(),
                    {"Content-Type": "application/json"},
                )
                self.assertEqual(status, expected_status)
                self.assertEqual(payload["error"]["code"], expected_code)
                self.assertNotIn("secret", json.dumps(payload))

    def test_sixteen_concurrent_requests_do_not_mix_answers(self):
        def provider(payload, authorization):
            content = payload["messages"][-1]["content"]
            return completion_payload(f"L2:{content}")

        server = self.start_server(provider)

        def send(index):
            body = json.dumps(
                {"messages": [{"role": "user", "content": f"question-{index}"}]}
            ).encode()
            return index, self.request(
                server,
                "POST",
                "/v1/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(send, range(32)))

        for index, (status, payload, elapsed) in results:
            self.assertEqual(status, 200)
            self.assertEqual(payload["choices"][0]["message"]["content"], f"L2:question-{index}")
            self.assertLess(elapsed, 3)


if __name__ == "__main__":
    unittest.main()
