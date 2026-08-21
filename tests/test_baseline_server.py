import http.client
import inspect
import json
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError

import main
from main import MODEL_ID, create_server


class FakeResponse:
    def __init__(self, payload, *, status=200):
        self.status = status
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]


class RecordingOpener:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class BoundedL2FallbackTest(unittest.TestCase):
    def test_exposes_one_call_l2_fallback_boundary(self):
        self.assertTrue(callable(getattr(main, "request_l2_or_fallback", None)))

    def test_returns_l2_text_from_one_bounded_request(self):
        opener = RecordingOpener(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "맞춤 의료 답변"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                }
            )
        )

        result = main.request_l2_or_fallback(
            {
                "messages": [{"role": "user", "content": "혈압이 높으면 어떻게 해야 하나요?"}],
                "max_tokens": 6_144,
            },
            "Bearer evaluator-secret",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "맞춤 의료 답변")
        self.assertEqual(len(opener.requests), 1)
        self.assertGreater(opener.timeouts[0], 0)
        self.assertLessEqual(opener.timeouts[0], 18)
        outbound = opener.requests[0]
        self.assertEqual(
            outbound.full_url,
            "https://model.hackathon.lunit.io/v1/chat/completions",
        )
        self.assertEqual(outbound.get_header("Authorization"), "Bearer evaluator-secret")
        body = json.loads(outbound.data)
        self.assertEqual(body["model"], "Lunit/L2-preview")
        self.assertEqual(body["max_tokens"], 4_096)
        self.assertEqual(body["messages"][-1]["content"], "혈압이 높으면 어떻게 해야 하나요?")

    def test_l2_failure_returns_baseline_without_retry(self):
        failures = [
            URLError(TimeoutError("stalled")),
            FakeResponse({"error": "unavailable"}, status=503),
            FakeResponse({"choices": [{"message": {"role": "assistant", "content": ""}}]}),
        ]

        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                opener = RecordingOpener(failure)
                result = main.request_l2_or_fallback(
                    {"messages": [{"role": "user", "content": "질문"}]},
                    "Bearer evaluator-secret",
                    opener=opener,
                    environ={},
                )

                self.assertEqual(
                    result["choices"][0]["message"]["content"],
                    main.KOREAN_BASELINE_RESPONSE,
                )
                self.assertEqual(len(opener.requests), 1)

    def test_server_supports_injected_completion_provider(self):
        self.assertIn("completion_provider", inspect.signature(create_server).parameters)

    def test_server_forwards_valid_coeval_request_to_provider(self):
        calls = []

        def provider(payload, authorization):
            calls.append((payload, authorization))
            return main.completion_payload("L2 기반 답변")

        server = create_server(
            "127.0.0.1",
            0,
            completion_provider=provider,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_address[1],
                timeout=1,
            )
            body = json.dumps(
                {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "질문"}],
                    "stream": False,
                }
            ).encode()
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": "Bearer evaluator-secret",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], "L2 기반 답변")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["messages"][-1]["content"], "질문")
        self.assertEqual(calls[0][1], "Bearer evaluator-secret")


class BaselineServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=1)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            self.port,
            timeout=1,
        )
        started = time.perf_counter()
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw_body = response.read()
        elapsed = time.perf_counter() - started
        connection.close()
        payload = json.loads(raw_body) if raw_body else None
        return response.status, payload, elapsed

    def test_health_and_models_contract(self):
        for path in ("/health", "/healthz"):
            status, payload, elapsed = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertEqual(payload, {"status": "ok"})
            self.assertLess(elapsed, 1)

        status, payload, elapsed = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], MODEL_ID)
        self.assertLess(elapsed, 1)

    def test_chat_is_always_immediate_nonempty_korean_completion(self):
        cases = [
            (
                b'{"messages":[{"role":"user","content":"English question"}]}',
                {"Content-Type": "application/json"},
            ),
            (
                b'{"stream":true,"unknown":"accepted"}',
                {
                    "Authorization": "Bearer evaluator-secret",
                    "Content-Type": "application/json",
                },
            ),
            (b"", {}),
            (b"not-json", {"Content-Type": "application/json"}),
        ]

        for body, headers in cases:
            with self.subTest(body=body):
                status, payload, elapsed = self.request(
                    "POST",
                    "/v1/chat/completions",
                    body,
                    headers,
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["object"], "chat.completion")
                self.assertEqual(payload["model"], MODEL_ID)
                self.assertEqual(
                    payload["choices"][0]["message"]["role"],
                    "assistant",
                )
                content = payload["choices"][0]["message"]["content"]
                self.assertTrue(content)
                self.assertRegex(content, re.compile(r"[가-힣]"))
                self.assertEqual(
                    payload["choices"][0]["finish_reason"],
                    "stop",
                )
                self.assertEqual(payload["usage"]["total_tokens"], 0)
                self.assertLess(elapsed, 1)

    def test_concurrent_requests_all_succeed(self):
        def send(index):
            body = json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"English health question {index}",
                        }
                    ]
                }
            ).encode()
            return self.request(
                "POST",
                "/v1/chat/completions",
                body,
                {"Content-Type": "application/json"},
            )

        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(send, range(100)))

        self.assertTrue(all(status == 200 for status, _, _ in results))
        self.assertTrue(
            all(
                re.search(r"[가-힣]", payload["choices"][0]["message"]["content"])
                for _, payload, _ in results
            )
        )
        self.assertLess(max(elapsed for _, _, elapsed in results), 1)


if __name__ == "__main__":
    unittest.main()
