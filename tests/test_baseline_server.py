import http.client
import json
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from main import MODEL_ID, create_server


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
