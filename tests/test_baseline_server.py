import http.client
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError

from main import (
    DIRECT_MEDICAL_SYSTEM_PROMPT,
    MODEL_ID,
    ClientRequestError,
    ConfigurationError,
    L2ResponseError,
    L2TimeoutError,
    completion_payload,
    create_server,
    request_l2_completion,
)


class FakeResponse:
    def __init__(self, payload, *, status=200):
        self.status = status
        self.body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]


class SequenceOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, *, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def l2_response(content="L2가 생성한 의료 답변", *, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
    }


class MinimalL2ClientTest(unittest.TestCase):
    def test_forwards_coeval_bearer_multiturn_and_full_token_budget(self):
        opener = SequenceOpener(FakeResponse(l2_response(finish_reason="length")))
        request_payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": "Evaluator instruction"}],
                },
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "후속 질문"},
            ],
            "max_tokens": 6_144,
            "stream": False,
            "ignored": "allowed",
        }

        result = request_l2_completion(
            request_payload,
            "Bearer evaluator-secret",
            opener=opener,
            environ={
                "LUNIT_FM_API_KEY": "stale-environment-secret",
            },
        )

        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "L2가 생성한 의료 답변",
        )
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        self.assertEqual(result["usage"]["total_tokens"], 18)
        self.assertEqual(len(opener.requests), 1)
        outbound = opener.requests[0]
        self.assertEqual(
            outbound.full_url,
            "https://model.hackathon.lunit.io/v1/chat/completions",
        )
        self.assertEqual(outbound.get_header("Authorization"), "Bearer evaluator-secret")
        body = json.loads(outbound.data)
        self.assertEqual(body["model"], "Lunit/L2-preview")
        self.assertEqual(body["max_tokens"], 6_144)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["temperature"], 0.0)
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)
        self.assertEqual(body["messages"][0]["content"], DIRECT_MEDICAL_SYSTEM_PROMPT)
        self.assertEqual(
            body["messages"][1:],
            [
                {"role": "system", "content": "Evaluator instruction"},
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "후속 질문"},
            ],
        )

    def test_uses_environment_key_when_request_has_no_bearer(self):
        opener = SequenceOpener(FakeResponse(l2_response()))

        request_l2_completion(
            {"messages": [{"role": "user", "content": "질문"}]},
            None,
            opener=opener,
            environ={"LUNIT_FM_API_KEY": "environment-secret"},
        )

        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer environment-secret",
        )

    def test_rejects_missing_or_placeholder_key_without_network(self):
        for environment in ({}, {"LUNIT_FM_API_KEY": "여기에_직접_입력"}):
            with self.subTest(environment=environment):
                opener = SequenceOpener(FakeResponse(l2_response()))
                with self.assertRaises(ConfigurationError):
                    request_l2_completion(
                        {"messages": [{"role": "user", "content": "질문"}]},
                        None,
                        opener=opener,
                        environ=environment,
                    )
                self.assertEqual(opener.requests, [])

    def test_token_budget_keeps_coeval_default_and_honors_smaller_request(self):
        cases = [
            ({}, 6_144),
            ({"max_tokens": 4_000}, 4_000),
            ({"max_tokens": 6_144, "max_completion_tokens": 3_000}, 3_000),
        ]
        for extra, expected in cases:
            with self.subTest(extra=extra):
                opener = SequenceOpener(FakeResponse(l2_response()))
                request_l2_completion(
                    {
                        "messages": [{"role": "user", "content": "질문"}],
                        **extra,
                    },
                    "Bearer evaluator-secret",
                    opener=opener,
                    environ={},
                )
                self.assertEqual(json.loads(opener.requests[0].data)["max_tokens"], expected)

    def test_polluted_environment_cannot_restore_the_broken_1024_cap(self):
        opener = SequenceOpener(FakeResponse(l2_response()))

        request_l2_completion(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer evaluator-secret",
            opener=opener,
            environ={"MAX_COMPLETION_TOKENS": "1024"},
        )

        self.assertEqual(json.loads(opener.requests[0].data)["max_tokens"], 6_144)

    def test_polluted_integration_environment_cannot_change_official_l2_call(self):
        opener = SequenceOpener(FakeResponse(l2_response()))

        request_l2_completion(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer evaluator-secret",
            opener=opener,
            environ={
                "LUNIT_FM_API_URL": "https://stale.invalid",
                "LUNIT_FM_MODEL": "stale-model",
                "LUNIT_REASONING_EFFORT": "high",
                "REQUEST_TIMEOUT_SECONDS": "3",
                "UPSTREAM_TIMEOUT_SECONDS": "3",
            },
        )

        outbound = opener.requests[0]
        body = json.loads(outbound.data)
        self.assertEqual(
            outbound.full_url,
            "https://model.hackathon.lunit.io/v1/chat/completions",
        )
        self.assertEqual(body["model"], "Lunit/L2-preview")
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertGreater(opener.timeouts[0], 160)

    def test_blank_completion_gets_one_plain_text_l2_recovery(self):
        opener = SequenceOpener(
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": None, "reasoning": "unfinished"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 1_024,
                        "total_tokens": 1_044,
                    },
                }
            ),
            FakeResponse(l2_response("복구된 L2 최종 답변")),
        )

        result = request_l2_completion(
            {"messages": [{"role": "user", "content": "질문"}], "max_tokens": 6_144},
            "Bearer evaluator-secret",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "복구된 L2 최종 답변")
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(json.loads(opener.requests[0].data)["max_tokens"], 6_144)
        recovery_body = json.loads(opener.requests[1].data)
        self.assertIn("complete user-facing answer", recovery_body["messages"][-1]["content"])
        self.assertEqual(recovery_body["max_tokens"], 6_144)

    def test_empty_choices_gets_one_plain_text_l2_recovery(self):
        opener = SequenceOpener(
            FakeResponse({"choices": []}),
            FakeResponse(l2_response("빈 choices 뒤 복구된 L2 답변")),
        )

        result = request_l2_completion(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer evaluator-secret",
            opener=opener,
            environ={},
        )

        self.assertEqual(
            result["choices"][0]["message"]["content"],
            "빈 choices 뒤 복구된 L2 답변",
        )
        self.assertEqual(len(opener.requests), 2)

    def test_transient_status_is_not_amplified_inside_driver(self):
        opener = SequenceOpener(FakeResponse({"error": "overloaded"}, status=503))

        with self.assertRaises(L2ResponseError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 1)

    def test_real_urllib_transient_http_error_is_not_retried(self):
        opener = SequenceOpener(
            HTTPError("https://model.example.test", 503, "overloaded", {}, None)
        )

        with self.assertRaises(L2ResponseError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 1)

    def test_nonretryable_http_error_fails_after_one_attempt(self):
        opener = SequenceOpener(
            HTTPError("https://model.example.test", 401, "unauthorized", {}, None)
        )

        with self.assertRaises(L2ResponseError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 1)

    def test_two_blank_completions_fail_instead_of_returning_python_authored_text(self):
        blank = {"choices": [{"message": {"content": " "}, "finish_reason": "length"}]}
        opener = SequenceOpener(FakeResponse(blank), FakeResponse(blank))

        with self.assertRaises(L2ResponseError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 2)

    def test_wrapped_url_timeouts_are_reported_as_l2_timeout(self):
        opener = SequenceOpener(URLError(TimeoutError("first")))

        with self.assertRaises(L2TimeoutError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 1)

    def test_timeout_is_not_retried_inside_driver(self):
        opener = SequenceOpener(TimeoutError("first"))

        with self.assertRaises(L2TimeoutError):
            request_l2_completion(
                {"messages": [{"role": "user", "content": "질문"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(len(opener.requests), 1)

    def test_rejects_unsupported_message_content_before_network(self):
        opener = SequenceOpener(FakeResponse(l2_response()))

        with self.assertRaises(ClientRequestError):
            request_l2_completion(
                {"messages": [{"role": "tool", "content": "not accepted"}]},
                "Bearer evaluator-secret",
                opener=opener,
                environ={},
            )
        self.assertEqual(opener.requests, [])


class RecordingProvider:
    def __init__(self):
        self.calls = []
        self.error = None
        self.lock = threading.Lock()
        self.delay = 0.0
        self.active = 0
        self.max_active = 0

    def reset(self):
        with self.lock:
            self.calls.clear()
        self.error = None
        self.delay = 0.0
        self.active = 0
        self.max_active = 0

    def __call__(self, payload, authorization):
        if self.error is not None:
            raise self.error
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ClientRequestError("messages are required")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((payload, authorization))
        try:
            if self.delay:
                time.sleep(self.delay)
            final_content = messages[-1].get("content", "")
            return completion_payload(
                f"L2 응답: {final_content}",
                usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            )
        finally:
            with self.lock:
                self.active -= 1


class MinimalL2ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = RecordingProvider()
        cls.server = create_server(
            "127.0.0.1",
            0,
            completion_provider=cls.provider,
        )
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

    def setUp(self):
        self.provider.reset()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
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

    def test_coeval_multiturn_request_returns_l2_completion(self):
        request_body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": "Careful medical assistant"},
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "후속 질문"},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 6_144,
            "stream": False,
        }

        status, payload, elapsed = self.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(request_body).encode(),
            {
                "Authorization": "Bearer evaluator-secret",
                "Content-Type": "application/json",
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["choices"][0]["message"]["content"], "L2 응답: 후속 질문")
        self.assertEqual(payload["usage"]["total_tokens"], 6)
        self.assertEqual(self.provider.calls, [(request_body, "Bearer evaluator-secret")])
        self.assertLess(elapsed, 1)

    def test_invalid_requests_never_reach_l2(self):
        cases = [
            (b"", {}),
            (b"not-json", {"Content-Type": "application/json"}),
            (
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "질문"}],
                        "stream": True,
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            ),
            (json.dumps({"messages": []}).encode(), {"Content-Type": "application/json"}),
        ]

        for body, headers in cases:
            with self.subTest(body=body):
                status, payload, elapsed = self.request(
                    "POST",
                    "/v1/chat/completions",
                    body,
                    headers,
                )
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "invalid_request")
                self.assertLess(elapsed, 1)
        self.assertEqual(self.provider.calls, [])

    def test_expected_l2_failures_have_sanitized_statuses(self):
        cases = [
            (ConfigurationError("secret detail"), 503, "l2_not_configured"),
            (L2TimeoutError("secret detail"), 504, "l2_timeout"),
            (L2ResponseError("secret detail"), 502, "l2_failure"),
        ]
        body = json.dumps({"messages": [{"role": "user", "content": "질문"}]}).encode()

        for error, expected_status, expected_code in cases:
            with self.subTest(error=type(error).__name__):
                self.provider.error = error
                status, payload, elapsed = self.request(
                    "POST",
                    "/v1/chat/completions",
                    body,
                    {"Content-Type": "application/json"},
                )
                self.assertEqual(status, expected_status)
                self.assertEqual(payload["error"]["code"], expected_code)
                self.assertNotIn("secret detail", json.dumps(payload))
                self.assertLess(elapsed, 1)

    def test_sixteen_parallel_requests_are_isolated(self):
        self.provider.delay = 0.02

        def send(index):
            body = json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"병렬 질문 {index}"},
                    ],
                    "max_tokens": 6_144,
                }
            ).encode()
            return self.request(
                "POST",
                "/v1/chat/completions",
                body,
                {
                    "Authorization": f"Bearer evaluator-secret-{index}",
                    "Content-Type": "application/json",
                },
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(send, range(16)))

        self.assertTrue(all(status == 200 for status, _, _ in results))
        self.assertEqual(
            len({payload["id"] for _, payload, _ in results}),
            16,
        )
        self.assertEqual(
            {
                payload["choices"][0]["message"]["content"]
                for _, payload, _ in results
            },
            {f"L2 응답: 병렬 질문 {index}" for index in range(16)},
        )
        self.assertEqual(len(self.provider.calls), 16)
        self.assertGreaterEqual(self.provider.max_active, 2)
        self.assertLess(max(elapsed for _, _, elapsed in results), 2)


if __name__ == "__main__":
    unittest.main()
