import http.client
import inspect
import json
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from unittest.mock import patch
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


class SlowTrickleResponse:
    def __init__(self):
        self.reads = 0

    def read1(self, limit):
        del limit
        time.sleep(0.02)
        self.reads += 1
        return b"x"


class BoundedL2GatewayTest(unittest.TestCase):
    def test_exposes_one_call_l2_boundary(self):
        self.assertTrue(callable(getattr(main, "request_l2_or_fallback", None)))

    def test_medical_prompt_covers_context_triage_and_safe_management(self):
        required_phrases = (
            "sole author",
            "additional system",
            "patient or caregiver guidance",
            "consultation;",
            "health-data calculation",
            "Do not force",
            "Emergency now",
            "Urgent same-day care",
            "Routine outpatient care",
            "Self-care with monitoring",
            "never escalate by group membership alone",
            "For clinician tasks",
            "Obey exact requested counts",
            "Do not introduce yourself",
            "urgency and disposition explicitly",
            "Do not volunteer a new exact dose",
            "For data tasks",
            "For writing",
            "completeness",
            "accuracy",
            "instruction following",
        )

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, main.MEDICAL_SYSTEM_PROMPT)

    def test_full_multiturn_history_follows_medical_system_prompt(self):
        opener = RecordingOpener(
            FakeResponse({"choices": [{"message": {"content": "follow-up L2 answer"}}]})
        )
        history = [
            {"role": "system", "content": "Evaluator context"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "What should I do now?"},
        ]

        main.request_l2_or_fallback(
            {"messages": history},
            "Bearer lunit_test_key",
            opener=opener,
            environ={},
        )

        body = json.loads(opener.requests[0].data)
        self.assertEqual(
            body["messages"][0],
            {"role": "system", "content": main.MEDICAL_SYSTEM_PROMPT},
        )
        self.assertIn("entire final answer in English", body["messages"][1]["content"])
        self.assertEqual(body["messages"][2:], history)

    def test_latest_user_language_instruction_reinforces_korean_and_english(self):
        cases = [
            ("혈압약을 어떻게 먹나요?", "entire final answer in Korean"),
            ("How should I take this medicine?", "entire final answer in English"),
        ]

        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                instruction = main._latest_user_language_instruction(
                    [{"role": "user", "content": prompt}]
                )
                self.assertIsNotNone(instruction)
                self.assertIn(expected, instruction)

    def test_latest_user_language_instruction_leaves_short_or_mixed_text_ambiguous(self):
        for prompt in ("Why?", "MRI 결과?", "혈압 BP?"):
            with self.subTest(prompt=prompt):
                instruction = main._latest_user_language_instruction(
                    [{"role": "user", "content": prompt}]
                )
                self.assertIsNone(instruction)

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
            "Bearer lunit_request_test",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "맞춤 의료 답변")
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(opener.timeouts, [main.L2_TIMEOUT_SECONDS])
        outbound = opener.requests[0]
        self.assertEqual(
            outbound.full_url,
            "https://model.hackathon.lunit.io/v1/chat/completions",
        )
        self.assertEqual(outbound.get_header("Authorization"), "Bearer lunit_request_test")
        body = json.loads(outbound.data)
        self.assertEqual(body["model"], "Lunit/L2-preview")
        self.assertEqual(body["max_tokens"], 6_144)
        self.assertEqual(body["messages"][-1]["content"], "혈압이 높으면 어떻게 해야 하나요?")

    def test_l2_failure_raises_retryable_error_without_internal_retry(self):
        failures = [
            URLError(TimeoutError("stalled")),
            FakeResponse({"error": "unavailable"}, status=503),
            FakeResponse({"choices": [{"message": {"role": "assistant", "content": ""}}]}),
        ]

        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                opener = RecordingOpener(failure)
                with self.assertRaises(main.L2RequestError) as raised:
                    main.request_l2_or_fallback(
                        {"messages": [{"role": "user", "content": "질문"}]},
                        "Bearer lunit_request_test",
                        opener=opener,
                        environ={},
                    )

                self.assertEqual(raised.exception.status, HTTPStatus.FAILED_DEPENDENCY)
                self.assertEqual(len(opener.requests), 1)

    def test_queue_wait_is_bounded_before_upstream(self):
        with patch.object(main, "_L2_REQUEST_SLOTS") as slots:
            slots.acquire.return_value = False
            with self.assertRaises(main.L2RequestError) as raised:
                main.request_l2_or_fallback(
                    {"messages": [{"role": "user", "content": "질문"}]},
                    "Bearer lunit_request_test",
                    environ={},
                )

        self.assertEqual(raised.exception.code, "queue_timeout")
        self.assertEqual(raised.exception.status, HTTPStatus.FAILED_DEPENDENCY)
        slots.acquire.assert_called_once()
        queue_wait = slots.acquire.call_args.kwargs["timeout"]
        self.assertGreater(queue_wait, 0)
        self.assertLessEqual(queue_wait, main.L2_QUEUE_TIMEOUT_SECONDS)
        slots.release.assert_not_called()

    def test_invalid_messages_raise_bad_request_without_l2_call(self):
        opener = RecordingOpener(FakeResponse({"choices": []}))

        with self.assertRaises(main.L2RequestError) as raised:
            main.request_l2_or_fallback(
                {"messages": []},
                "Bearer lunit_request_test",
                opener=opener,
                environ={},
            )

        self.assertEqual(raised.exception.status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(opener.requests, [])

    def test_environment_lunit_key_wins_over_evaluator_placeholder(self):
        opener = RecordingOpener(FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]}))

        result = main.request_l2_or_fallback(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer evaluator-placeholder",
            opener=opener,
            environ={"LUNIT_FM_API_KEY": "lunit_environment_test"},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "L2 답변")
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer lunit_environment_test",
        )

    def test_valid_lunit_bearer_is_used_without_environment_key(self):
        opener = RecordingOpener(FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]}))

        result = main.request_l2_or_fallback(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer lunit_request_test",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "L2 답변")
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer lunit_request_test",
        )

    def test_non_lunit_bearer_without_environment_uses_embedded_credential(self):
        opener = RecordingOpener(FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]}))

        result = main.request_l2_or_fallback(
            {"messages": [{"role": "user", "content": "질문"}]},
            "Bearer evaluator-placeholder",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "L2 답변")
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            f"Bearer {main.EMBEDDED_LUNIT_API_KEY}",
        )

    def test_malformed_environment_keys_use_embedded_credential(self):
        malformed_keys = [
            "lunit_test\nsecond-line",
            "lunit_" + ("x" * 4_096),
        ]

        for malformed_key in malformed_keys:
            with self.subTest(key_length=len(malformed_key)):
                opener = RecordingOpener(
                    FakeResponse({"choices": [{"message": {"content": "L2 답변"}}]})
                )
                result = main.request_l2_or_fallback(
                    {"messages": [{"role": "user", "content": "질문"}]},
                    None,
                    opener=opener,
                    environ={"LUNIT_FM_API_KEY": malformed_key},
                )

                self.assertEqual(result["choices"][0]["message"]["content"], "L2 답변")
                self.assertEqual(
                    opener.requests[0].get_header("Authorization"),
                    f"Bearer {main.EMBEDDED_LUNIT_API_KEY}",
                )

    def test_extended_coeval_shape_preserves_messages_and_smaller_token_budget(self):
        opener = RecordingOpener(
            FakeResponse({"choices": [{"message": {"content": "확장 형식 답변"}}]})
        )

        result = main.request_l2_or_fallback(
            {
                "messages": [
                    {"role": "developer", "content": "개발자 지침"},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "사용자 질문"}],
                    },
                ],
                "max_completion_tokens": 2_048,
            },
            "Bearer lunit_test-key",
            opener=opener,
            environ={},
        )

        self.assertEqual(result["choices"][0]["message"]["content"], "확장 형식 답변")
        outbound = json.loads(opener.requests[0].data)
        self.assertEqual(outbound["max_tokens"], 2_048)
        self.assertEqual(
            outbound["messages"][-2:],
            [
                {"role": "system", "content": "개발자 지침"},
                {"role": "user", "content": "사용자 질문"},
            ],
        )

    def test_thirty_two_calls_never_exceed_the_l2_concurrency_cap(self):
        lock = threading.Lock()
        active = 0
        maximum = 0
        response = json.dumps({"choices": [{"message": {"content": "동시성 제한 답변"}}]}).encode()

        def fake_post_json(**kwargs):
            nonlocal active, maximum
            del kwargs
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return 200, response

        def send(index):
            return main.request_l2_or_fallback(
                {"messages": [{"role": "user", "content": f"질문 {index}"}]},
                "Bearer lunit_test-key",
                environ={},
            )

        with patch.object(main, "_post_json", side_effect=fake_post_json):
            with ThreadPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(send, range(32)))

        self.assertEqual(maximum, main.MAX_CONCURRENT_L2_REQUESTS)
        self.assertTrue(
            all(
                result["choices"][0]["message"]["content"] == "동시성 제한 답변"
                for result in results
            )
        )

    def test_response_body_trickle_cannot_bypass_the_total_deadline(self):
        response = SlowTrickleResponse()
        started = time.monotonic()

        with self.assertRaises(TimeoutError):
            main._read_bounded_response(response, time.monotonic() + 0.05)

        self.assertLess(time.monotonic() - started, 0.15)
        self.assertLess(response.reads, 10)

    def test_sigterm_requests_clean_process_exit(self):
        with self.assertRaises(KeyboardInterrupt):
            main._handle_termination_signal(15, None)

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

    def test_server_exposes_retryable_provider_error_as_424(self):
        def provider(payload, authorization):
            del payload, authorization
            raise main.L2RequestError(HTTPStatus.FAILED_DEPENDENCY, "timeout")

        server = create_server("127.0.0.1", 0, completion_provider=provider)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1",
                server.server_address[1],
                timeout=1,
            )
            body = json.dumps(
                {"model": MODEL_ID, "messages": [{"role": "user", "content": "질문"}]}
            ).encode()
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

        self.assertEqual(response.status, HTTPStatus.FAILED_DEPENDENCY)
        self.assertEqual(payload["error"]["code"], "timeout")
        self.assertNotIn("choices", payload)


class BaselineServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(
            "127.0.0.1",
            0,
            completion_provider=lambda request_payload, authorization: main.completion_payload(
                "테스트 L2 답변"
            ),
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
