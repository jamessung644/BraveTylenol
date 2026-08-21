import json

import main


class _Response:
    status = 200

    def __init__(self, content: str = "L2 final answer") -> None:
        self._body = json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self, limit=-1):
        return self._body if limit < 0 else self._body[:limit]


class _RecordingOpener:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, *, timeout):
        del timeout
        self.requests.append(request)
        return _Response()


class _SequentialOpener:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        del timeout
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _RecordingMCP:
    def __init__(self, result: str | None = '{"official":"evidence"}') -> None:
        self.result = result
        self.routes = []

    def __call__(self, route, api_key):
        assert api_key == "lunit_test_key"
        self.routes.append(route)
        return self.result


def _complete(question: str, mcp: _RecordingMCP):
    opener = _RecordingOpener()
    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": question}]},
        "Bearer lunit_test_key",
        opener=opener,
        mcp_provider=mcp,
        environ={},
    )
    outbound = json.loads(opener.requests[0].data)
    return result, outbound


def test_general_medical_and_drug_counseling_stay_one_call_direct():
    questions = (
        "어제부터 두통이 있는데 집에서 어떻게 관리하면 될까요?",
        "타이레놀의 흔한 부작용과 복용 시 주의점을 알려줘.",
        "I have shortness of breath. What should I do now?",
    )

    for question in questions:
        mcp = _RecordingMCP()
        _, outbound = _complete(question, mcp)

        assert mcp.routes == []
        assert outbound["messages"][-1]["content"] == question
        assert len(outbound["messages"]) <= 3


def test_exact_kcd_request_uses_one_official_lookup_then_one_l2_call():
    mcp = _RecordingMCP('{"code":"I10","name":"본태성 고혈압"}')

    result, outbound = _complete("KCD-9 I10의 공식 한글·영문 질병명은?", mcp)

    assert result["choices"][0]["message"]["content"] == "L2 final answer"
    assert len(mcp.routes) == 1
    assert mcp.routes[0].tool_name == "kcd_get_name"
    assert mcp.routes[0].arguments == {
        "code": "I10",
        "lang": "both",
        "revision": "KCD-9",
    }
    system_messages = [
        message["content"] for message in outbound["messages"] if message["role"] == "system"
    ]
    assert any("본태성 고혈압" in content for content in system_messages)
    assert all("must include cite_uid" not in content.casefold() for content in system_messages)


def test_explicit_mfds_label_request_uses_indication_tool_only():
    mcp = _RecordingMCP('{"product":"타이레놀정","warning":"간질환 주의"}')

    _, outbound = _complete(
        "제품명: 타이레놀정. 식약처 허가사항의 효능과 경고를 알려줘.", mcp
    )

    assert [route.tool_name for route in mcp.routes] == [
        "openapi_mfds_get_drug_indication"
    ]
    assert any(
        "간질환 주의" in message["content"]
        for message in outbound["messages"]
        if message["role"] == "system"
    )


def test_mcp_failure_fails_open_to_the_same_direct_l2_request():
    class _FailingMCP:
        def __call__(self, route, api_key):
            del route, api_key
            raise TimeoutError("private MCP detail")

    opener = _RecordingOpener()

    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": "KCD-9 I10의 공식명은?"}]},
        "Bearer lunit_test_key",
        opener=opener,
        mcp_provider=_FailingMCP(),
        environ={},
    )

    assert result["choices"][0]["message"]["content"] == "L2 final answer"
    outbound = json.loads(opener.requests[0].data)
    assert outbound["messages"][-1]["content"] == "KCD-9 I10의 공식명은?"
    assert not any(
        "OFFICIAL REFERENCE" in message["content"] for message in outbound["messages"]
    )


def test_personal_pubmed_request_does_not_export_patient_context_to_mcp():
    mcp = _RecordingMCP()

    _complete(
        "제가 42세이고 주민번호는 900101-1234567인데, 저에게 맞는 논문을 PubMed에서 찾아줘.",
        mcp,
    )

    assert mcp.routes == []


class _MCPHTTPResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]


def test_default_provider_is_used_only_for_a_selected_route(monkeypatch):
    calls = []

    def fake_request(route, api_key):
        calls.append((route, api_key))
        return '{"name":"본태성 고혈압"}'

    monkeypatch.setattr(main, "request_mcp_evidence", fake_request)
    opener = _RecordingOpener()

    main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": "KCD-9 I10의 공식명은?"}]},
        "Bearer lunit_test_key",
        opener=opener,
        environ={},
    )

    assert len(calls) == 1
    assert calls[0][0].tool_name == "kcd_get_name"
    outbound = json.loads(opener.requests[0].data)
    assert any(
        "본태성 고혈압" in message["content"]
        for message in outbound["messages"]
        if message["role"] == "system"
    )


def test_mcp_jsonrpc_request_returns_bounded_structured_content():
    requests = []
    response = _MCPHTTPResponse(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "server-id",
                "result": {
                    "structuredContent": {
                        "code": "I10",
                        "name": "본태성 고혈압",
                    },
                    "isError": False,
                },
            },
            ensure_ascii=False,
        ).encode()
    )

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return response

    route = main.MCPRoute(
        tool_name="kcd_get_name",
        arguments={"code": "I10", "lang": "both", "revision": "KCD-9"},
    )

    evidence = main.request_mcp_evidence(route, "lunit_test_key", opener=opener)

    assert json.loads(evidence) == {"code": "I10", "name": "본태성 고혈압"}
    request, timeout = requests[0]
    assert request.full_url == "https://mcp.hackathon.lunit.io/mcp"
    assert request.get_header("Authorization") == "Bearer lunit_test_key"
    assert timeout <= 5
    payload = json.loads(request.data)
    assert payload["method"] == "tools/call"
    assert payload["params"] == {
        "name": "kcd_get_name",
        "arguments": {"code": "I10", "lang": "both", "revision": "KCD-9"},
    }


def test_mcp_sse_response_uses_text_content_without_protocol_envelope():
    result = {
        "jsonrpc": "2.0",
        "id": "server-id",
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": '{"code":"I10","name":"본태성 고혈압"}',
                }
            ]
        },
    }
    response = _MCPHTTPResponse(
        f"event: message\ndata: {json.dumps(result, ensure_ascii=False)}\n\n".encode()
    )
    route = main.MCPRoute("kcd_get_name", {"code": "I10"})

    evidence = main.request_mcp_evidence(
        route,
        "lunit_test_key",
        opener=lambda request, timeout: response,
    )

    assert json.loads(evidence) == {"code": "I10", "name": "본태성 고혈압"}


def test_invalid_mcp_response_fails_open_without_raw_protocol():
    route = main.MCPRoute("kcd_get_name", {"code": "I10"})
    response = _MCPHTTPResponse(b'{"jsonrpc":"2.0","error":{"code":-1}}')

    evidence = main.request_mcp_evidence(
        route,
        "lunit_test_key",
        opener=lambda request, timeout: response,
    )

    assert evidence is None


def test_explicit_mfds_permission_request_uses_permission_tool():
    mcp = _RecordingMCP()

    _complete("제품명: 타이레놀정. 식약처 품목 허가가 현재 유효한지 알려줘.", mcp)

    assert len(mcp.routes) == 1
    assert mcp.routes[0].tool_name == "openapi_mfds_check_drug_permission"
    assert mcp.routes[0].arguments == {"drug_name": "타이레놀정", "num_rows": 3}


def test_explicit_hira_drug_price_request_uses_price_tool():
    mcp = _RecordingMCP()

    _complete("제품명: 타이레놀정. 현재 심평원 급여 등재 약가를 알려줘.", mcp)

    assert len(mcp.routes) == 1
    assert mcp.routes[0].tool_name == "openapi_hira_get_drug_price"
    assert mcp.routes[0].arguments == {"drug_name": "타이레놀정", "num_rows": 5}


def test_explicit_nonpersonal_pubmed_request_uses_vector_search():
    mcp = _RecordingMCP()
    question = (
        "PubMed에서 만성 신장질환 성인의 목표 혈압에 대한 "
        "최근 메타분석 근거를 찾아줘."
    )

    _complete(question, mcp)

    assert len(mcp.routes) == 1
    assert mcp.routes[0].tool_name == "rag_vector_query"
    assert mcp.routes[0].arguments == {
        "query": question,
        "collection_name": "pubmed_abstracts",
        "top_k": 3,
    }


def test_complex_multirequirement_request_gets_private_plan_then_final_answer():
    opener = _SequentialOpener(
        [
            _Response("1. urgency 2. differential 3. actions 4. red flags"),
            _Response("complete final medical answer"),
        ]
    )
    question = (
        "3일째 발열과 배아픔 및 구토가 있습니다. 가능한 원인을 우선순위로 "
        "설명하고, 집에서 할 수 있는 조치, 병원에 가야 할 시점, 즉시 응급실에 "
        "가야 할 위험 신호를 구분해서 알려주세요."
    )

    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": question}]},
        "Bearer lunit_test_key",
        opener=opener,
        mcp_provider=_RecordingMCP(result=None),
        environ={},
    )

    assert result["choices"][0]["message"]["content"] == "complete final medical answer"
    assert len(opener.requests) == 2
    planning_payload = json.loads(opener.requests[0].data)
    final_payload = json.loads(opener.requests[1].data)
    assert planning_payload["max_tokens"] <= 768
    assert "coverage plan" in planning_payload["messages"][0]["content"].casefold()
    assert any(
        "1. urgency 2. differential 3. actions 4. red flags" in message["content"]
        for message in final_payload["messages"]
        if message["role"] == "system"
    )


def test_failed_private_plan_keeps_the_original_final_generation_path():
    opener = _SequentialOpener(
        [TimeoutError("private plan timeout"), _Response("direct final after plan failure")]
    )
    question = (
        "복용 중인 약과 상호작용, 가능한 부작용, 지금 할 조치, "
        "응급실에 가야 할 위험 신호를 모두 설명해주세요."
    )

    result = main.request_l2_or_fallback(
        {"messages": [{"role": "user", "content": question}]},
        "Bearer lunit_test_key",
        opener=opener,
        mcp_provider=_RecordingMCP(result=None),
        environ={},
    )

    assert result["choices"][0]["message"]["content"] == "direct final after plan failure"
    assert len(opener.requests) == 2
