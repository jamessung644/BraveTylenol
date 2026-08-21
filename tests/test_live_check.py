import json
from contextlib import asynccontextmanager

import pytest
from pydantic import SecretStr

from lunit_hackathon.artifacts import MCP_TOOL_ALIASES
from lunit_hackathon.config import Settings
from lunit_hackathon.live_check import (
    LiveCanaryError,
    _decode_arguments,
    main,
    run_live_canary,
)
from lunit_hackathon.schemas import L2Completion, MCPCallResult, MCPTool, ToolCall


def _settings(**updates):
    return Settings(_env_file=None).model_copy(
        update={"lunit_fm_api_key": SecretStr("lunit_live_test_key"), **updates}
    )


class FakeMCP:
    def __init__(self):
        self.list_count = 0
        self.calls = []

    @asynccontextmanager
    async def connect(self):
        yield self

    async def list_tools(self):
        self.list_count += 1
        return [
            MCPTool(
                name=alias,
                description="read-only canary tool",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            )
            for alias in MCP_TOOL_ALIASES
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return MCPCallResult(content='{"data_sources":[]}', is_error=False)


class FakeModel:
    def __init__(self):
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["tool_choice"]["function"]["name"]
        arguments = (
            {"query": "Which organizer MCP data sources are available?"}
            if name == "retrieve_relevant_content"
            else {}
        )
        return L2Completion(
            tool_calls=[
                ToolCall(
                    id=f"call-{len(self.calls)}",
                    function={"name": name, "arguments": json.dumps(arguments)},
                )
            ],
            finish_reason="stop",
        )


class FakeModelContext:
    def __init__(self, model):
        self.model = model

    async def __aenter__(self):
        return self.model

    async def __aexit__(self, *args):
        return None


@pytest.mark.parametrize("arguments", ['{"query":NaN}', '{"query":Infinity}'])
def test_live_canary_rejects_nonfinite_model_arguments(arguments):
    with pytest.raises(LiveCanaryError) as caught:
        _decode_arguments(arguments)

    assert caught.value.code == "model_tool_arguments_invalid"


@pytest.mark.parametrize(
    "argv",
    [[], ["--live"], ["--confirm-lunit-network"]],
)
def test_cli_without_both_opt_in_flags_performs_no_setup(argv, monkeypatch, capsys):
    def forbidden_settings():
        raise AssertionError("Settings must not be constructed before explicit opt-in")

    monkeypatch.setattr("lunit_hackathon.live_check.Settings", forbidden_settings)

    assert main(argv) == 2
    assert "No network request was made" in capsys.readouterr().out


async def test_live_canary_refuses_nonofficial_target_before_factories_run():
    def forbidden_factory(settings):
        del settings
        raise AssertionError("network-capable factories must not be constructed")

    settings = _settings(lunit_fm_api_url="https://example.test")

    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            settings,
            model_factory=forbidden_factory,
            mcp_factory=forbidden_factory,
        )

    assert caught.value.code == "model_endpoint_mismatch"


async def test_live_canary_checks_exact_registry_strict_tools_and_read_only_mcp_call():
    fake_mcp = FakeMCP()
    fake_model = FakeModel()

    report = await run_live_canary(
        _settings(),
        commit_sha="a" * 40,
        image_digest=f"sha256:{'b' * 64}",
        model_factory=lambda settings: FakeModelContext(fake_model),
        mcp_factory=lambda settings: fake_mcp,
    )

    assert report["status"] == "passed"
    assert report["release"]["commit_sha"] == "a" * 40
    assert len(report["release"]["live_binding_registry_sha256"]) == 64
    phase_hashes = report["release"]["generation_phase_prompt_sha256"]
    assert set(phase_hashes) == {
        "tool_decision",
        "direct_final",
        "post_retrieval_final",
        "mcp_failure_final",
        "emergency_final",
        "clean_recovery_final",
        "safe_completion_final",
    }
    assert all(len(digest) == 64 for digest in phase_hashes.values())
    assert report["release"]["generation_prompt_sha256"] == (
        report["release"]["generation_phase_registry_sha256"]
    )
    discovery = report["checks"]["mcp_exact_discovery_and_strict_binding"]
    assert discovery == {
        "status": "passed",
        "approved_alias_count": 21,
        "bound_tool_count": 21,
    }
    wrapper_check = report["checks"]["retrieval_mcp_strict_wrapper"]
    assert wrapper_check["executed"] is True
    assert wrapper_check["result_content_chars"] == len('{"data_sources":[]}')
    assert fake_mcp.list_count == 1
    assert fake_mcp.calls == [("rag_get_all_data_sources", {})]
    assert [
        call["tool_choice"]["function"]["name"] for call in fake_model.calls
    ] == ["retrieve_relevant_content", "rag_get_all_data_sources"]
    retrieval_registry = fake_model.calls[1]["tools"]
    assert len(retrieval_registry) == 22
    assert {tool["function"]["name"] for tool in retrieval_registry} == {
        *MCP_TOOL_ALIASES,
        "finalize_retrieval",
    }


async def test_live_canary_rejects_incomplete_discovery():
    fake_mcp = FakeMCP()

    async def incomplete_tools():
        return (await FakeMCP().list_tools())[:-1]

    fake_mcp.list_tools = incomplete_tools

    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            _settings(),
            model_factory=lambda settings: FakeModelContext(FakeModel()),
            mcp_factory=lambda settings: fake_mcp,
        )

    assert caught.value.code == "approved_registry_mismatch"


async def test_live_canary_rejects_mixed_text_and_forced_tool_call():
    class MixedModel(FakeModel):
        async def complete(self, **kwargs):
            completion = await super().complete(**kwargs)
            completion.content = "I also answered in text."
            return completion

    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            _settings(),
            model_factory=lambda settings: FakeModelContext(MixedModel()),
            mcp_factory=lambda settings: FakeMCP(),
        )

    assert caught.value.code == "model_strict_tool_rejected"


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "function_call"])
async def test_live_canary_accepts_supported_tool_call_finish_reasons(finish_reason):
    class FinishReasonModel(FakeModel):
        async def complete(self, **kwargs):
            completion = await super().complete(**kwargs)
            completion.finish_reason = finish_reason
            return completion

    report = await run_live_canary(
        _settings(),
        model_factory=lambda settings: FakeModelContext(FinishReasonModel()),
        mcp_factory=lambda settings: FakeMCP(),
    )

    assert report["status"] == "passed"


@pytest.mark.parametrize("finish_reason", [None, "length", "content_filter"])
async def test_live_canary_rejects_invalid_tool_finish_reason_before_mcp(finish_reason):
    class InvalidFinishReasonModel(FakeModel):
        async def complete(self, **kwargs):
            completion = await super().complete(**kwargs)
            completion.finish_reason = finish_reason
            return completion

    fake_mcp = FakeMCP()
    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            _settings(),
            model_factory=lambda settings: FakeModelContext(InvalidFinishReasonModel()),
            mcp_factory=lambda settings: fake_mcp,
        )

    assert caught.value.code == "model_strict_tool_rejected"
    assert fake_mcp.calls == []


async def test_live_canary_rejects_error_result_from_read_only_mcp_call():
    fake_mcp = FakeMCP()

    async def error_result(name, arguments):
        fake_mcp.calls.append((name, arguments))
        return MCPCallResult(content='{"error":true}', is_error=True)

    fake_mcp.call_tool = error_result

    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            _settings(),
            model_factory=lambda settings: FakeModelContext(FakeModel()),
            mcp_factory=lambda settings: fake_mcp,
        )

    assert caught.value.code == "mcp_canary_call_failed"
    assert fake_mcp.calls == [("rag_get_all_data_sources", {})]


async def test_live_canary_rejects_unbounded_read_only_mcp_result():
    fake_mcp = FakeMCP()

    async def oversized_result(name, arguments):
        fake_mcp.calls.append((name, arguments))
        return MCPCallResult(content="x" * 1_001, is_error=False)

    fake_mcp.call_tool = oversized_result

    with pytest.raises(LiveCanaryError) as caught:
        await run_live_canary(
            _settings(max_tool_result_chars=1_000),
            model_factory=lambda settings: FakeModelContext(FakeModel()),
            mcp_factory=lambda settings: fake_mcp,
        )

    assert caught.value.code == "mcp_canary_result_invalid"
