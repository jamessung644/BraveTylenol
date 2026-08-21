import importlib.util
import sys
from pathlib import Path


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "concurrent_smoke.py"
    spec = importlib.util.spec_from_file_location("concurrent_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_completion_accepts_a_valid_fallback_envelope():
    """Dropping valid safety fallbacks would turn expected upstream failures into smoke failures."""
    smoke = _load_smoke_module()
    payload = {
        "model": "team-chatbot",
        "choices": [
            {
                "message": {"role": "assistant", "content": smoke.SAFETY_FALLBACK},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    assert smoke.validate_completion(payload, latency_seconds=1.5) == (True, True)


def test_validate_completion_rejects_bad_envelopes_and_slow_responses():
    """Removing a response-shape or deadline check must fail this test."""
    smoke = _load_smoke_module()
    valid = {
        "model": "team-chatbot",
        "choices": [
            {"message": {"role": "assistant", "content": "안내입니다."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    assert smoke.validate_completion({**valid, "model": "other"}, 1.0) == (False, False)
    assert smoke.validate_completion({**valid, "usage": {"prompt_tokens": -1}}, 1.0) == (
        False,
        False,
    )
    assert smoke.validate_completion(valid, latency_seconds=165.01) == (False, False)


def test_aggregate_counts_structural_fallbacks_without_exposing_response_text():
    """A regression that loses fallback or HTTP count accounting must fail this test."""
    smoke = _load_smoke_module()
    results = [
        smoke.RequestResult(status=200, latency_seconds=1.0, success=True, fallback=False),
        smoke.RequestResult(status=200, latency_seconds=3.0, success=True, fallback=True),
        smoke.RequestResult(status=503, latency_seconds=2.0, success=False, fallback=False),
    ]

    assert smoke.aggregate(results) == {
        "requests": 3,
        "success": 2,
        "fallback": 1,
        "failure": 1,
        "status_counts": {200: 2, 503: 1},
        "latencies": [1.0, 3.0, 2.0],
    }


def test_exit_code_is_nonzero_when_any_request_failed():
    """Returning success despite an invalid completion must fail this test."""
    smoke = _load_smoke_module()
    summary = {
        "requests": 2,
        "success": 1,
        "fallback": 0,
        "failure": 1,
        "status_counts": {200: 1, 500: 1},
        "latencies": [0.1, 0.2],
    }

    assert smoke.exit_code(summary) == 1
