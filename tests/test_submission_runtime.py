from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_submission_image_runs_the_real_l2_harness() -> None:
    """The evaluator image must not replace the model with a fixed reply server."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "main.py").read_text(encoding="utf-8")

    assert "COPY requirements.txt" in dockerfile
    assert "app.py" in dockerfile
    assert "lunit_hackathon" in dockerfile
    assert "uvicorn" in entrypoint
    assert "KOREAN_BASELINE_RESPONSE" not in entrypoint
