"""Verified model-facing prompts and local tools compiled from the design pack."""

from __future__ import annotations

from datetime import date

from lunit_hackathon.artifacts import (
    FINALIZE_RETRIEVAL_TOOL,
    RETRIEVAL_SYSTEM_PROMPT,
    RETRIEVE_RELEVANT_CONTENT_TOOL,
    render_generation_phase_prompt,
)


def generation_system_prompt() -> str:
    """Render the hash-verified tool-decision phase prompt."""

    return render_generation_phase_prompt("tool_decision", current_date=date.today())


def direct_final_system_prompt() -> str:
    """Render the independent no-evidence final-answer prompt."""

    return render_generation_phase_prompt("direct_final", current_date=date.today())


def post_retrieval_final_system_prompt() -> str:
    """Render the independent evidence-grounded final-answer prompt."""

    return render_generation_phase_prompt(
        "post_retrieval_final",
        current_date=date.today(),
    )


def mcp_failure_final_system_prompt() -> str:
    """Render the independent evidence-unavailable final-answer prompt."""

    return render_generation_phase_prompt("mcp_failure_final", current_date=date.today())


def emergency_generation_prompt() -> str:
    """Render the independent emergency final-answer prompt."""

    return render_generation_phase_prompt("emergency_final", current_date=date.today())


def clean_recovery_final_system_prompt() -> str:
    """Render the clean, single-use final-answer recovery prompt."""

    return render_generation_phase_prompt("clean_recovery_final", current_date=date.today())


def safe_completion_final_system_prompt() -> str:
    """Render the fixed-input, single-use minimal safety completion prompt."""

    return render_generation_phase_prompt("safe_completion_final", current_date=date.today())


# Compatibility constants for callers that import prompt text directly. Generation
# renders each phase per request so the trusted date cannot become stale.
MEDICAL_GENERATION_SYSTEM_PROMPT = generation_system_prompt()
DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT = direct_final_system_prompt()
RETRIEVAL_PLANNER_SYSTEM_PROMPT = RETRIEVAL_SYSTEM_PROMPT

__all__ = [
    "clean_recovery_final_system_prompt",
    "DIRECT_MEDICAL_GENERATION_SYSTEM_PROMPT",
    "direct_final_system_prompt",
    "emergency_generation_prompt",
    "FINALIZE_RETRIEVAL_TOOL",
    "MEDICAL_GENERATION_SYSTEM_PROMPT",
    "mcp_failure_final_system_prompt",
    "post_retrieval_final_system_prompt",
    "RETRIEVAL_SYSTEM_PROMPT",
    "RETRIEVAL_PLANNER_SYSTEM_PROMPT",
    "RETRIEVE_RELEVANT_CONTENT_TOOL",
    "safe_completion_final_system_prompt",
    "generation_system_prompt",
]
