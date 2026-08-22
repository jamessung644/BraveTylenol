"""Fail-closed outbound network policy for the isolated submission runtime.

The evaluated container may talk only to the two organizer-provided Lunit
services.  Keep these values independent of environment configuration so a
misconfigured override cannot widen the egress boundary.
"""

from lunit_hackathon.errors import ConfigurationError

OFFICIAL_MODEL_BASE_URL = "https://model.hackathon.lunit.io"
OFFICIAL_MODEL_CHAT_COMPLETIONS_URL = (
    "https://model.hackathon.lunit.io/v1/chat/completions"
)
OFFICIAL_MCP_URL = "https://mcp.hackathon.lunit.io/mcp"


def require_official_model_endpoint(url: str) -> None:
    """Reject any Model target other than the exact organizer endpoint."""

    if url != OFFICIAL_MODEL_CHAT_COMPLETIONS_URL:
        raise ConfigurationError("Model endpoint is outside the Lunit runtime allowlist")


def require_official_mcp_endpoint(url: str) -> None:
    """Reject any MCP target other than the exact organizer endpoint."""

    if url != OFFICIAL_MCP_URL:
        raise ConfigurationError("MCP endpoint is outside the Lunit runtime allowlist")
