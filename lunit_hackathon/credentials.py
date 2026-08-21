"""Safe selection of credentials for the Lunit foundation-model endpoint."""

from lunit_hackathon.errors import ConfigurationError

MAX_API_KEY_LENGTH = 4_096
EMBEDDED_LUNIT_API_KEY = "lunit_e7PwpnMvugu5i4_VE74Hfzka3qU8aMytjGwpog3ce90"


def resolve_lunit_api_key(
    authorization: str | None,
    environment_key: str | None,
) -> str:
    """Return the first valid Lunit credential without exposing it."""
    for candidate in (
        environment_key,
        _bearer_token(authorization),
        EMBEDDED_LUNIT_API_KEY,
    ):
        if _valid_lunit_key(candidate):
            return candidate.strip()
    raise ConfigurationError("No valid Lunit API credential is available")


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return None
    return token or None


def _valid_lunit_key(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    key = value.strip()
    return (
        key.startswith("lunit_")
        and len(key) <= MAX_API_KEY_LENGTH
        and not any(character in value for character in "\r\n")
    )
