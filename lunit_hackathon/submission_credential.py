"""Load the explicitly packaged submission credential without exposing it."""

from importlib import import_module


def embedded_main_api_key(enabled: bool) -> str | None:
    """Return ``main.EMBEDDED_LUNIT_API_KEY`` only for packaged submissions.

    Importing ``main`` is deliberately gated by runtime configuration.  Local
    development and library imports therefore never consult the submission
    file implicitly.  Validation and logging remain the caller's responsibility;
    this helper never renders the secret.
    """

    if not enabled:
        return None
    try:
        module = import_module("main")
    except ModuleNotFoundError:
        return None
    value = getattr(module, "EMBEDDED_LUNIT_API_KEY", None)
    return value if isinstance(value, str) else None
