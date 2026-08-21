class LunitHackathonError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(LunitHackathonError):
    """Required runtime configuration is missing or invalid."""


class UpstreamTimeoutError(LunitHackathonError):
    """The L2 request exceeded its deadline."""


class UpstreamTransportError(LunitHackathonError):
    """The L2 service could not be reached."""


class UpstreamResponseError(LunitHackathonError):
    """The L2 service returned an unsuccessful response."""


class MalformedUpstreamResponseError(UpstreamResponseError):
    """The L2 response did not contain a usable completion."""


class RetrievalError(LunitHackathonError):
    """Evidence retrieval failed without invalidating direct L2 generation."""

    def __init__(self, message: str, *, code: str = "retrieval_failed") -> None:
        super().__init__(message)
        self.code = code
