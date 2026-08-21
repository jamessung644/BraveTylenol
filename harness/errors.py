class ConfigurationError(Exception):
    """The harness cannot be configured for the requested operation."""


class UpstreamTimeoutError(Exception):
    """The upstream model request exceeded its deadline."""


class UpstreamTransportError(Exception):
    """The upstream model could not be reached."""


class UpstreamResponseError(Exception):
    """The upstream model returned an unsuccessful response."""


class MalformedUpstreamResponseError(UpstreamResponseError):
    """The upstream model response was not a usable completion."""


class RetrievalError(Exception):
    """Medical evidence retrieval failed."""

    def __init__(self, message: str, *, code: str = "retrieval_failed") -> None:
        super().__init__(message)
        self.code = code
