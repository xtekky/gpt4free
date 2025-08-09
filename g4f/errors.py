class G4FError(Exception):
    """Base exception for all g4f-related errors."""
    pass


class ProviderNotFoundError(G4FError):
    """Raised when a provider is not found."""
    pass


class ProviderNotWorkingError(G4FError):
    """Raised when the provider is unavailable or failing."""
    pass


class StreamNotSupportedError(G4FError):
    """Raised when the requested provider does not support streaming."""
    pass


class ModelNotFoundError(G4FError):
    """Raised when a model is not found."""
    pass


class ModelNotAllowedError(G4FError):
    """Raised when a model is not allowed by configuration or policy."""
    pass


class RetryProviderError(G4FError):
    """Raised to retry with another provider."""
    pass


class RetryNoProviderError(G4FError):
    """Raised when there are no providers left to retry."""
    pass


class VersionNotFoundError(G4FError):
    """Raised when the version could not be determined."""
    pass


class MissingRequirementsError(G4FError):
    """Raised when a required dependency is missing."""
    pass


class NestAsyncioError(MissingRequirementsError):
    """Raised when 'nest_asyncio' is missing."""
    pass


class MissingAuthError(G4FError):
    """Raised when authentication details are missing."""
    pass


class PaymentRequiredError(G4FError):
    """Raised when a provider requires payment before access."""
    pass


class NoMediaResponseError(G4FError):
    """Raised when a media request returns no response."""
    pass


class ResponseError(G4FError):
    """Base class for response-related errors."""
    pass


class ResponseStatusError(ResponseError):
    """Raised when an HTTP response returns a non-success status code."""
    pass


class CloudflareError(ResponseStatusError):
    """Raised when a request is blocked by Cloudflare."""
    pass


class RateLimitError(ResponseStatusError):
    """Raised when the provider's rate limit has been exceeded."""
    pass


class NoValidHarFileError(G4FError):
    """Raised when no valid HAR file is found."""
    pass


class TimeoutError(G4FError):
    """Raised for timeout errors during API requests."""
    pass


class ConversationLimitError(G4FError):
    """Raised when a conversation limit is reached on the provider."""
    pass