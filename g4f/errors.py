class ProviderNotFoundError(Exception):
    pass

class ProviderNotWorkingError(Exception):
    pass

class StreamNotSupportedError(Exception):
    pass

class AuthenticationRequiredError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass

class ModelNotAllowed(Exception):
    pass

class RetryProviderError(Exception):
    pass

class RetryNoProviderError(Exception):
    pass

class VersionNotFoundError(Exception):
    pass