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

class ModelNotAllowedError(Exception):
    pass

class RetryProviderError(Exception):
    pass

class RetryNoProviderError(Exception):
    pass

class VersionNotFoundError(Exception):
    pass

class NestAsyncioError(Exception):
    pass

class ModelNotSupportedError(Exception):
    pass

class MissingRequirementsError(Exception):
    pass

class MissingAiohttpSocksError(MissingRequirementsError):
    pass

class MissingAccessToken(Exception):
    pass