from g4f.providers.base_provider import (
    AbstractProvider,
    AsyncProvider,
    AsyncGeneratorProvider,
    AsyncAuthedProvider,
)
from g4f.providers.response import AuthResult, ImageResponse
from g4f.errors import MissingAuthError


class ProviderMock(AbstractProvider):
    working = True
    use_stream_timeout = False

    @classmethod
    def create_completion(cls, model, messages, stream, **kwargs):
        yield "Mock"


class AsyncProviderMock(AsyncProvider):
    working = True
    use_stream_timeout = False

    @classmethod
    async def create_async(cls, model, messages, **kwargs):
        return "Mock"


class AsyncGeneratorProviderMock(AsyncGeneratorProvider):
    working = True
    use_stream_timeout = False

    @classmethod
    async def create_async_generator(cls, model, messages, stream, **kwargs):
        yield "Mock"


class ModelProviderMock(AbstractProvider):
    working = True
    use_stream_timeout = False  # Added to fix unittest error

    @classmethod
    def create_completion(cls, model, messages, stream, **kwargs):
        yield model


class YieldProviderMock(AsyncGeneratorProvider):
    working = True
    use_stream_timeout = False

    @classmethod
    async def create_async_generator(cls, model, messages, stream, **kwargs):
        for message in messages:
            yield message["content"]


class YieldImageResponseProviderMock(AsyncGeneratorProvider):
    working = True
    use_stream_timeout = False

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, prompt: str, **kwargs
    ):
        yield ImageResponse(prompt, "")


class MissingAuthProviderMock(AbstractProvider):
    use_stream_timeout = False
    working = True

    @classmethod
    def create_completion(cls, model, messages, stream, **kwargs):
        raise MissingAuthError(cls.__name__)
        yield cls.__name__


class RaiseExceptionProviderMock(AbstractProvider):
    working = True

    @classmethod
    def create_completion(cls, model, messages, stream, **kwargs):
        raise RuntimeError(cls.__name__)
        yield cls.__name__


class AsyncRaiseExceptionProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(cls, model, messages, stream, **kwargs):
        raise RuntimeError(cls.__name__)
        yield cls.__name__


class YieldNoneProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(cls, model, messages, stream, **kwargs):
        yield None


class RetryAuthedProviderMock(AsyncAuthedProvider):
    """Authed provider that fails once with stale auth, then succeeds.

    Mimics the OpenaiChat behaviour where a cached access token is rejected
    by the server (MissingAuthError) and has to be re-logged in. The first
    create_authed call raises; on_auth_async yields a fresh AuthResult and a
    second create_authed call succeeds.
    """

    working = True
    parent = "RetryAuthedProviderMock"

    _api_key = None
    _headers = None
    _cookies = None
    _expires = None

    @classmethod
    def reset_auth(cls):
        cls._api_key = None
        cls._headers = None
        cls._cookies = None
        cls._expires = None
        cls.delete_cache_file()

    @classmethod
    async def create_authed(cls, model, messages, auth_result, **kwargs):
        if getattr(auth_result, "api_key", None) != "fresh-token":
            raise MissingAuthError("Access token is not valid")
        for message in messages:
            yield message["content"]

    @classmethod
    async def on_auth_async(cls, **kwargs):
        cls._api_key = "fresh-token"
        yield AuthResult(api_key="fresh-token")
