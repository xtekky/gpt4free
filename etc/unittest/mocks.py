from g4f.providers.base_provider import AbstractProvider, AsyncProvider, AsyncGeneratorProvider
from g4f.image import ImageResponse
from g4f.errors import MissingAuthError

class ProviderMock(AbstractProvider):
    working = True

    @classmethod
    def create_completion(
        cls, model, messages, stream, **kwargs
    ):
        yield "Mock"

class AsyncProviderMock(AsyncProvider):
    working = True

    @classmethod
    async def create_async(
        cls, model, messages, **kwargs
    ):
        return "Mock"

class AsyncGeneratorProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, **kwargs
    ):
        yield "Mock"

class ModelProviderMock(AbstractProvider):
    working = True

    @classmethod
    def create_completion(
        cls, model, messages, stream, **kwargs
    ):
        yield model

class YieldProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, **kwargs
    ):
        for message in messages:
            yield message["content"]

class YieldImageResponseProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, prompt: str, **kwargs
    ):
        yield ImageResponse(prompt, "")

class MissingAuthProviderMock(AbstractProvider):
    working = True

    @classmethod
    def create_completion(
        cls, model, messages, stream, **kwargs
    ):
        raise MissingAuthError(cls.__name__)
        yield cls.__name__

class RaiseExceptionProviderMock(AbstractProvider):
    working = True

    @classmethod
    def create_completion(
        cls, model, messages, stream, **kwargs
    ):
        raise RuntimeError(cls.__name__)
        yield cls.__name__

class AsyncRaiseExceptionProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, **kwargs
    ):
        raise RuntimeError(cls.__name__)
        yield cls.__name__

class YieldNoneProviderMock(AsyncGeneratorProvider):
    working = True

    @classmethod
    async def create_async_generator(
        cls, model, messages, stream, **kwargs
    ):
        yield None