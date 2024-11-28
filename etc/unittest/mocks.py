from g4f.providers.base_provider import AbstractProvider, AsyncProvider, AsyncGeneratorProvider

class ProviderMock(AbstractProvider):
    working = True

    def create_completion(
        model, messages, stream, **kwargs
    ):
        yield "Mock"

class AsyncProviderMock(AsyncProvider):
    working = True

    async def create_async(
        model, messages, **kwargs
    ):
        return "Mock"

class AsyncGeneratorProviderMock(AsyncGeneratorProvider):
    working = True

    async def create_async_generator(
        model, messages, stream, **kwargs
    ):
        yield "Mock"

class ModelProviderMock(AbstractProvider):
    working = True

    def create_completion(
        model, messages, stream, **kwargs
    ):
        yield model

class YieldProviderMock(AsyncGeneratorProvider):
    working = True

    async def create_async_generator(
        model, messages, stream, **kwargs
    ):
        for message in messages:
            yield message["content"]

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
    
    async def create_async_generator(
        model, messages, stream, **kwargs
    ):
        yield None