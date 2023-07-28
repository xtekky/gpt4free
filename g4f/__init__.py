from . import models
from .provider import BaseProvider
from .typing import Any, CreateResult

logging = False


class ChatCompletion:
    @staticmethod
    def create(
        model: models.Model | str,
        messages: list[dict[str, str]],
        provider: type[BaseProvider] | None = None,
        stream: bool = False,
        auth: str | None = None,
        **kwargs: Any,
    ) -> CreateResult | str:
        if isinstance(model, str):
            try:
                model = models.ModelUtils.convert[model]
            except KeyError:
                raise Exception(f"The model: {model} does not exist")

        provider = model.best_provider if provider == None else provider

        if not provider.working:
            raise Exception(f"{provider.__name__} is not working")

        if provider.needs_auth and not auth:
            raise Exception(
                f'ValueError: {provider.__name__} requires authentication (use auth="cookie or token or jwt ..." param)'
            )
        if provider.needs_auth:
            kwargs["auth"] = auth

        if not provider.supports_stream and stream:
            raise Exception(
                f"ValueError: {provider.__name__} does not support 'stream' argument"
            )

        if logging:
            print(f"Using {provider.__name__} provider")

        result = provider.create_completion(model.name, messages, stream, **kwargs)
        return result if stream else "".join(result)
