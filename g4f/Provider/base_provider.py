from abc import ABC, abstractmethod

from ..typing import Any, CreateResult


class BaseProvider(ABC):
    url: str
    working = False
    needs_auth = False
    supports_stream = False
    supports_gpt_35_turbo = False
    supports_gpt_4 = False

    @staticmethod
    @abstractmethod
    def create_completion(
        model: str,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> CreateResult:
        raise NotImplementedError()

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"