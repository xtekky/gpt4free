from __future__ import annotations

from ..template import OpenaiTemplate

class Custom(OpenaiTemplate):
    label = "Custom Provider"
    working = True
    needs_auth = False
    models_needs_auth = False
    base_url = "http://localhost:8080/v1"
    sort_models = False

    @classmethod
    def get_models(cls, api_key: str = None, base_url: str = None, **kwargs) -> list[str]:
        if cls.models:
            return cls.models
        try:
            return super().get_models(api_key=api_key, base_url=base_url, **kwargs)
        except Exception as e:
            # If no explicit base_url is provided and it fails (e.g. hits the Flask GUI itself -> 404),
            # silently ignore to prevent flooding logs. If explicit base_url provided, raise the error.
            if base_url is None:
                return []
            raise e

class Feature(Custom):
    label = "Feature Provider"
    working = False