from __future__ import annotations

from ..template import OpenaiTemplate


class OrcaRouter(OpenaiTemplate):
    label = "OrcaRouter"
    url = "https://www.orcarouter.ai"
    login_url = "https://www.orcarouter.ai"
    base_url = "https://api.orcarouter.ai/v1"
    working = True
    needs_auth = True
    default_model = "orcarouter/auto"

    @classmethod
    def get_headers(
        cls, stream: bool, api_key: str = None, headers: dict = None
    ) -> dict:
        return {
            **super().get_headers(stream, api_key, headers),
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        }
