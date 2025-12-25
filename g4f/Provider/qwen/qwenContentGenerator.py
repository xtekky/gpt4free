from typing import Optional, Dict

from .sharedTokenManager import SharedTokenManager
from .qwenOAuth2 import IQwenOAuth2Client

# Default base URL if not specified
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenContentGenerator:
    def __init__(
        self,
        qwen_client: IQwenOAuth2Client
    ):
        self.qwen_client = qwen_client
        self.base_url = DEFAULT_QWEN_BASE_URL
        self.shared_manager = SharedTokenManager.getInstance()

        # Initialize API URL with default, may be updated later
        self.base_url = self.base_url

    def get_current_endpoint(self, resource_url: Optional[str]) -> str:
        url = resource_url if resource_url else self.base_url
        if not url.startswith("http"):
            url = "https://" + url
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return url

    async def get_valid_token(self) -> Dict[str, str]:
        """
        Obtain a valid token and endpoint from shared token manager.
        """
        credentials = await self.shared_manager.getValidCredentials(self.qwen_client)
        token = credentials.get("access_token")
        resource_url = credentials.get("resource_url")
        endpoint = self.get_current_endpoint(resource_url)
        if not token:
            raise Exception("No valid access token obtained.")
        return {"token": token, "endpoint": endpoint}