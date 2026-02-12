from typing import Dict, Optional, Union


class ErrorDataDict(Dict):
    pass


class GithubCredentials(Dict):
    pass


class IGithubOAuth2Client:
    def setCredentials(self, credentials: GithubCredentials):
        raise NotImplementedError

    def getCredentials(self) -> GithubCredentials:
        raise NotImplementedError

    async def getAccessToken(self) -> Dict[str, Optional[str]]:
        raise NotImplementedError

    async def requestDeviceAuthorization(self, options: dict) -> Union[Dict, ErrorDataDict]:
        raise NotImplementedError

    async def pollDeviceToken(self, options: dict) -> Union[Dict, ErrorDataDict]:
        raise NotImplementedError
