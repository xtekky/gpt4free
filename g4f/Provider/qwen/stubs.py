from typing import Dict, Optional, Union

class ErrorDataDict(Dict):
    pass

class QwenCredentials(Dict):
    pass

class IQwenOAuth2Client:
    def setCredentials(self, credentials: QwenCredentials):
        raise NotImplementedError

    def getCredentials(self) -> QwenCredentials:
        raise NotImplementedError

    async def getAccessToken(self) -> Dict[str, Optional[str]]:
        raise NotImplementedError

    async def requestDeviceAuthorization(self, options: dict) -> Union[Dict, ErrorDataDict]:
        raise NotImplementedError

    async def pollDeviceToken(self, options: dict) -> Union[Dict, ErrorDataDict]:
        raise NotImplementedError

    async def refreshAccessToken(self) -> Union[Dict, ErrorDataDict]:
        raise NotImplementedError
