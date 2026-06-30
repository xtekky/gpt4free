from __future__ import annotations

import os
try:
    import ssl
    has_ssl = True
except ImportError:
    has_ssl = False
import time
import uuid
from pathlib import Path

import json
from aiohttp import ClientSession, TCPConnector, BaseConnector
from ...requests import raise_for_status

from ...typing import AsyncResult, Messages
from ..base_provider import AsyncGeneratorProvider, ProviderModelMixin
from ...errors import MissingAuthError
from ..helper import get_connector
from ...cookies import get_cookies_dir

access_token = ""
token_expires_at = 0

RUSSIAN_CA_CERT = """-----BEGIN CERTIFICATE-----
MIIFwjCCA6qgAwIBAgICEAAwDQYJKoZIhvcNAQELBQAwcDELMAkGA1UEBhMCUlUx
PzA9BgNVBAoMNlRoZSBNaW5pc3RyeSBvZiBEaWdpdGFsIERldmVsb3BtZW50IGFu
ZCBDb21tdW5pY2F0aW9uczEgMB4GA1UEAwwXUnVzc2lhbiBUcnVzdGVkIFJvb3Qg
Q0EwHhcNMjIwMzAxMjEwNDE1WhcNMzIwMjI3MjEwNDE1WjBwMQswCQYDVQQGEwJS
VTE/MD0GA1UECgw2VGhlIE1pbmlzdHJ5IG9mIERpZ2l0YWwgRGV2ZWxvcG1lbnQg
YW5kIENvbW11bmljYXRpb25zMSAwHgYDVQQDDBdSdXNzaWFuIFRydXN0ZWQgUm9v
dCBDQTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAMfFOZ8pUAL3+r2n
qqE0Zp52selXsKGFYoG0GM5bwz1bSFtCt+AZQMhkWQheI3poZAToYJu69pHLKS6Q
XBiwBC1cvzYmUYKMYZC7jE5YhEU2bSL0mX7NaMxMDmH2/NwuOVRj8OImVa5s1F4U
zn4Kv3PFlDBjjSjXKVY9kmjUBsXQrIHeaqmUIsPIlNWUnimXS0I0abExqkbdrXbX
YwCOXhOO2pDUx3ckmJlCMUGacUTnylyQW2VsJIyIGA8V0xzdaeUXg0VZ6ZmNUr5Y
Ber/EAOLPb8NYpsAhJe2mXjMB/J9HNsoFMBFJ0lLOT/+dQvjbdRZoOT8eqJpWnVD
U+QL/qEZnz57N88OWM3rabJkRNdU/Z7x5SFIM9FrqtN8xewsiBWBI0K6XFuOBOTD
4V08o4TzJ8+Ccq5XlCUW2L48pZNCYuBDfBh7FxkB7qDgGDiaftEkZZfApRg2E+M9
G8wkNKTPLDc4wH0FDTijhgxR3Y4PiS1HL2Zhw7bD3CbslmEGgfnnZojNkJtcLeBH
BLa52/dSwNU4WWLubaYSiAmA9IUMX1/RpfpxOxd4Ykmhz97oFbUaDJFipIggx5sX
ePAlkTdWnv+RWBxlJwMQ25oEHmRguNYf4Zr/Rxr9cS93Y+mdXIZaBEE0KS2iLRqa
OiWBki9IMQU4phqPOBAaG7A+eP8PAgMBAAGjZjBkMB0GA1UdDgQWBBTh0YHlzlpf
BKrS6badZrHF+qwshzAfBgNVHSMEGDAWgBTh0YHlzlpfBKrS6badZrHF+qwshzAS
BgNVHRMBAf8ECDAGAQH/AgEEMA4GA1UdDwEB/wQEAwIBhjANBgkqhkiG9w0BAQsF
AAOCAgEAALIY1wkilt/urfEVM5vKzr6utOeDWCUczmWX/RX4ljpRdgF+5fAIS4vH
tmXkqpSCOVeWUrJV9QvZn6L227ZwuE15cWi8DCDal3Ue90WgAJJZMfTshN4OI8cq
W9E4EG9wglbEtMnObHlms8F3CHmrw3k6KmUkWGoa+/ENmcVl68u/cMRl1JbW2bM+
/3A+SAg2c6iPDlehczKx2oa95QW0SkPPWGuNA/CE8CpyANIhu9XFrj3RQ3EqeRcS
AQQod1RNuHpfETLU/A2gMmvn/w/sx7TB3W5BPs6rprOA37tutPq9u6FTZOcG1Oqj
C/B7yTqgI7rbyvox7DEXoX7rIiEqyNNUguTk/u3SZ4VXE2kmxdmSh3TQvybfbnXV
4JbCZVaqiZraqc7oZMnRoWrXRG3ztbnbes/9qhRGI7PqXqeKJBztxRTEVj8ONs1d
WN5szTwaPIvhkhO3CO5ErU2rVdUr89wKpNXbBODFKRtgxUT70YpmJ46VVaqdAhOZ
D9EUUn4YaeLaS8AjSF/h7UkjOibNc4qVDiPP+rkehFWM66PVnP1Msh93tc+taIfC
EYVMxjh8zNbFuoc7fzvvrFILLe7ifvEIUqSVIC/AzplM/Jxw7buXFeGP1qVCBEHq
391d/9RAfaZ12zkwFsl+IKwE/OZxW8AHa9i1p4GO0YSNuczzEm4=
-----END CERTIFICATE-----"""

class GigaChat(AsyncGeneratorProvider, ProviderModelMixin):
    url = "https://developers.sber.ru/gigachat"
    working = True
    supports_message_history = True
    supports_system_message = True
    supports_stream = True
    needs_auth = True
    default_model = "GigaChat"
    models = ["GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max", default_model, "GigaChat-Pro", "GigaChat-Max"]

    @classmethod
    async def create_async_generator(
            cls,
            model: str,
            messages: Messages,
            stream: bool = True,
            proxy: str = None,
            api_key: str = None,
            connector: BaseConnector = None,
            scope: str = "GIGACHAT_API_PERS",
            update_interval: float = 0,
            **kwargs
    ) -> AsyncResult:
        global access_token, token_expires_at
        model = cls.get_model(model)
        if not api_key:
            raise MissingAuthError('Missing "api_key"')

        # Create certificate file in cookies directory
        cookies_dir = Path(get_cookies_dir())
        cert_file = cookies_dir / 'russian_trusted_root_ca.crt'

        # Write certificate if it doesn't exist
        if not cert_file.exists():
            cert_file.write_text(RUSSIAN_CA_CERT)

        if has_ssl and connector is None:
            ssl_context = ssl.create_default_context(cafile=str(cert_file))
            connector = TCPConnector(ssl_context=ssl_context)

        async with ClientSession(connector=get_connector(connector, proxy)) as session:
            if token_expires_at - int(time.time() * 1000) < 60000:
                async with session.post(url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                                        headers={"Authorization": f"Bearer {api_key}",
                                                 "RqUID": str(uuid.uuid4()),
                                                 "Content-Type": "application/x-www-form-urlencoded"},
                                        data={"scope": scope}) as response:
                    await raise_for_status(response)
                    data = await response.json()
                access_token = data['access_token']
                token_expires_at = data['expires_at']

            async with session.post(url="https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {access_token}"},
                                    json={
                                        "model": model,
                                        "messages": messages,
                                        "stream": stream,
                                        "update_interval": update_interval,
                                        **kwargs
                                    }) as response:
                await raise_for_status(response)

                async for line in response.content:
                    if not stream:
                        yield json.loads(line.decode("utf-8"))['choices'][0]['message']['content']
                        return

                    if line and line.startswith(b"data:"):
                        line = line[6:-1]  # remove "data: " prefix and "\n" suffix
                        if line.strip() == b"[DONE]":
                            return
                        else:
                            msg = json.loads(line.decode("utf-8"))['choices'][0]
                            content = msg['delta']['content']

                            if content:
                                yield content

                            if 'finish_reason' in msg:
                                return
