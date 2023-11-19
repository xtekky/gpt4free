from __future__ import annotations

import uuid, json, asyncio
from py_arkose_generator.arkose import get_values_for_request
from asyncstdlib.itertools import tee
from async_property import async_cached_property

from ..base_provider import AsyncGeneratorProvider
from ..helper import get_browser, get_event_loop
from ...typing import AsyncResult, Messages
from ...requests import StreamSession

models = {
    "gpt-3.5":       "text-davinci-002-render-sha",
    "gpt-3.5-turbo": "text-davinci-002-render-sha",
    "gpt-4":         "gpt-4",
    "gpt-4-gizmo":   "gpt-4-gizmo"
}

class OpenaiChat(AsyncGeneratorProvider):
    url                   = "https://chat.openai.com"
    working               = True
    needs_auth            = True
    supports_gpt_35_turbo = True
    supports_gpt_4        = True
    _access_token: str    = None

    @classmethod
    async def create(
        cls,
        prompt: str = None,
        model: str = "",
        messages: Messages = [],
        history_disabled: bool = False,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        **kwargs
    ) -> Response:
        if prompt:
            messages.append({"role": "user", "content": prompt})
        generator = cls.create_async_generator(
            model,
            messages,
            history_disabled=history_disabled,
            action=action,
            conversation_id=conversation_id,
            parent_id=parent_id,
            response_fields=True,
            **kwargs
        )
        fields: ResponseFields = await anext(generator)
        if "access_token" not in kwargs:
            kwargs["access_token"] = cls._access_token
        return Response(
            generator,
            fields,
            action,
            messages,
            kwargs
        )

    @classmethod
    async def create_async_generator(
        cls,
        model: str,
        messages: Messages,
        proxy: str = None,
        timeout: int = 120,
        access_token: str = None,
        auto_continue: bool = False,
        history_disabled: bool = True,
        action: str = "next",
        conversation_id: str = None,
        parent_id: str = None,
        response_fields: bool = False,
        **kwargs
    ) -> AsyncResult:
        if not model:
            model = "gpt-3.5"
        elif model not in models:
            raise ValueError(f"Model are not supported: {model}")
        if not parent_id:
            parent_id = str(uuid.uuid4())
        if not access_token:
            access_token = await cls.get_access_token(proxy)
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "Cookie": 'intercom-device-id-dgkjq2bp=0f047573-a750-46c8-be62-6d54b56e7bf0; ajs_user_id=user-iv3vxisaoNodwWpxmNpMfekH; ajs_anonymous_id=fd91be0b-0251-4222-ac1e-84b1071e9ec1; __Host-next-auth.csrf-token=d2b5f67d56f7dd6a0a42ae4becf2d1a6577b820a5edc88ab2018a59b9b506886%7Ce5c33eecc460988a137cbc72d90ee18f1b4e2f672104f368046df58e364376ac; _cfuvid=gt_mA.q6rue1.7d2.AR0KHpbVBS98i_ppfi.amj2._o-1700353424353-0-604800000; cf_clearance=GkHCfPSFU.NXGcHROoe4FantnqmnNcluhTNHz13Tk.M-1700353425-0-1-dfe77f81.816e9bc2.714615da-0.2.1700353425; __Secure-next-auth.callback-url=https%3A%2F%2Fchat.openai.com; intercom-session-dgkjq2bp=UWdrS1hHazk5VXN1c0V5Q1F0VXdCQmsyTU9pVjJMUkNpWnFnU3dKWmtIdGwxTC9wbjZuMk5hcEc0NWZDOGdndS0tSDNiaDNmMEdIL1RHU1dFWDBwOHFJUT09--f754361b91fddcd23a13b288dcb2bf8c7f509e91; _uasid="Z0FBQUFBQmxXVnV0a3dmVno4czRhcDc2ZVcwaUpSNUdZejlDR25YSk5NYTJQQkpyNmRvOGxjTHMyTlAxWmJhaURrMVhjLXZxQXdZeVpBbU1aczA5WUpHT2dwaS1MOWc4MnhyNWFnbGRzeGdJcGFKT0ZRdnBTMVJHcGV2MGNTSnVQY193c0hqUWIycHhQRVF4dENlZ3phcDdZeHgxdVhoalhrZmtZME9NbWhMQjdVR3Vzc3FRRk0ybjJjNWMwTWtIRjdPb19lUkFtRmV2MDVqd1kwWU11QTYtQkdZenEzVHhLMGplY1hZM3FlYUt1cVZaNWFTRldleEJETzJKQjk1VTJScy1GUnMxUVZWMnVxYklxMjdockVZbkZyd1R4U1RtMnA1ZzlSeXphdmVOVk9xeEdrRkVOSjhwTVd1QzFtQjhBcWdDaE92Q1VlM2pwcjFQTXRuLVJNRVlZSGpIdlZ0aGV3PT0="; _dd_s=rum=0&expire=1700356244884; __Secure-next-auth.session-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..3aK6Fbdy2_8f07bf.8eT2xgonrCnz7ySY6qXFsg3kzL6UQfXKAYaw3tyn-6_X9657zy47k9qGvmi9mF0QKozj5jau3_Ca62AQQ7FmeC6Y2F1urtzqrXqwTTsQ2LuzFPIQkx6KKb2DXc8zW2-oyEzJ_EY5yxfLB2RlRkSh3M7bYNZh4_ltEcfkj38s_kIPGMxv34udtPWGWET99MCjkdwQWXylJag4s0fETA0orsBAKnGCyqAUNJbb_D7BYtGSV-MQ925kZMG6Di_QmfO0HQWURDYjmdRNcuy1PT_xJ1DJko8sjL42i4j3RhkNDkhqCIqyYImz2eHFWHW7rYKxTkrBhlCPMS5hRdcCswD7JYPcSBiwnVRYgyOocFGXoFvQgIZ2FX9NiZ3SMEVM1VwIGSE-qH0H2nMa8_iBvsOgOWJgKjVAvzzyzZvRVDUUHzJrikSFPNONVDU3h-04c1kVL4qIu9DfeTPN7n8AvNmYwMbro0L9-IUAeXNo4-pwF0Kt-AtTsamqWvMqnK4O_YOyLnDDlvkmnOvDC2d5uinwlQIxr6APO6qFfGLlHiLZemKoekxEE1Fx70dl-Ouhk1VIzbF3OC6XNNxeBm9BUYUiHdL0wj2H9rHgX4cz6ZmS_3VTgpD6UJh-evu5KJ2gIvjYmVbyzEN0aPNDxfvBaOm-Ezpy4bUJ2bUrOwNn-0knWkDiTvjYmNhCyefPCtCF6rpKNay8PCw_yh79C4SdEP6Q4V7LI0Tvdi5uz7kLCiBC4AT9L0ao1WDX03mkUOpjvzHDvPLmj8chW3lTVm_kA0eYGQY4wT0jzleWlfV0Q8rB2oYECNLWksA3F1zlGfcl4lQjprvTXRePkvAbMpoJEsZD3Ylq7-foLDLk4-M2LYAFZDs282AY04sFjAjQBxTELFCCuDgTIgTXSIskY_XCxpVXDbdLlbCJY7XVK45ybwtfqwlKRp8Mo0B131uQAFc-migHaUaoGujxJJk21bP8F0OmhNYHBo4FQqE1rQm2JH5bNM7txKeh5KXdJgVUVbRSr7OIp_OF5-Bx_v9eRBGAIDkue26E2-O8Rnrp5zQ5TnvecQLDaUzWavCLPwsZ0_gsOLBxNOmauNYZtF8IElCsQSFDdhoiMxXsYUm4ZYKEAy3GWq8HGTAvBhNkh1hvnI7y-d8-DOaZf_D_D98-olZfm-LUkeosLNpPB9rxYMqViCiW3KrXE9Yx0wlFm5ePKaVvR7Ym_EPhSOhJBKFPCvdTdMZSNPUcW0ZJBVByq0A9sxD51lYq3gaFyqh94S4s_ox182AQ3szGzHkdgLcnQmJG9OYvKxAVcd43eg6_gODAYhx02GjbMw-7JTAhyXSeCrlMteHyOXl8hai-3LilC3PmMzi7Vbu49dhF1s4LcVlUowen5ira44rQQaB26mdaOUoQfodgt66M3RTWGPXyK1Nb72AzSXsCKyaQPbzeb6cN0fdGSdG4ktwvR04eFNEkquo_3aKu2GmUKTD0XcRx9dYrfXjgY-X1DDTVs1YND2gRhdx7FFEeBVjtbj2UqmG3Rvd4IcHGe7OnYWw2MHDcol68SsR1KckXWwWREz7YTGUnDB2M1kx_H4W2mjclytnlHOnYU3RflegRPeSTbdzUZJvGKXCCz45luHkQWN_4DExE76D-9YqbFIz-RY5yL4h-Zs-i2xjm2K-4xCMM9nQIOqhLMqixIZQ2ldDAidKoYtbs5ppzbcBLyrZM96bq9DwRBY3aacqWdlRd-TfX0wv5KO4fo0sSh5FsuhuN0zcEV_NNXgqIEM_p14EcPqgbrAvCBQ8os70TRBQLXiF0EniSofGjxwF8kQvUk3C6Wfc8cTTeN-E6GxCVTn91HBwA1iSEZlRLMVb8_BcRJNqwbgnb_07jR6-eo42u88CR3KQdAWwbQRdMxsURFwZ0ujHXVGG0Ll6qCFBcHXWyDO1x1yHdHnw8_8yF26pnA2iPzrFR-8glMgIA-639sLuGAxjO1_ZuvJ9CAB41Az9S_jaZwaWy215Hk4-BRYD-MKmHtonwo3rrxhE67WJgbbu14efsw5nT6ow961pffgwXov5VA1Rg7nv1E8RvQOx7umWW6o8R4W6L8f2COsmPTXfgwIjoJKkjhUqAQ8ceG7cM0ET-38yaC0ObU8EkXfdGGgxI28qTEZWczG66_iM4hw7QEGCY5Cz2kbO6LETAiw9OsSigtBvDS7f0Ou0bZ41pdK7G3FmvdZAnjWPjObnDF4k4uWfn7mzt0fgj3FyqK20JezRDyGuAbUUhOvtZpc9sJpzxR34eXEZTouuALrHcGuNij4z6rx51FrQsaMtiup8QVrhtZbXtKLMYnWYSbkhuTeN2wY-xV1ZUsQlakIZszzGF7kuIG87KKWMpuPMvbXjz6Pp_gWJiIC6aQuk8xl5g0iBPycf_6Q-MtpuYxzNE2TpI1RyR9mHeXmteoRzrFiWp7yEC-QGNFyAJgxTqxM3CjHh1Jt6IddOsmn89rUo1dZM2Smijv_fbIv3avXLkIPX1KZjILeJCtpU0wAdsihDaRiRgDdx8fG__F8zuP0n7ziHas73cwrfg-Ujr6DhC0gTNxyd9dDA_oho9N7CQcy6EFmfNF2te7zpLony0859jtRv2t1TnpzAa1VvMK4u6mXuJ2XDo04_6GzLO3aPHinMdl1BcIAWnqAqWAu3euGFLTHOhXlfijut9N1OCifd_zWjhVtzlR39uFeCQBU5DyQArzQurdoMx8U1ETsnWgElxGSStRW-YQoPsAJ87eg9trqKspFpTVlAVN3t1GtoEAEhcwhe81SDssLmKGLc.7PqS6jRGTIfgTPlO7Ognvg; __cf_bm=VMWoAKEB45hQSwxXtnYXcurPaGZDJS4dMi6dIMFLwdw-1700355394-0-ATVsbq97iCaTaJbtYr8vtg1Zlbs3nLrJLKVBHYa2Jn7hhkGclqAy8Gbyn5ePEhDRqj93MsQmtayfYLqY5n4WiLY=; __cflb=0H28vVfF4aAyg2hkHFH9CkdHRXPsfCUf6VpYf2kz3RX'
        }
        async with StreamSession(
            proxies={"https": proxy},
            impersonate="chrome110",
            headers=headers,
            timeout=timeout
        ) as session:
            data = {
                "action": action,
                "arkose_token": await get_arkose_token(proxy, timeout),
                "conversation_id": conversation_id,
                "parent_message_id": parent_id,
                "model": models[model],
                "history_and_training_disabled": history_disabled and not auto_continue,
            }
            if action != "continue":
                data["messages"] = [{
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [messages[-1]["content"]]},
                }]
            first = True
            end_turn = EndTurn()
            while first or auto_continue and not end_turn.is_end:
                first = False
                async with session.post(f"{cls.url}/backend-api/conversation", json=data) as response:
                    try:
                        response.raise_for_status()
                    except:
                        raise RuntimeError(f"Error {response.status_code}: {await response.text()}")
                    last_message = 0
                    async for line in response.iter_lines():
                        if line.startswith(b"data: "):
                            line = line[6:]
                            if line == b"[DONE]":
                                break
                            try:
                                line = json.loads(line)
                            except:
                                continue
                            if "message" not in line:
                                continue
                            if "error" in line and line["error"]:
                                raise RuntimeError(line["error"])
                            if "message_type" not in line["message"]["metadata"]:
                                continue
                            if line["message"]["author"]["role"] != "assistant":
                                continue
                            if line["message"]["metadata"]["message_type"] in ("next", "continue", "variant"):
                                conversation_id = line["conversation_id"]
                                parent_id = line["message"]["id"]
                                if response_fields:
                                    response_fields = False
                                    yield ResponseFields(conversation_id, parent_id, end_turn)
                                new_message = line["message"]["content"]["parts"][0]
                                yield new_message[last_message:]
                                last_message = len(new_message)
                            if "finish_details" in line["message"]["metadata"]:
                                if line["message"]["metadata"]["finish_details"]["type"] == "max_tokens":
                                    end_turn.end()

                data = {
                    "action": "continue",
                    "arkose_token": await get_arkose_token(proxy, timeout),
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_id,
                    "model": models[model],
                    "history_and_training_disabled": False,
                }
                await asyncio.sleep(5)

    @classmethod
    async def browse_access_token(cls, proxy: str = None) -> str:
        def browse() -> str:
            try:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC

                driver = get_browser("~/openai", proxy=proxy)
            except ImportError:
                return
            try:
                driver.get(f"{cls.url}/")
                WebDriverWait(driver, 1200).until(
                    EC.presence_of_element_located((By.ID, "prompt-textarea"))
                )
                javascript = "return (await (await fetch('/api/auth/session')).json())['accessToken']"
                return driver.execute_script(javascript)
            finally:
                driver.quit()
        loop = get_event_loop()
        return await loop.run_in_executor(
            None,
            browse
        )

    @classmethod
    async def get_access_token(cls, proxy: str = None) -> str:
        if not cls._access_token:
            cls._access_token = await cls.browse_access_token(proxy)
        if not cls._access_token:
            raise RuntimeError("Read access token failed")
        return cls._access_token

    @classmethod
    @property
    def params(cls):
        params = [
            ("model", "str"),
            ("messages", "list[dict[str, str]]"),
            ("stream", "bool"),
            ("proxy", "str"),
            ("access_token", "str"),
        ]
        param = ", ".join([": ".join(p) for p in params])
        return f"g4f.provider.{cls.__name__} supports: ({param})"
    
async def get_arkose_token(proxy: str = None, timeout: int = None) -> str:
    config = {
        "pkey": "3D86FBBA-9D22-402A-B512-3420086BA6CC",
        "surl": "https://tcr9i.chat.openai.com",
        "headers": {
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        },
        "site": "https://chat.openai.com",
    }
    args_for_request = get_values_for_request(config)
    async with StreamSession(
        proxies={"https": proxy},
        impersonate="chrome107",
        timeout=timeout
    ) as session:
        async with session.post(**args_for_request) as response:
            response.raise_for_status()
            decoded_json = await response.json()
            if "token" in decoded_json:
                return decoded_json["token"]
            raise RuntimeError(f"Response: {decoded_json}")
        
class EndTurn():
    def __init__(self):
        self.is_end = False

    def end(self):
        self.is_end = True

class ResponseFields():
    def __init__(
        self,
        conversation_id: str,
        message_id: str,
        end_turn: EndTurn
    ):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self._end_turn = end_turn
        
class Response():
    def __init__(
        self,
        generator: AsyncResult,
        fields: ResponseFields,
        action: str,
        messages: Messages,
        options: dict
    ):
        self.aiter, self.copy = tee(generator)
        self.fields = fields
        self.action = action
        self._messages = messages
        self._options = options

    def __aiter__(self):
        return self.aiter
    
    @async_cached_property
    async def message(self) -> str:
        return "".join([chunk async for chunk in self.copy])
    
    async def next(self, prompt: str, **kwargs) -> Response:
        return await OpenaiChat.create(
            **self._options,
            prompt=prompt,
            messages=await self.messages,
            action="next",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    async def do_continue(self, **kwargs) -> Response:
        if self.end_turn:
            raise RuntimeError("Can't continue message. Message already finished.")
        return await OpenaiChat.create(
            **self._options,
            messages=await self.messages,
            action="continue",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    async def variant(self, **kwargs) -> Response:
        if self.action != "next":
            raise RuntimeError("Can't create variant with continue or variant request.")
        return await OpenaiChat.create(
            **self._options,
            messages=self._messages,
            action="variant",
            conversation_id=self.fields.conversation_id,
            parent_id=self.fields.message_id,
            **kwargs
        )
    
    @async_cached_property
    async def messages(self):
        messages = self._messages
        messages.append({
            "role": "assistant", "content": await self.message
        })
        return messages
    
    @property
    def end_turn(self):
        return self.fields._end_turn.is_end