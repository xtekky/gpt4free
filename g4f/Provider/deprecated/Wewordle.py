from __future__ import annotations

import random, string, time
from aiohttp import ClientSession

from ..base_provider import AsyncProvider


class Wewordle(AsyncProvider):
    url                    = "https://wewordle.org"
    working                = False
    supports_gpt_35_turbo  = True

    @classmethod
    async def create_async(
        cls,
        model: str,
        messages: list[dict[str, str]],
        proxy: str = None,
        **kwargs
    ) -> str:
        
        headers = {
            "accept"        : "*/*",
            "pragma"        : "no-cache",
            "Content-Type"  : "application/json",
            "Connection"    : "keep-alive"
        }

        _user_id = "".join(random.choices(f"{string.ascii_lowercase}{string.digits}", k=16))
        _app_id = "".join(random.choices(f"{string.ascii_lowercase}{string.digits}", k=31))
        _request_date = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        data = {
            "user"      : _user_id,
            "messages"  : messages,
            "subscriber": {
                "originalPurchaseDate"          : None,
                "originalApplicationVersion"    : None,
                "allPurchaseDatesMillis"        : {},
                "entitlements"                  : {"active": {}, "all": {}},
                "allPurchaseDates"              : {},
                "allExpirationDatesMillis"      : {},
                "allExpirationDates"            : {},
                "originalAppUserId"             : f"$RCAnonymousID:{_app_id}",
                "latestExpirationDate"          : None,
                "requestDate"                   : _request_date,
                "latestExpirationDateMillis"    : None,
                "nonSubscriptionTransactions"   : [],
                "originalPurchaseDateMillis"    : None,
                "managementURL"                 : None,
                "allPurchasedProductIdentifiers": [],
                "firstSeen"                     : _request_date,
                "activeSubscriptions"           : [],
            }
        }


        async with ClientSession(
            headers=headers
        ) as session:
            async with session.post(f"{cls.url}/gptapi/v1/android/turbo", proxy=proxy, json=data) as response:
                response.raise_for_status()
                content = (await response.json())["message"]["content"]
                if content:
                    return content