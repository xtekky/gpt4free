import json, random, string, time, requests

from ..typing       import Any, CreateResult
from .base_provider import BaseProvider


class Wewordle(BaseProvider):
    url                    = "https://wewordle.org/"
    working                = True
    supports_gpt_35_turbo  = True

    @classmethod
    def create_completion(
        cls,
        model: str,
        messages: list[dict[str, str]],
        stream: bool, **kwargs: Any) -> CreateResult:
        
        # randomize user id and app id
        _user_id = "".join(
            random.choices(f"{string.ascii_lowercase}{string.digits}", k=16))
        
        _app_id = "".join(
            random.choices(f"{string.ascii_lowercase}{string.digits}", k=31))
        
        # make current date with format utc
        _request_date = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        headers = {
            "accept"        : "*/*",
            "pragma"        : "no-cache",
            "Content-Type"  : "application/json",
            "Connection"    : "keep-alive"
            # user agent android client
            # 'User-Agent': 'Dalvik/2.1.0 (Linux; U; Android 10; SM-G975F Build/QP1A.190711.020)',
        }
        
        data: dict[str, Any] = {
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

        response = requests.post(f"{cls.url}gptapi/v1/android/turbo", 
                                 headers=headers, data=json.dumps(data))
        
        response.raise_for_status()
        _json = response.json()
        if "message" in _json:
            yield _json["message"]["content"]
