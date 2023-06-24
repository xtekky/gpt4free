import json
import re
from typing import Optional, List, Dict, Any
from uuid import uuid4

from fake_useragent import UserAgent
from pydantic import BaseModel
from requests import RequestException
from retrying import retry
from tls_client import Session
from tls_client.response import Response


class YouResponse(BaseModel):
    text: Optional[str] = None
    links: List[str] = []
    extra: Dict[str, Any] = {}


class Completion:
    @staticmethod
    def create(
        prompt: str,
        page: int = 1,
        count: int = 10,
        safe_search: str = 'Moderate',
        on_shopping_page: bool = False,
        mkt: str = '',
        response_filter: str = 'WebPages,Translations,TimeZone,Computation,RelatedSearches',
        domain: str = 'youchat',
        query_trace_id: str = None,
        chat: list = None,
        include_links: bool = False,
        detailed: bool = False,
        debug: bool = False,
        proxy: Optional[str] = None,
    ) -> YouResponse:
        if chat is None:
            chat = []

        proxies = {'http': 'http://' + proxy, 'https': 'http://' + proxy} if proxy else {}

        client = Session(client_identifier='chrome_108')
        client.headers = Completion.__get_headers()
        client.proxies = proxies

        params = {
            'q': prompt,
            'page': page,
            'count': count,
            'safeSearch': safe_search,
            'onShoppingPage': on_shopping_page,
            'mkt': mkt,
            'responseFilter': response_filter,
            'domain': domain,
            'queryTraceId': str(uuid4()) if query_trace_id is None else query_trace_id,
            'chat': str(chat),  # {'question':'','answer':' ''}
        }

        try:
            response = Completion.__make_request(client, params)
        except Exception:
            return Completion.__get_failure_response()

        if debug:
            print('\n\n------------------\n\n')
            print(response.text)
            print('\n\n------------------\n\n')

        you_chat_serp_results = re.search(
            r'(?<=event: youChatSerpResults\ndata:)(.*\n)*?(?=event: )', response.text
        ).group()
        third_party_search_results = re.search(
            r'(?<=event: thirdPartySearchResults\ndata:)(.*\n)*?(?=event: )', response.text
        ).group()
        # slots                   = findall(r"slots\ndata: (.*)\n\nevent", response.text)[0]

        text = ''.join(re.findall(r'{\"youChatToken\": \"(.*?)\"}', response.text))

        extra = {
            'youChatSerpResults': json.loads(you_chat_serp_results),
            # 'slots'                   : loads(slots)
        }

        response = YouResponse(text=text.replace('\\n', '\n').replace('\\\\', '\\').replace('\\"', '"'))
        if include_links:
            response.links = json.loads(third_party_search_results)['search']['third_party_search_results']

        if detailed:
            response.extra = extra

        return response

    @staticmethod
    def __get_headers() -> dict:
        return {
            'authority': 'you.com',
            'accept': 'text/event-stream',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'cache-control': 'no-cache',
            'referer': 'https://you.com/search?q=who+are+you&tbm=youchat',
            'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'cookie': f'safesearch_guest=Moderate; uuid_guest={str(uuid4())}',
            'user-agent': UserAgent().random,
        }

    @staticmethod
    def __get_failure_response() -> YouResponse:
        return YouResponse(text='Unable to fetch the response, Please try again.')

    @staticmethod
    @retry(
        wait_fixed=5000,
        stop_max_attempt_number=5,
        retry_on_exception=lambda e: isinstance(e, RequestException),
    )
    def __make_request(client: Session, params: dict) -> Response:
        response = client.get(f'https://you.com/api/streamingSearch', params=params)
        if 'youChatToken' not in response.text:
            print('retry')
            raise RequestException('Unable to get the response from server')
        return response
