from json import loads
from re import findall
from uuid import uuid4

from tls_client import Session


class Completion:
    def __init__(self, chat: list = []):
        self.chat = chat
        self.client = Session(client_identifier="chrome_108")
        self.client.headers = {
            "authority": "you.com",
            "accept": "text/event-stream",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "cache-control": "no-cache",
            "referer": "https://you.com/search?q=who+are+you&tbm=youchat",
            "sec-ch-ua": '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            'cookie': f'safesearch_guest=Moderate; uuid_guest={str(uuid4())}',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }

    def create(
            self,
            prompt: str,
            page: int = 1,
            count: int = 10,
            safe_search: str = "Moderate",
            on_shopping_page: bool = False,
            mkt: str = "",
            response_filter: str = "WebPages,Translations,TimeZone,Computation,RelatedSearches",
            domain: str = "youchat",
            query_trace_id: str = None,
            include_links: bool = False,
            detailed: bool = False,
            debug: bool = False) -> dict:

        response = self.client.get(f"https://you.com/api/streamingSearch", params={
            "q": prompt,
            "page": page,
            "count": count,
            "safeSearch": safe_search,
            "onShoppingPage": on_shopping_page,
            "mkt": mkt,
            "responseFilter": response_filter,
            "domain": domain,
            "queryTraceId": str(uuid4()) if query_trace_id is None else query_trace_id,
            "chat": str(self.chat),  # {"question":"","answer":" '"}
        }
                              )

        if debug:
            print('\n\n------------------\n\n')
            print(response.text)
            print('\n\n------------------\n\n')

        you_chat_serp_results = findall(r'youChatSerpResults\ndata: (.*)\n\nevent', response.text)[0]
        third_party_search_results = findall(r"thirdPartySearchResults\ndata: (.*)\n\nevent", response.text)[0]

        text = response.text.split('}]}\n\nevent: youChatToken\ndata: {"youChatToken": "')[-1]
        text = text.replace('"}\n\nevent: youChatToken\ndata: {"youChatToken": "', '')
        text = text.replace('event: done\ndata: I\'m Mr. Meeseeks. Look at me.\n\n', '')

        extra = {
            'youChatSerpResults': loads(you_chat_serp_results)
        }

        return {
            'response': text,
            'links': loads(third_party_search_results)['search']["third_party_search_results"] if include_links else None,
            'extra': extra if detailed else None,
        }
