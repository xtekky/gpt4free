from tls_client import Session
from re         import findall
from json       import loads, dumps
from uuid       import uuid4


class Completion:
    def __init__(self):
        self.chat_history = []

        self.client         = Session(client_identifier="chrome_108")
        self.client.headers = {
            "authority"          : "you.com",
            "accept"             : "text/event-stream",
            "accept-language"    : "tr-TR,tr;q=0.9,en-US;q=0.7,fr-FR;q=0.5",
            "cache-control"      : "no-cache",
            "referer"            : "https://you.com/search?q=who+are+you&tbm=youchat",
            "sec-ch-ua"          : '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            "sec-ch-ua-mobile"   : "?0",
            "sec-ch-ua-platform" : '"Windows"',
            "sec-fetch-dest"     : "empty",
            "sec-fetch-mode"     : "cors",
            "sec-fetch-site"     : "same-origin",
            "cookie"             : f"safesearch_guest=Moderate; uuid_guest={uuid4()}",
            "user-agent"         : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        }

    def create(self, prompt:str, debug:bool=False, includelinks:bool=False, detailed:bool=False) -> dict:
        response = self.client.get(
            url    = "https://you.com/api/streamingSearch",
            params = {
                "q"              : prompt,
                "page"           : 1,
                "count"          : 10,
                "safeSearch"     : "Moderate",
                "onShoppingPage" : False,
                "mkt"            : "",
                "responseFilter" : "WebPages,Translations,TimeZone,Computation,RelatedSearches",
                "domain"         : "youchat",
                "queryTraceId"   : f"{uuid4()}",
                "chat"           : dumps(self.chat_history)
            }
        )
        
        if debug:
            print("\n\n------------------\n\n")
            print(response.text)
            print("\n\n------------------\n\n")

        youChatSerpResults      = findall(r"youChatSerpResults\ndata: (.*)\n\nevent", response.text)[0]
        thirdPartySearchResults = findall(r"thirdPartySearchResults\ndata: (.*)\n\nevent", response.text)[0]
        #slots                   = findall(r"slots\ndata: (.*)\n\nevent", response.text)[0]
        
        text = response.text.split('}]}\n\nevent: youChatToken\ndata: {"youChatToken": "')[-1]
        text = text.replace('"}\n\nevent: youChatToken\ndata: {"youChatToken": "', '')
        text = text.replace('event: done\ndata: I\'m Mr. Meeseeks. Look at me.\n\n', '')
        text = text[:-4] # trims '"}', along with the last two remaining newlines

        extra = {
            "youChatSerpResults" : loads(youChatSerpResults).get("youChatSerpResults"),
            #"slots"              : loads(slots)
        }

        data = {
            "response" : text.encode("utf-8").decode("unicode-escape").strip(),
            "links"    : loads(thirdPartySearchResults)["search"]["third_party_search_results"] if includelinks else None,
            "extra"    : extra if detailed else None
        }
        self.chat_history.append({"question": prompt, "answer": data["response"]})
        return data