from gpt4free import quora, forefront, theb, you
import os
import sys
from typing import Optional
from abc import ABC, abstractmethod
import random

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


class BaseQuery(ABC):
    @abstractmethod
    def query(self, question: str, proxy: Optional[str] = None) -> str:
        pass


class ForefrontQuery(BaseQuery):
    def query(self, question: str, proxy: Optional[str] = None) -> str:
        token = forefront.Account.create(logging=False, proxy=proxy)
        try:
            return forefront.Completion.create(token=token, prompt='hello world', model='gpt-4', proxy=proxy).text
        except Exception as e:
            return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


class QuoraQuery(BaseQuery):
    def query(self, question: str, proxy: Optional[str] = None) -> str:
        token = quora.Account.create(
            logging=False, enable_bot_creation=True, proxy=proxy)
        return quora.Completion.create(model='gpt-4', prompt=question, token=token, proxy=proxy).text


class ThebQuery(BaseQuery):
    def query(self, question: str, proxy: Optional[str] = None) -> str:
        try:
            return ''.join(theb.Completion.create(prompt=question, proxy=proxy))
        except Exception as e:
            return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


class YouQuery(BaseQuery):
    def query(self, question: str, proxy: Optional[str] = None) -> str:
        try:
            result = you.Completion.create(prompt=question, proxy=proxy)
            return result.text
        except Exception as e:
            return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


avail_query_methods = {
    "Forefront": ForefrontQuery(),
    "Poe": QuoraQuery(),
    "Theb": ThebQuery(),
    "You": YouQuery(),
}


def query(user_input: str, selected_method: str = "Random", proxy: Optional[str] = None) -> str:
    if selected_method != "Random" and selected_method in avail_query_methods:
        try:
            return avail_query_methods[selected_method].query(user_input, proxy=proxy)
        except Exception as e:
            print(f"Error with {selected_method}: {e}")
            return "😵 Sorry, some error occurred please try again."

    success = False
    result = "😵 Sorry, some error occurred please try again."
    query_methods_list = list(avail_query_methods.values())

    while not success and query_methods_list:
        chosen_query = random.choice(query_methods_list)
        chosen_query_name = [
            k for k, v in avail_query_methods.items() if v == chosen_query][0]
        try:
            result = chosen_query.query(user_input, proxy=proxy)
            success = True
        except Exception as e:
            print(f"Error with {chosen_query_name}: {e}")
            query_methods_list.remove(chosen_query)

    return result
