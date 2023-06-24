import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from gpt4free import quora, forefront, theb, you
import random


def query_forefront(question: str, proxy: Optional[str] = None) -> str:
    # create an account
    token = forefront.Account.create(logging=False, proxy=proxy)

    response = ""
    # get a response
    try:
        return forefront.Completion.create(token=token, prompt='hello world', model='gpt-4', proxy=proxy).text
    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )


def query_quora(question: str, proxy: Optional[str] = None) -> str:
    token = quora.Account.create(logging=False, enable_bot_creation=True, proxy=proxy)
    return quora.Completion.create(model='gpt-4', prompt=question, token=token, proxy=proxy).text


def query_theb(question: str, proxy: Optional[str] = None) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    response = ""
    try:
        return ''.join(theb.Completion.create(prompt=question, proxy=proxy))

    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )


def query_you(question: str, proxy: Optional[str] = None) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = you.Completion.create(prompt=question, proxy=proxy)
        return result.text

    except Exception as e:
        # Return error message if an exception occurs
        return (
            f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'
        )


# Define a dictionary containing all query methods
avail_query_methods = {
    "Forefront": query_forefront,
    "Poe": query_quora,
    "Theb": query_theb,
    "You": query_you,
    # "Writesonic": query_writesonic,
    # "T3nsor": query_t3nsor,
    # "Phind": query_phind,
    # "Ora": query_ora,
}


def query(user_input: str, selected_method: str = "Random", proxy: Optional[str] = None) -> str:
    # If a specific query method is selected (not "Random") and the method is in the dictionary, try to call it
    if selected_method != "Random" and selected_method in avail_query_methods:
        try:
            return avail_query_methods[selected_method](user_input, proxy=proxy)
        except Exception as e:
            print(f"Error with {selected_method}: {e}")
            return "😵 Sorry, some error occurred please try again."

    # Initialize variables for determining success and storing the result
    success = False
    result = "😵 Sorry, some error occurred please try again."
    # Create a list of available query methods
    query_methods_list = list(avail_query_methods.values())

    # Continue trying different methods until a successful result is obtained or all methods have been tried
    while not success and query_methods_list:
        # Choose a random method from the list
        chosen_query = random.choice(query_methods_list)
        # Find the name of the chosen method
        chosen_query_name = [k for k, v in avail_query_methods.items() if v == chosen_query][0]
        try:
            # Try to call the chosen method with the user input
            result = chosen_query(user_input, proxy=proxy)
            success = True
        except Exception as e:
            print(f"Error with {chosen_query_name}: {e}")
            # Remove the failed method from the list of available methods
            query_methods_list.remove(chosen_query)

    return result
