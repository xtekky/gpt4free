import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import forefront, quora, theb, you
import random



def query_forefront(question: str) -> str:
    # create an account
    token = forefront.Account.create(logging=False)

    response = ""
    # get a response
    try:
        for i in forefront.StreamingCompletion.create(token = token, prompt = 'hello world', model='gpt-4'):
            response += i.completion.choices[0].text
        
        return response
    
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


def query_quora(question: str) -> str:
    token = quora.Account.create(logging=False, enable_bot_creation=True)
    response = quora.Completion.create(
        model='gpt-4',
        prompt=question,
        token=token
    )

    return response.completion.choices[0].tex


def query_theb(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    response = ""
    try:
        result = theb.Completion.create(
            prompt = question)
        return result
    
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


def query_you(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = you.Completion.create(
            prompt = question)
        return result["response"]
    
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'

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

def query(user_input: str, selected_method: str = "Random") -> str:

    # If a specific query method is selected (not "Random") and the method is in the dictionary, try to call it
    if selected_method != "Random" and selected_method in avail_query_methods:
        try:
            return avail_query_methods[selected_method](user_input)
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
            result = chosen_query(user_input)
            success = True
        except Exception as e:
            print(f"Error with {chosen_query_name}: {e}")
            # Remove the failed method from the list of available methods
            query_methods_list.remove(chosen_query)

    return result

