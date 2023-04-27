import forefront, quora, theb, you
import random



def query_forefront(question: str) -> str:
    # create an account
    token = forefront.Account.create(logging=True)
    
    # get a response
    try:
        result = forefront.StreamingCompletion.create(token = token, prompt = 'hello world', model='gpt-4')
        
        return result['response']
    
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
    try:
        result = theb.Completion.create(
            prompt = question)
        
        return result['response']
    
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'


def query_you(question: str) -> str:
    # Set cloudflare clearance cookie and get answer from GPT-4 model
    try:
        result = you.Completion.create(
            prompt = question)
        
        return result['response']
    
    except Exception as e:
        # Return error message if an exception occurs
        return f'An error occurred: {e}. Please make sure you are using a valid cloudflare clearance token and user agent.'

# Define a dictionary containing all query methods
avail_query_methods = {
    "Forefront": query_forefront,
    "Quora": query_quora,
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


__all__ = ['query', 'avail_query_methods']



# def query_ora(question:str)->str:
#     result =""
#     try:
#         gpt4_chatbot_ids = ['b8b12eaa-5d47-44d3-92a6-4d706f2bcacf', 'fbe53266-673c-4b70-9d2d-d247785ccd91', 'bd5781cf-727a-45e9-80fd-a3cfce1350c6', '993a0102-d397-47f6-98c3-2587f2c9ec3a', 'ae5c524e-d025-478b-ad46-8843a5745261', 'cc510743-e4ab-485e-9191-76960ecb6040', 'a5cd2481-8e24-4938-aa25-8e26d6233390', '6bca5930-2aa1-4bf4-96a7-bea4d32dcdac', '884a5f2b-47a2-47a5-9e0f-851bbe76b57c', 'd5f3c491-0e74-4ef7-bdca-b7d27c59e6b3', 'd72e83f6-ef4e-4702-844f-cf4bd432eef7', '6e80b170-11ed-4f1a-b992-fd04d7a9e78c', '8ef52d68-1b01-466f-bfbf-f25c13ff4a72', 'd0674e11-f22e-406b-98bc-c1ba8564f749', 'a051381d-6530-463f-be68-020afddf6a8f', '99c0afa1-9e32-4566-8909-f4ef9ac06226', '1be65282-9c59-4a96-99f8-d225059d9001', 'dba16bd8-5785-4248-a8e9-b5d1ecbfdd60', '1731450d-3226-42d0-b41c-4129fe009524', '8e74635d-000e-4819-ab2c-4e986b7a0f48', 'afe7ed01-c1ac-4129-9c71-2ca7f3800b30', 'e374c37a-8c44-4f0e-9e9f-1ad4609f24f5']
#         chatbot_id = random.choice(gpt4_chatbot_ids)
#         model = ora.CompletionModel.load(chatbot_id, 'gpt-4')
#         response = ora.Completion.create(model, question)
#         result = response.completion.choices[0].text
#     except  Exception as e:
#         print(f"Error : {e}")
#         result = "😵 Sorry, some error occurred please try again."
#     return result


# def query_writesonic(question:str)->str:
#     account = writesonic.Account.create(logging = False)
#     response = writesonic.Completion.create(
#         api_key = account.key,
#         prompt  = question,
#     )
    
#     return response.completion.choices[0].text


# def query_t3nsor(question: str) -> str:
#     messages = []
    
#     user = question

#     t3nsor_cmpl = t3nsor.Completion.create(
#         prompt=user,
#         messages=messages
#     )

#     messages.extend([
#         {'role': 'user', 'content': user},
#         {'role': 'assistant', 'content': t3nsor_cmpl.completion.choices[0].text}
#     ])

#     return t3nsor_cmpl.completion.choices[0].text



# def query_phind(question:str)->str:
#     phind.cf_clearance = 'KvXc1rh.TFQG1rNF0eMlcpJbsdmJkYgvmqS42OOfqUk-1682393898-0-160'
#     # phind.cf_clearance = 'heguhSRBB9d0sjLvGbQECS8b80m2BQ31xEmk9ChshKI-1682268995-0-160'
#     # phind.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
#     phind.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4.1 Safari/605.1.15'
#     result = phind.Completion.create(
#     model  = 'gpt-4',
#     prompt = question,
#     results     = phind.Search.create(question, actualSearch = False),
#     creative    = False,
#     detailed    = False,
#     codeContext = '') 
#     # print(result.completion.choices[0].text)
#     return result.completion.choices[0].text
