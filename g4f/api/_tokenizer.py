# import tiktoken
# from typing import Union

# def tokenize(text: str, model: str = 'gpt-3.5-turbo') -> Union[int, str]:
#     encoding   = tiktoken.encoding_for_model(model)
#     encoded    = encoding.encode(text)
#     num_tokens = len(encoded)
    
#     return num_tokens, encoded