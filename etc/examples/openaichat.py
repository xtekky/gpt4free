from g4f.client   import Client
from g4f.Provider import OpenaiChat, RetryProvider

# compatible countries: https://pastebin.com/UK0gT9cn
client = Client(
    proxies = {
        'http': 'http://username:password@host:port', # MUST BE WORKING OPENAI COUNTRY PROXY ex: USA
        'https': 'http://username:password@host:port' # MUST BE WORKING OPENAI COUNTRY PROXY ex: USA
    },
    provider = RetryProvider([OpenaiChat],
                             single_provider_retry=True, max_retries=5)
)

messages = [
    {'role': 'user', 'content': 'Hello'}
]

response = client.chat.completions.create(model='gpt-3.5-turbo',
                                     messages=messages, 
                                     stream=True)

for message in response:
    print(message.choices[0].delta.content or "")