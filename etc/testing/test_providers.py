from g4f.Provider import __all__, ProviderUtils
from g4f import ChatCompletion
import concurrent.futures

_ = [
    'BaseProvider',
    'AsyncProvider',
    'AsyncGeneratorProvider',
    'RetryProvider'
]

def test_provider(provider):
    try:
        provider = (ProviderUtils.convert[provider])
        if provider.working and not provider.needs_auth:
            print('testing', provider.__name__)
            completion = ChatCompletion.create(model='gpt-3.5-turbo', 
                                            messages=[{"role": "user", "content": "hello"}], provider=provider)
            return completion, provider.__name__
    except Exception as e:
        #print(f'Failed to test provider: {provider} | {e}')
        return None

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for provider in __all__:
        if provider not in _:
            futures.append(executor.submit(test_provider, provider))
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            print(f'{result[1]} | {result[0]}')