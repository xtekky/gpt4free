import quora
from time import sleep

token = quora.Account.create(proxy = None,logging = True)
print('token', token)

sleep(2)

for response in quora.StreamingCompletion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token):
    
    print(response.completion.choices[0].text, end="", flush=True)