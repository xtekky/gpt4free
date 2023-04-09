import quora
from time import sleep

token = quora.Account.create(proxy = 'xtekky:ogingoi2n3g@geo.iproyal.com:12321',logging = True)
print('token', token)

sleep(2)

for response in quora.StreamingCompletion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token):
    
    print(response.completion.choices[0].text, end="", flush=True)