import poe
from time import sleep

token = poe.Account.create(proxy = 'xtekky:ogingoi2n3g@geo.iproyal.com:12321',logging = True)
print('token', token)

sleep(2)

for response in poe.StreamingCompletion.create(model  = 'gpt-4',
    prompt = 'hello world',
    token  = token):
    
    print(response.completion.choices[0].text, end="", flush=True)